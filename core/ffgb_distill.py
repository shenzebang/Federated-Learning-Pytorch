from api import FedAlgorithm
import ray
from collections import namedtuple
from utils.model_utils import FunctionEnsemble, Residual
import torch
from torch.optim import Adam
import copy
from tqdm import trange
from utils.data_utils import LocalDataset
from torch.utils.data.dataloader import DataLoader
FFGB_D_server_state = namedtuple("FFGB_D_server_state", ['global_round', 'model'])
FFGB_D_client_state = namedtuple("FFGB_D_client_state", ['global_round', 'model', 'model_delta'])

oracle_config = namedtuple("oracle_config", ['epoch', 'weight_decay', 'lr'])

class FFGB_D(FedAlgorithm):
    def __init__(self,
                 init_model,
                 make_model,
                 client_dataloaders,
                 distill_dataloder,
                 Dx_loss,
                 loggers,
                 config,
                 device):
        super(FFGB_D, self).__init__(init_model,
                                     client_dataloaders,
                                     None,
                                     loggers,
                                     config,
                                     device)
        self.distill_dataloder = distill_dataloder
        self.make_model = make_model
        self.Dx_loss = Dx_loss
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FFGB_D_server_state(global_round=1, model=init_model)

    def client_init(self, server_state, client_dataloader):
        return FFGB_D_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None)

    def clients_step(self, clients_state, weights, active_ids):
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [client_step(self.config, self.make_model, self.Dx_loss, client_state, client_dataloader, self.device)
                                 for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [ray_dispatch.remote(self.config, self.make_model, self.Dx_loss, client_state, client_dataloader, self.device)
                for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state, client_states, weights, active_ids):
        print("#"*30)
        print("start server step")
        active_clients = [client_states[i] for i in active_ids]
        new_model = copy.deepcopy(server_state.model)
        f = FunctionEnsemble()
        for client_state in active_clients:
            f.add_ensemble(client_state.model_delta)
        f.rescale_weights(1./len(active_clients))
        f.add_function(server_state.model, 1.)

        distill_config = oracle_config(
            epoch=self.config.distill_oracle_epoch,
            weight_decay=self.config.distill_oracle_weight_decay,
            lr=self.config.distill_oracle_lr
        )

        if self.config.distill_oracle == "kl":
            oracle = kl_oracle
            target = f
        elif self.config.distill_oracle == "l2":
            oracle = l2_oracle
            target = lambda data, label: f(data)
        else:
            return NotImplementedError


        if 'emnist-digit' == self.config.dataset_distill or 'emnist-letter' == self.config.dataset_distill:
            # create a new dataloader from self.distill_dataloader since there is no data augmentation step
            # the new dataloader returns (data, target(data)) pairs
            # this avoids the repetitive evaluation of target(data)
            print('create a new dataloader for efficient knowledge distillation')
            distill_dataloader = new_dataloader_from_target(target, self.distill_dataloder, self.device)
            # generate new_model
            new_model = oracle_from_dataloader(distill_config, new_model, distill_dataloader, self.device)
        else:
            new_model = oracle(distill_config, target, new_model, self.distill_dataloder, self.device)

        new_server_state = FFGB_D_server_state(
            global_round=server_state.global_round + 1,
            model=new_model
        )
        return new_server_state

    def clients_update(self, server_state, clients_state, active_ids):
        return [FFGB_D_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None) for client_state in clients_state]


def new_dataloader_from_target(target, distill_dataloder, device):
    labels = []
    # distill_dataloder.shuffle = False
    for data, _ in distill_dataloder:
        labels.append(target(data.to(device), None))

    labels = torch.cat(labels).to('cpu')
    new_dataset = LocalDataset(distill_dataloder.dataset.data.numpy(),
                               labels, train=True, transform=distill_dataloder.dataset.transform)
    return DataLoader(new_dataset, distill_dataloder.batch_size, shuffle=True, num_workers=distill_dataloder.num_workers)



@ray.remote(num_gpus=.15)
def ray_dispatch(config, make_model, Dx_loss_fn, client_state: FFGB_D_client_state, client_dataloader, device):
    return client_step(config, make_model, Dx_loss_fn, client_state, client_dataloader, device)


def client_step(config, make_model, Dx_loss_fn, client_state: FFGB_D_client_state, client_dataloader, device):
    f_inc = FunctionEnsemble()
    residual = Residual()
    # assert(config.local_steps == 1)


    # func_grad = lambda data, label: Dx_loss_fn(client_state.model(data), label)
    # for local_iter in range(config.local_steps):
    #     target = lambda data, label: func_grad(data, label) - f_inc(data)
    #     h = l2_oracle(config, target, make_model(), client_dataloader, device)
    #     f_inc.add_function(h, 1.)
    #
    # lr = config.functional_lr_0 if client_state.global_round == 1 else config.functional_lr
    # f_inc.rescale_weights(-lr)

    for local_iter in range(config.local_steps):
        def func_grad(data, label):
            f_data = client_state.model(data) + f_inc(data)
            return Dx_loss_fn(f_data, label) + config.f_l2_reg * f_data

        weak_learner_config = oracle_config(
            epoch=config.weak_learner_epoch,
            weight_decay=config.weak_learner_weight_decay,
            lr=config.weak_learner_lr
        )
        h = l2_oracle(weak_learner_config, func_grad, make_model(), client_dataloader, device)
        f_inc.add_function(h, -config.functional_lr)


    # for local_iter in range(config.local_steps):
    #     f = FunctionEnsemble()
    #     f.add_function(client_state.model, 1.)
    #     f.add_ensemble(f_inc)
    #     func_grad = lambda data, label: Dx_loss_fn(f(data), label)
    #     target = lambda data, label: func_grad(data, label) + residual(data, label)
    #     h = l2_oracle(config, target, make_model(), client_dataloader, device)
    #     lr = config.functional_lr_0 if client_state.global_round == 1 else config.functional_lr
    #     f_inc.add_function(h, -lr)
    #     residual.add(func_grad, h)
        # line search to fine an appropriate step size
        # ls_oracle(config, client_state.model, f_inc, client_dataloader, device)

        # check kl for debugging
        # if config.debug:
        #     check_kl(config, client_state.model, f_inc, client_dataloader, device)
    # print(f"local loss on client {client_state.id} at start {check_loss(client_state.model, client_dataloader, device)}")
    # f = FunctionEnsemble()
    # f.add_function(client_state.model, 1.)
    # f.add_ensemble(f_inc)
    # print(f"local loss on client {client_state.id} in the end {check_loss(f, client_dataloader, device)}")
    return FFGB_D_client_state(global_round=client_state.global_round, model=None, model_delta=f_inc)

def check_loss(f, dataloader, device):
    with torch.autograd.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_average = torch.zeros(1)
        n_sample = 0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            f_data = f(data)
            n_sample += data.shape[0]
            loss_average += loss_fn(f_data, label) * data.shape[0]
        loss_average /= n_sample
    return loss_average.item()


def check_kl(config, f_0, f_inc: FunctionEnsemble, dataloader, device):
    kl = lambda p, q: torch.mean(torch.sum(p * (p.log() - q.log()), dim=1))
    softmax = torch.nn.Softmax(dim=1)
    with torch.autograd.no_grad():
        kl_average = torch.zeros(1)
        n_sample = 0
        for data, _ in dataloader:
            data = data.to(device)
            f_0_data = f_0(data)
            f_inc_data = f_inc(data)
            n_sample += data.shape[0]
            kl_average += kl(softmax(f_0_data), softmax(f_inc_data + f_0_data))
        kl_average /= n_sample
        print(kl_average.item())


def ls_oracle(config, f_0, f_inc: FunctionEnsemble, dataloader, device):
    # line search oracle
    kl = lambda p, q: torch.sum(torch.sum(p * (p.log() - q.log()), dim=1))
    softmax = torch.nn.Softmax(dim=1)
    with torch.autograd.no_grad():
        scale = 1.
        while True:
            kl_average = 0
            n_sample = 0
            for data, _ in dataloader:
                data = data.to(device)
                n_sample += data.shape[0]
                f_0_data = f_0(data)
                f_inc_data = f_inc(data)
                kl_average += kl(softmax(f_0_data), softmax(f_inc_data + scale*f_0_data))
            kl_average /= n_sample
            if kl_average <= config.max_kl:
                break
            else:
                scale *= .5
        f_inc.rescale_weight(scale)


def l2_oracle(config: oracle_config, target, h, dataloader, device):
    h.requires_grad_(True)
    optimizer = Adam(h.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse_loss = torch.nn.MSELoss()
    epoch_loss_0 = 0.
    for epoch in range(config.epoch):
        epoch_loss = 0.
        for data, label in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = mse_loss(target(data, label).detach(), h(data))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=h.parameters(),
                                           max_norm=5)
            optimizer.step()

            epoch_loss += loss.item()
        if epoch == 0:
            epoch_loss_0 = epoch_loss
    # print(f"epoch_loss_0: {epoch_loss_0}, epoch_loss: {epoch_loss}")
    h.requires_grad_(False)
    return h

def oracle_from_dataloader(config: oracle_config, h, dataloader, device):
    h.requires_grad_(True)
    optimizer = Adam(h.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse_loss = torch.nn.MSELoss()
    epoch_loss_0 = 0.
    for epoch in range(config.epoch):
        epoch_loss = 0.
        for data, label in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = mse_loss(label, h(data))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=h.parameters(),
                                           max_norm=5)
            optimizer.step()

            epoch_loss += loss.item()
        if epoch == 0:
            epoch_loss_0 = epoch_loss
    # print(f"epoch_loss_0: {epoch_loss_0}, epoch_loss: {epoch_loss}")
    h.requires_grad_(False)
    return h

def kl_oracle(config: oracle_config, f_0: FunctionEnsemble, h, dataloader, device):
    h.requires_grad_(True)

    optimizer = Adam(h.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    kl_loss = lambda p, q: torch.mean(torch.sum(p*(p.log() - q.log()), dim=1))
    softmax = torch.nn.Softmax(dim=1)

    for _ in trange(config.epoch):
        kl_1 = 0.
        for data, _ in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            loss = kl_loss(softmax(f_0(data)), softmax(h(data)))
            kl_1 += loss.item() * data.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=h.parameters(),
                                           max_norm=5)
            optimizer.step()

    h.requires_grad_(False)

    return h