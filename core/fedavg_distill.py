from api import FedAlgorithm
import ray
from collections import namedtuple
from utils.model_utils import FunctionEnsemble
import torch
from torch.optim import SGD
import copy
from core.ffgb_distill import kl_oracle, oracle_config, l2_oracle, new_dataloader_from_target, oracle_from_dataloader
FEDAVG_D_server_state = namedtuple("FEDAVG_D_server_state", ['global_round', 'model'])
FEDAVG_D_client_state = namedtuple("FEDAVG_D_client_state", ['global_round', 'model'])



class FEDAVG_D(FedAlgorithm):
    def __init__(self,
                 init_model,
                 make_model,
                 client_dataloaders,
                 distill_dataloader,
                 Dx_loss,
                 loggers,
                 config,
                 device):
        super(FEDAVG_D, self).__init__(init_model,
                                     client_dataloaders,
                                     None,
                                     loggers,
                                     config,
                                     device)
        self.distill_dataloader = distill_dataloader
        self.make_model = make_model
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDAVG_D_server_state(global_round=1, model=init_model)

    def client_init(self, server_state, client_dataloader):
        return FEDAVG_D_client_state(global_round=server_state.global_round, model=server_state.model)

    def clients_step(self, clients_state, weights, active_ids):
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [client_step(self.config, client_state, client_dataloader, self.device)
                                 for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [ray_dispatch.remote(self.config, client_state, client_dataloader, self.device)
                for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state, client_states, weights, active_ids):
        active_clients = [client_states[i] for i in active_ids]
        new_model = server_state.model
        f = FunctionEnsemble()
        for client_state in active_clients:
            f.add_function(client_state.model, 1./len(active_ids))

        distill_config = oracle_config(
            epoch=self.config.distill_oracle_epoch,
            weight_decay=self.config.distill_oracle_weight_decay,
            lr=self.config.distill_oracle_lr
        )


        if 'emnist-digit' == self.config.dataset_distill or 'emnist-letter' == self.config.dataset_distill:
            # create a new dataloader from self.distill_dataloader since there is no data augmentation step
            # the new dataloader returns (data, target(data)) pairs
            # this avoids the repetitive evaluation of target(data)
            print('create a new dataloader for efficient knowledge distillation')
            target = f
            distill_dataloader = new_dataloader_from_target(target, self.distill_dataloader, self.device)
            # generate new_model
            new_model = oracle_from_dataloader(distill_config, new_model, distill_dataloader, self.device)
        else:
            if self.config.distill_oracle == "kl":
                oracle = kl_oracle
                target = f
            elif self.config.distill_oracle == "l2":
                oracle = l2_oracle
                target = lambda data, label: f(data)
            else:
                return NotImplementedError
            new_model = oracle(distill_config, target, new_model, self.distill_dataloader, self.device)


        new_server_state = FEDAVG_D_server_state(
            global_round=server_state.global_round + 1,
            model=new_model
        )
        return new_server_state

    def clients_update(self, server_state, clients_state, active_ids):
        return [FEDAVG_D_client_state(global_round=server_state.global_round, model=server_state.model) for client_state in clients_state]

@ray.remote(num_gpus=.25)
def ray_dispatch(config, client_state: FEDAVG_D_client_state, client_dataloader, device):
    return client_step(config, client_state, client_dataloader, device)


def client_step(config, client_state: FEDAVG_D_client_state, client_dataloader, device):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)

    optimizer = SGD(f_local.parameters(), lr=config.fedavg_d_local_lr, weight_decay=config.fedavg_d_weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config.fedavg_d_local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=f_local.parameters(),
                                           max_norm=5)
            optimizer.step()
    # print(f"local loss on client {client_state.id} at start {check_loss(client_state.model, client_dataloader, device)}")
    # print(f"local loss on client {client_state.id} in the end {check_loss(f_local, client_dataloader, device)}")
    f_local.requires_grad_(False)

    return FEDAVG_D_client_state(global_round=client_state.global_round, model=f_local)


