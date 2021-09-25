import torch
import copy
from api import FedAlgorithm
from utils.general_utils import weighted_sum_functions, compute_model_delta
from collections import namedtuple
from typing import List
import ray

SCAFFOLD_server_state = namedtuple("SCAFFOLD_server_state", ['global_round', 'model', 'c'])
SCAFFOLD_client_state = namedtuple("SCAFFOLD_client_state",
                                   ['global_round', 'model', 'model_delta', 'c_i', 'c_i_delta', 'c'])


class SCAFFOLD(FedAlgorithm):
    def __init__(self, init_model,
                 client_dataloaders,
                 loss,
                 loggers,
                 config,
                 device
                 ):
        super(SCAFFOLD, self).__init__(init_model, client_dataloaders, loss, loggers, config, device)
        self.n_workers = config.n_workers
        self.n_workers_per_round = config.n_workers_per_round
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return SCAFFOLD_server_state(global_round=0, model=init_model,
                                     c=tuple(torch.zeros_like(p) for p in init_model.parameters())
                                     )

    def client_init(self, server_state: SCAFFOLD_server_state, client_dataloader):
        return SCAFFOLD_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None,
                                     c_i=server_state.c,
                                     c_i_delta=None, c=server_state.c)

    def clients_step(self, clients_state, active_ids):
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [
                client_step(self.config, self.loss, self.device, client_state, client_dataloader)
                for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [ray_dispatch.remote(self.config, self.loss, self.device, client_state, client_dataloader)
                 for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: SCAFFOLD_server_state, client_states: SCAFFOLD_client_state, weights,
                    active_ids):
        # todo: implement the weighted version
        active_clients = [client_states[i] for i in active_ids]
        c_delta = []
        cc = [client_state.c_i_delta for client_state in active_clients]
        for ind in range(len(server_state.c)):
            # handles the int64 and float data types jointly
            c_delta.append(
                torch.mean(torch.stack([c_i_delta[ind].float() for c_i_delta in cc]), dim=0).to(server_state.c[ind].dtype)
                    )
        c_delta = tuple(c_delta)
        c = []
        for param_1, param_2 in zip(server_state.c, c_delta):
            c.append(param_1 + param_2 * self.config.n_workers_per_round / self.n_workers)
        c = tuple(c)

        new_server_state = SCAFFOLD_server_state(
            global_round=server_state.global_round + 1,
            model=weighted_sum_functions(
                [server_state.model] +
                [client_state.model_delta for client_state in active_clients],
                [1.] +
                [self.config.global_lr / self.n_workers_per_round] * self.n_workers_per_round
            ),
            c=c
        )
        return new_server_state

    def clients_update(self, server_state: SCAFFOLD_server_state, clients_state: List[SCAFFOLD_client_state],
                       active_ids):
        # c_i is updated in client_step
        return [
            SCAFFOLD_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None,
                                  c_i=client.c_i, c_i_delta=None, c=server_state.c)
            for client in clients_state]


@ray.remote(num_gpus=.25)
def ray_dispatch(config, loss_fn, device, client_state: SCAFFOLD_client_state, client_dataloader):
    return client_step(config, loss_fn, device, client_state, client_dataloader)


def client_step(config, loss_fn, device, client_state: SCAFFOLD_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    f_initial = client_state.model

    optimizer = torch.optim.SGD(f_local.parameters(), lr=config.local_lr, weight_decay=config.weight_decay)

    # c_i - c
    c_i_c = tuple(param_1 - param_2 for param_1, param_2 in zip(client_state.c_i, client_state.c))

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            # add the linear term:
            loss_linear = 0.
            for param_1, param_2 in zip(f_local.parameters(), c_i_c):
                loss_linear -= torch.sum(param_1 * param_2)

            loss += loss_linear

            loss.backward()
            if config.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(f_local.parameters(), config.gradient_clip_constant)
            optimizer.step()

    # print(loss.item())
    # Update the auxiliary variable

    with torch.autograd.no_grad():
        model_delta = compute_model_delta(f_local, f_initial)
        new_c_i = []
        for param_1, param_2 in zip(c_i_c, model_delta.parameters()):
            new_c_i.append(
                param_1 - param_2 / config.local_lr / config.local_epoch / config.client_step_per_epoch)
        new_c_i = tuple(new_c_i)
        c_i_delta = []
        for param_1, param_2 in zip(new_c_i, client_state.c_i):
            c_i_delta.append(param_1 - param_2)
        c_i_delta = tuple(c_i_delta)

    # no need to return f_local and c
    return SCAFFOLD_client_state(global_round=client_state.global_round, model=None, model_delta=model_delta,
                                 c_i=new_c_i, c_i_delta=c_i_delta, c=None)
