import torch
import torch.optim as optim
import copy
from api import FedAlgorithm
from utils import weighted_sum_functions, compute_model_delta
from collections import namedtuple
from typing import List
import ray

FEDAVG_server_state = namedtuple("FEDAVG_server_state", ['global_round', 'model'])
FEDAVG_client_state = namedtuple("FEDAVG_client_state", ['global_round', 'model', 'model_delta'])


class FEDAVG(FedAlgorithm):
    def __init__(self, init_model,
                 client_dataloaders,
                 loss,
                 loggers,
                 config,
                 device
                 ):
        super(FEDAVG, self).__init__(init_model, client_dataloaders, loss, loggers, config, device)
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDAVG_server_state(global_round=0, model=init_model)

    def client_init(self, server_state: FEDAVG_server_state, client_dataloader):
        return FEDAVG_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None)

    def clients_step(self, clients_state, active_ids):

        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [client_step(self.config, self.loss, self.device, client_state, client_dataloader)
                    for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get([ray_dispatch.remote(self.config, self.loss, self.device, client_state, client_dataloader)
                            for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: FEDAVG_server_state, client_states: List[FEDAVG_client_state], weights, active_ids):
        active_clients = [client_states[i] for i in active_ids]

        new_model = weighted_sum_functions([client_state.model_delta for client_state in active_clients] +
                                           [server_state.model],
                                           [weights[i] * self.config.global_lr / len(active_ids) for i in active_ids] +
                                           [1.])
        # new_model = weighted_sum_functions([client.model for client in active_clients], active_weights)
        new_server_state = FEDAVG_server_state(
            global_round=server_state.global_round + 1,
            model=new_model
        )
        return new_server_state

    def clients_update(self, server_state: FEDAVG_server_state, clients_state: List[FEDAVG_client_state], active_ids):
        return [FEDAVG_client_state(global_round=server_state.global_round, model=server_state.model, model_delta=None) for _ in clients_state]

@ray.remote(num_gpus=.14)
def ray_dispatch(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    return client_step(config, loss_fn, device, client_state, client_dataloader)

def client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    lr_decay = 1.
    # if client_state.global_round >= 1000:
    #     lr_decay = .1
    # elif client_state.global_round >= 1500:
    #     lr_decay = .01
    optimizer = optim.SGD(f_local.parameters(), lr=lr_decay*config.local_lr, weight_decay=config.weight_decay)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            if config.l2_reg > 0:
                l2_norm = torch.norm(torch.stack([torch.norm(param) for param in f_local.parameters()]))
                loss += .5 * config.l2_reg * l2_norm ** 2
            loss.backward()
            optimizer.step()

    model_delta = compute_model_delta(f_local, client_state.model)
    if config.use_gradient_clip:
        model_delta = clip_model_delta(model_delta, config.gradient_clip_constant)
    # no need to return f_local
    return FEDAVG_client_state(global_round=client_state.global_round, model=None, model_delta=model_delta)


# def _client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
#     f_local = copy.deepcopy(client_state.model)
#     f_local.requires_grad_(True)
#     lr_decay = 1.
#     # if client_state.global_round >= 1000:
#     #     lr_decay = .1
#     # elif client_state.global_round >= 1500:
#     #     lr_decay = .01
#     optimizer = optim.SGD(f_local.parameters(), lr=lr_decay*config.local_lr)
#     for epoch in range(config.local_epoch):
#         for data, label in client_dataloader:
#             optimizer.zero_grad()
#             data = data.to(device)
#             label = label.to(device)
#             loss = loss_fn(f_local(data), label)
#             if config.l2_reg > 0:
#                 l2_norm = torch.norm(torch.stack([torch.norm(param) for param in f_local.parameters()]))
#                 loss += .5 * config.l2_reg * l2_norm ** 2
#             loss.backward()
#             optimizer.step()
#
#     model_delta = compute_model_delta(f_local, client_state.model)
#     if config.use_gradient_clip:
#         model_delta = clip_model_delta(model_delta, config.gradient_clip_constant)
#
#     # no need to return f_local
#     return FEDAVG_client_state(global_round=client_state.global_round, model=None, model_delta=model_delta)


def clip_model_delta(model_delta, threshold=1.):
    sd = model_delta.state_dict()
    total_norm = torch.norm(torch.stack([torch.norm(sd[key]) for key in sd])).item()
    clip_coef = threshold / (total_norm + 1e-6)
    if clip_coef < 1:
        for key in sd:
            sd[key].mul_(clip_coef)
    model_delta.load_state_dict(sd)
    return model_delta
