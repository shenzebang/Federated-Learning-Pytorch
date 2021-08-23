import torch
import torch.optim as optim
import copy
from api import FedAlgorithm
from utils import weighted_sum_functions
from collections import namedtuple
from typing import List
import ray

FEDAVG_server_state = namedtuple("FEDAVG_server_state", ['global_round', 'model'])
FEDAVG_client_state = namedtuple("FEDAVG_client_state", ['global_round', 'model'])


class FEDAVG(FedAlgorithm):
    def __init__(self, model,
                 client_dataloaders,
                 loss,
                 test_fn,
                 logger,
                 config,
                 device
                 ):
        super(FEDAVG, self).__init__(model, client_dataloaders, loss, test_fn, logger, config, device)
        if self.config.use_ray:
            ray.init()

    def server_init(self):
        return FEDAVG_server_state(global_round=0, model=self.model)

    def client_init(self, server_state: FEDAVG_server_state, client_dataloader):
        return FEDAVG_client_state(global_round=server_state.global_round, model=server_state.model)

    def clients_step(self, clients_state, active_ids):

        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [_client_step(self.config, self.loss, self.device, client_state, client_dataloader)
                    for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get([client_step.remote(self.config, self.loss, self.device, client_state, client_dataloader)
                            for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: FEDAVG_server_state, client_states: FEDAVG_client_state, weights, active_ids):
        # todo: add the implementation for non-uniform weight
        active_clients = [client_states[i] for i in active_ids]

        active_weights = None if weights is None else [weights[i] for i in active_ids]

        new_model = weighted_sum_functions([client_state.model for client_state in active_clients], active_weights)
        new_server_state = FEDAVG_server_state(
            global_round=server_state.global_round + 1,
            model=weighted_sum_functions([new_model, server_state.model], [self.config.global_lr, (1-self.config.global_lr)])
        )
        return new_server_state

    def clients_update(self, server_state: FEDAVG_server_state, clients_state: List[FEDAVG_client_state], active_ids):
        return [FEDAVG_client_state(global_round=server_state.global_round, model=server_state.model) for _ in clients_state]


@ray.remote(num_gpus=.3)
def client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    lr_decay = 1.
    if client_state.global_round >= 1000:
        lr_decay = .1
    elif client_state.global_round >= 1500:
        lr_decay = .01
    optimizer = optim.SGD(f_local.parameters(), lr=lr_decay*config.local_lr)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            optimizer.step()

    return FEDAVG_client_state(global_round=client_state.global_round, model=f_local)


def _client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    lr_decay = 1.
    if client_state.global_round >= 100:
        lr_decay = .1
    elif client_state.global_round >= 200:
        lr_decay = .01
    optimizer = optim.SGD(f_local.parameters(), lr=lr_decay*config.local_lr)
    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            optimizer.step()

    return FEDAVG_client_state(global_round=client_state.global_round, model=f_local)
