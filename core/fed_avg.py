import torch
import torch.optim as optim
import copy
from api import FedAlgorithm
from utils import average_functions
from collections import namedtuple
from typing import List
import ray

FEDAVG_server_state = namedtuple("FEDAVG_server_state", ['global_round', 'model'])
FEDAVG_client_state = namedtuple("FEDAVG_client_state", ['model'])


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
        return FEDAVG_client_state(model=server_state.model)

    def clients_step(self, clients_state):
        if not self.config.use_ray:
            return [_client_step(self.config, self.loss, self.device, client_state, client_dataloader)
                    for client_state, client_dataloader in zip(clients_state, self.client_dataloaders)]
        else:
            return ray.get([client_step.remote(self.config, self.loss, self.device, client_state, client_dataloader)
                           for client_state, client_dataloader in zip(clients_state, self.client_dataloaders)])

    def server_step(self, server_state: FEDAVG_server_state, client_states: FEDAVG_client_state, weights):
        # todo: add the implementation for non-uniform weight
        new_server_state = FEDAVG_server_state(
            global_round=server_state.global_round + 1,
            model=average_functions([client_state.model for client_state in client_states])
        )
        return new_server_state

    def clients_update(self, server_state: FEDAVG_server_state, clients_state: List[FEDAVG_client_state]):
        return [FEDAVG_client_state(model=server_state.model) for _ in clients_state]


@ray.remote(num_gpus=.5)
def client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    optimizer = optim.SGD(f_local.parameters(), lr=config.local_lr)
    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            optimizer.step()

    return FEDAVG_client_state(model=f_local)

def _client_step(config, loss_fn, device, client_state: FEDAVG_client_state, client_dataloader):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)
    optimizer = optim.SGD(f_local.parameters(), lr=config.local_lr)
    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            optimizer.step()

    return FEDAVG_client_state(model=f_local)
