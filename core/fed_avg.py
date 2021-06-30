import torch
import torch.optim as optim
import copy
from api import FedAlgorithm
from utils import average_functions
from collections import namedtuple

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

    def server_init(self):
        return FEDAVG_server_state(global_round=0, model=self.model)

    def client_init(self, server_state: FEDAVG_server_state, client_dataloader):
        return FEDAVG_client_state(model=server_state.model)

    def client_step(self, client_state: FEDAVG_client_state, client_dataloader):
        f_local = copy.deepcopy(client_state.model)
        f_local.requires_grad_(True)
        optimizer = optim.SGD(f_local.parameters(), lr=self.config.local_lr)
        for epoch in range(self.config.local_epoch):
            for data, label in client_dataloader:
                optimizer.zero_grad()
                data = data.to(self.device)
                label = label.to(self.device)
                loss = self.loss(f_local(data), label)
                loss.backward()
                optimizer.step()

        return f_local

    def server_step(self, server_state: FEDAVG_server_state, client_states: FEDAVG_client_state, weights):
        # todo: add the implementation for non-uniform weight
        new_server_state = FEDAVG_server_state(
            global_round=server_state.global_round + 1,
            model=average_functions([client_state.model for client_state in client_states])
        )
        return new_server_state

    def client_update(self, server_state: FEDAVG_server_state, client_state: FEDAVG_client_state, client_dataloader):
        return FEDAVG_client_state(model=server_state.model)

