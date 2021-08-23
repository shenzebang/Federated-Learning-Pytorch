import torch
from tqdm import trange


class FedAlgorithm(object):
    def __init__(self,
                 model,
                 client_dataloaders,
                 loss,
                 test_fn,
                 logger,
                 config,
                 device
                 ):
        self.model = model
        self.client_dataloaders = client_dataloaders
        self.loss = loss
        self.test_fn = test_fn
        self.logger = logger
        self.config = config
        self.device = device

    def step(self, server_state, client_states, weights):
        # server_state contains the (global) model, (global) auxiliary variables, weights of clients
        # client_states contain the (local) auxiliary variables

        # sample active clients
        active_ids = torch.randperm(self.config.n_workers)[:self.config.n_workers_per_round].tolist()

        client_states = self.clients_step(client_states, active_ids)

        # aggregate
        server_state = self.server_step(server_state, client_states, weights, active_ids)

        # broadcast
        client_states = self.clients_update(server_state, client_states, active_ids)

        return server_state, client_states

    def fit(self, weights):
        # if weights is None, use uniform weights among clients

        server_state = self.server_init()
        client_states = [self.client_init(server_state, client_dataloader) for client_dataloader in
                         self.client_dataloaders]
        for round in trange(self.config.n_global_rounds):
            server_state, client_states = self.step(server_state, client_states, weights)
            if round % self.config.eval_freq == 0:
                metric = self.test_fn(server_state.model)
                self.logger.log(round, metric)

    def server_init(self):
        raise NotImplementedError

    def client_init(self, server_state, client_dataloader):
        raise NotImplementedError

    def clients_step(self, client_state, active_ids):
        raise NotImplementedError

    def server_step(self, server_state, client_states, weights, active_ids):
        raise NotImplementedError

    def clients_update(self, server_state, client_state, active_ids):
        raise NotImplementedError
