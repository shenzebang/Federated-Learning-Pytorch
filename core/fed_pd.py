import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from api import FedAlgorithm
from utils import weighted_sum_functions
from collections import namedtuple
from typing import List
import ray
from torch.optim.optimizer import Optimizer



FEDPD_server_state = namedtuple("FEDPD_server_state", ['global_round', 'model'])
FEDPD_client_state = namedtuple("FEDPD_client_state", ['global_round', 'model', 'lambda_var', 'model_delta'])


class FEDPD(FedAlgorithm):
    def __init__(self, init_model,
                 client_dataloaders,
                 loss,
                 loggers,
                 config,
                 device
                 ):
        super(FEDPD, self).__init__(init_model, client_dataloaders, loss, loggers, config, device)
        self.eta = config.eta
        self.n_workers = config.n_workers
        self.n_workers_per_round = config.n_workers_per_round
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDPD_server_state(global_round=0, model=init_model)

    def client_init(self, server_state: FEDPD_server_state, client_dataloader):
        return FEDPD_client_state(global_round=server_state.global_round, model=server_state.model, lambda_var=None, model_delta=None)

    def clients_step(self, clients_state, active_ids):
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [
                client_step(self.config, self.loss, self.device, client_state, client_dataloader, self.eta)
                for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [ray_dispatch.remote(self.config, self.loss, self.device, client_state, client_dataloader, self.eta)
                 for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: FEDPD_server_state, client_states: FEDPD_client_state, weights, active_ids):
        # todo: implement the partial-participating version
        active_clients = [client_states[i] for i in active_ids]

        new_model = weighted_sum_functions([client_state.model_delta for client_state in active_clients] +
                                           [server_state.model],
                                           [weights[i] * self.config.global_lr / len(active_ids) for i in active_ids] +
                                           [1.])


        new_server_state = FEDPD_server_state(
            global_round=server_state.global_round + 1,
            model=new_model
        )
        return new_server_state

    def clients_update(self, server_state: FEDPD_server_state, clients_state: List[FEDPD_client_state], active_ids):
        return [FEDPD_client_state(global_round=server_state.global_round, model=server_state.model, lambda_var=client.lambda_var, model_delta=None)
                for client in clients_state]


def client_step(config, loss_fn, device, client_state: FEDPD_client_state, client_dataloader, eta):
    f_local = copy.deepcopy(client_state.model)
    f_initial = client_state.model
    f_local.requires_grad_(True)

    lr_decay = 1.
    optimizer = torch.optim.SGD(f_local.parameters(), lr=config.local_lr, weight_decay=config.weight_decay)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            if client_state.lambda_var is not None:
                linear_penalty = 0.
                for param_1, param_2 in zip(f_local.parameters(), client_state.lambda_var):
                    linear_penalty += torch.sum(param_1 * param_2)
                loss += linear_penalty

            quad_penalty = 0.0
            for theta, theta_init in zip(f_local.parameters(), f_initial.parameters()):
                quad_penalty += F.mse_loss(theta, theta_init, reduction='sum')

            loss += quad_penalty / 2. / eta

            # Now take loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=f_local.parameters(), max_norm=config.gradient_clip_constant) # Clip gradients

            optimizer.step()

    # Update the dual variable
    # print(loss.item())
    with torch.autograd.no_grad():

        lambda_delta = tuple(
            (param_1 - param_2) / eta * config.fed_pd_dual_lr for param_1, param_2 in zip(f_local.parameters(), f_initial.parameters()))

        if client_state.lambda_var is None:
            lambda_var = lambda_delta
        else:
            lambda_var = tuple((param_1 + param_2) for param_1, param_2 in zip(client_state.lambda_var, lambda_delta))

        # compute model_delta, stored in f_local.
        sd = f_local.state_dict()
        for key, param in zip(sd, lambda_var):
            sd[key] = eta * param
        f_local.load_state_dict(sd)

    # model is not used. Only model_delta is used.
    return FEDPD_client_state(global_round=client_state.global_round, model=None, lambda_var=lambda_var, model_delta=f_local)

@ray.remote(num_gpus=.25)
def ray_dispatch(config, loss_fn, device, client_state: FEDPD_client_state, client_dataloader, eta):
    return client_step(config, loss_fn, device, client_state, client_dataloader, eta)
