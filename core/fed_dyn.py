import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from api import FedAlgorithm
from utils import weighted_sum_functions
from collections import namedtuple
from typing import List
import ray

FEDDYN_server_state = namedtuple("FEDDYN_server_state", ['global_round', 'model', 'h'])
FEDDYN_client_state = namedtuple("FEDDYN_client_state", ['global_round', 'model', 'grad'])


class FEDDYN(FedAlgorithm):
    def __init__(self, init_model,
                 client_dataloaders,
                 loss,
                 loggers,
                 config,
                 device
                 ):
        super(FEDDYN, self).__init__(init_model, client_dataloaders, loss, loggers, config, device)
        self.alpha = config.alpha
        self.n_workers = config.n_workers
        self.n_workers_per_round = config.n_workers_per_round
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDDYN_server_state(global_round=0, model=init_model, h=init_model)

    def client_init(self, server_state: FEDDYN_server_state, client_dataloader):
        return FEDDYN_client_state(global_round=server_state.global_round, model=server_state.model, grad=None)

    def clients_step(self, clients_state, active_ids):

        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [
                _client_step(self.config, self.loss, self.device, client_state, client_dataloader, self.alpha)
                for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [client_step.remote(self.config, self.loss, self.device, client_state, client_dataloader, self.alpha)
                 for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: FEDDYN_server_state, client_states: FEDDYN_client_state, weights, active_ids):
        active_clients = [client_states[i] for i in active_ids]
        h_new = weighted_sum_functions(
            [server_state.h] +
            [client_state.model for client_state in active_clients] +
            [server_state.model],
            [1] +
            [-self.alpha * (1 / self.n_workers_per_round) for client_state in active_clients] +
            [self.alpha * self.n_workers_per_round / self.n_workers])
        new_server_state = FEDDYN_server_state(
            global_round=server_state.global_round + 1,
            model=weighted_sum_functions(
                [client_state.model for client_state in active_clients] +
                [server_state.model],
                [(1. / self.n_workers_per_round) for client_state in active_clients] +
                [-1./self.alpha]),
            h=h_new
        )
        return new_server_state

    def clients_update(self, server_state: FEDDYN_server_state, clients_state: List[FEDDYN_client_state], active_ids):
        return [FEDDYN_client_state(global_round=server_state.global_round, model=server_state.model, grad=client.grad)
                for client in clients_state]


@ray.remote(num_gpus=.2)
def client_step(config, loss_fn, device, client_state: FEDDYN_client_state, client_dataloader, alpha):
    f_local = copy.deepcopy(client_state.model)
    f_initial = client_state.model
    f_local.requires_grad_(True)
    grad_local = client_state.grad

    lr_decay = 1.
    # if client_state.global_round >= 1000:
    #     lr_decay = .1
    # elif client_state.global_round >= 1500:
    #     lr_decay = .01
    optimizer = optim.SGD(f_local.parameters(), lr=lr_decay * config.local_lr)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            # Start with MSE part
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            # Now compute the inner product
            if grad_local is not None:
                curr_params = None
                for theta in f_local.parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = theta.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, theta.view(-1)), dim=0)
                lin_penalty = torch.sum(torch.mul(curr_params, grad_local))
                loss -= lin_penalty

            # Now compute the quadratic part
            quad_penalty = 0.0
            for theta, theta_init in zip(f_local.parameters(), f_initial.parameters()):
                quad_penalty += F.mse_loss(theta, theta_init, reduction='sum')

            loss += alpha / 2.0 * quad_penalty

            # Now take loss
            loss.backward()

            # Update the previous gradients
            # grad_local = None
            # for param in f_local.parameters():
            #     if not isinstance(grad_local, torch.Tensor):
            #         grad_local = param.grad.view(-1).clone()
            #     else:
            #         grad_local = torch.cat((grad_local, param.grad.view(-1).clone()), dim=0)

            # Now take step
            optimizer.step()

    # Update the previous gradients
    if grad_local is None:
        for param_1, param_2 in zip(f_local.parameters(), f_initial.parameters()):
            if not isinstance(grad_local, torch.Tensor):
                grad_local = (param_1 - param_2).view(-1) * - alpha
            else:
                grad_local = torch.cat((grad_local, (param_1 - param_2).view(-1) * - alpha), dim=0)
    else:
        for param_1, param_2, param_3 in zip(f_local.parameters(), f_initial.parameters(), grad_local):
                param_3.add_(- alpha * (param_1 - param_2).view(-1))

    return FEDDYN_client_state(global_round=client_state.global_round, model=f_local, grad=grad_local)


def _client_step(config, loss_fn, device, client_state: FEDDYN_client_state, client_dataloader, alpha):
    f_local = copy.deepcopy(client_state.model)
    f_initial = client_state.model
    f_local.requires_grad_(True)
    grad_local = client_state.grad

    lr_decay = 1.
    # if client_state.global_round >= 1000:
    #     lr_decay = .1
    # elif client_state.global_round >= 1500:
    #     lr_decay = .01
    optimizer = optim.SGD(f_local.parameters(), lr=lr_decay * config.local_lr)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            # Start with MSE part
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            # Now compute the inner product
            if grad_local is not None:
                curr_params = None
                for theta in f_local.parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = theta.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, theta.view(-1)), dim=0)
                lin_penalty = torch.sum(torch.mul(curr_params, grad_local))
                loss -= lin_penalty

            # Now compute the quadratic part
            quad_penalty = 0.0
            for theta, theta_init in zip(f_local.parameters(), f_initial.parameters()):
                quad_penalty += F.mse_loss(theta, theta_init, reduction='sum')

            loss += alpha / 2.0 * quad_penalty

            # Now take loss
            loss.backward()

            # Update the previous gradients
            # grad_local = None
            # for param in f_local.parameters():
            #     if not isinstance(grad_local, torch.Tensor):
            #         grad_local = param.grad.view(-1).clone()
            #     else:
            #         grad_local = torch.cat((grad_local, param.grad.view(-1).clone()), dim=0)

            # Now take step
            optimizer.step()

    # Update the previous gradients
    if grad_local is None:
        for param_1, param_2 in zip(f_local.parameters(), f_initial.parameters()):
            if not isinstance(grad_local, torch.Tensor):
                grad_local = (param_1 - param_2).view(-1) * - alpha
            else:
                grad_local = torch.cat((grad_local, (param_1 - param_2).view(-1) * - alpha), dim=0)
    else:
        for param_1, param_2, param_3 in zip(f_local.parameters(), f_initial.parameters(), grad_local):
            param_3.add_(- alpha * (param_1 - param_2).view(-1))

    return FEDDYN_client_state(global_round=client_state.global_round, model=f_local, grad=grad_local)
