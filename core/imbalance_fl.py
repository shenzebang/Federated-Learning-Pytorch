from api import PrimalDualFedAlgorithm, FedAlgorithm
from tqdm import trange
from collections import namedtuple
import torch

ImFL_server_state = namedtuple("ImFL_server_state", ['global_round', 'model', 'lambda_var', 'perturbation'])


class ImbalanceFL(PrimalDualFedAlgorithm):
    def __init__(self, fed_algorithm: FedAlgorithm, config, logger, auxiliary_data=None):
        super(ImbalanceFL, self).__init__(fed_algorithm, config, logger, auxiliary_data)

    def server_init(self) -> ImFL_server_state:
        model = self.primal_fed_algorithm.server_state.model
        lambda_var = torch.zeros(self.config.n_workers)
        perturbation = torch.zeros(self.config.n_workers)
        return ImFL_server_state(global_round=0, model=model, lambda_var=lambda_var, perturbation=perturbation)

    def step(self):
        sss = self.server_state
        weights = (1. + sss.lambda_var - torch.mean(sss.lambda_var))
        client_losses = torch.tensor(self.primal_fed_algorithm.clients_evaluate())
        self.primal_fed_algorithm.fit(weights, self.config.n_p_steps)
        model_new = self.primal_fed_algorithm.server_state.model
        lambda_new = sss.lambda_var + self.config.lambda_lr * (client_losses - torch.mean(client_losses) - self.config.tolerance_epsilon) / self.config.n_workers
        lambda_new = torch.clamp(lambda_new, min=0., max=100.)
        perturb_new =  sss.perturbation + self.config.perturbation_lr * (-self.config.perturbation_penalty * sss.perturbation + sss.lambda_var)
        perturb_new = torch.clamp(lambda_new, min=0.)
        self.server_state = ImFL_server_state(global_round=sss.global_round+1, model=model_new, lambda_var=lambda_new, perturbation=perturb_new)

        # print('client_losses', client_losses)
        # print('weights', weights)
