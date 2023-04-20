from api import PrimalDualFedAlgorithm, FedAlgorithm
from tqdm import trange
from collections import namedtuple
import torch
import wandb


ImFL_server_state = namedtuple("ImFL_server_state", ['global_round', 'model', 'lambda_var'])


class ImbalanceFL(PrimalDualFedAlgorithm):
    def __init__(self, fed_algorithm: FedAlgorithm, config, logger, auxiliary_data=None):
        super(ImbalanceFL, self).__init__(fed_algorithm, config, logger, auxiliary_data)

    def server_init(self) -> ImFL_server_state:
        model = self.primal_fed_algorithm.server_state.model
        lambda_var = torch.zeros(self.config.n_workers)
        return ImFL_server_state(global_round=0, model=model, lambda_var=lambda_var)

    def step(self):
        sss = self.server_state
        weights = (1. + sss.lambda_var - torch.mean(sss.lambda_var))
        client_losses, client_accs = torch.tensor(self.primal_fed_algorithm.clients_evaluate_train())
        self.primal_fed_algorithm.fit(weights, self.config.n_p_steps)
        model_new = self.primal_fed_algorithm.server_state.model
        lambda_new = sss.lambda_var + self.config.lambda_lr * (client_losses - torch.mean(client_losses) - self.config.tolerance_epsilon) / self.config.n_workers
        lambda_new = torch.clamp(lambda_new, min=0.)
        self.server_state = ImFL_server_state(global_round=sss.global_round+1, model=model_new, lambda_var=lambda_new)
        client_losses_test, client_accs_test = torch.tensor(self.primal_fed_algorithm.clients_evaluate_test())
        if sss.global_round==990:
            print(lambda_new)
        # print('client_losses', client_losses)
        # print('weights', weights)
        worst_acc = 1
        worst_loss = -1
        for i in range(len(sss.lambda_var)):
            if client_losses[i] > worst_loss:
                worst_loss_idx = i
            if client_accs[i] < worst_acc:
                worst_acc_idx = i
            wandb.log({f"lambda/client_{i}": sss.lambda_var[i].item()}) 
            wandb.log({f"loss/train/client_{i}": client_losses[i].item()}) 
            wandb.log({f"accuracy/train/client_{i}": client_accs[i].item()})
            wandb.log({f"loss/test/client_{i}": client_losses_test[i].item()}) 
            wandb.log({f"accuracy/test/client_{i}": client_accs_test[i].item()})
        wandb.log({f"worst_loss/train": client_losses[worst_loss_idx].item(),"worst_loss_idx" : worst_loss_idx,
                     "worst_acc/train":client_accs[worst_acc_idx], "worst_acc_idx":worst_acc_idx})
        wandb.log({f"worst_loss/test": client_losses_test[worst_loss_idx].item(),
                    "worst_acc/test": client_accs_test[worst_acc_idx]})
        wandb.log({f"worst_lambda": sss.lambda_var[worst_loss_idx].item()})       
        wandb.log({f"lambda/mean": sss.lambda_var.mean().item()}) 
        wandb.log({f"perturbation/mean": sss.perturbation.mean().item()})
        wandb.log({f"loss/train/mean": client_losses.mean().item()}) 
        wandb.log({f"accuracy/train/mean": client_accs.mean().item()})
        wandb.log({f"loss/test/mean": client_losses_test.mean().item()}) 
        wandb.log({f"accuracy/test/mean": client_accs_test.mean().item()})

ImFL_server_state_res = namedtuple("ImFL_server_state_res", ['global_round', 'model', 'lambda_var', 'perturbation'])

class ImbalanceFLRes(PrimalDualFedAlgorithm):
    def __init__(self, fed_algorithm: FedAlgorithm, config, logger, auxiliary_data=None):
        super(ImbalanceFLRes, self).__init__(fed_algorithm, config, logger, auxiliary_data)

    def server_init(self) -> ImFL_server_state_res:
        model = self.primal_fed_algorithm.server_state.model
        lambda_var = torch.zeros(self.config.n_workers)
        perturbation = torch.zeros(self.config.n_workers)
        return ImFL_server_state_res(global_round=0, model=model, lambda_var=lambda_var, perturbation=perturbation)

    def step(self):
        sss = self.server_state
        weights = (1. + sss.lambda_var - torch.mean(sss.lambda_var))
        client_losses, client_accs = torch.tensor(self.primal_fed_algorithm.clients_evaluate_train())
        self.primal_fed_algorithm.fit(weights, self.config.n_p_steps)
        model_new = self.primal_fed_algorithm.server_state.model
        lambda_new = sss.lambda_var + self.config.lambda_lr * (client_losses - torch.mean(client_losses) - self.config.tolerance_epsilon) / self.config.n_workers
        lambda_new = torch.clamp(lambda_new, min=0.)
        perturb_new =  sss.perturbation + self.config.perturbation_lr * (-self.config.perturbation_penalty * sss.perturbation + sss.lambda_var)
        perturb_new = torch.clamp(lambda_new, min=0.)
        self.server_state = ImFL_server_state_res(global_round=sss.global_round+1, model=model_new, lambda_var=lambda_new, perturbation=perturb_new)
        client_losses_test, client_accs_test = torch.tensor(self.primal_fed_algorithm.clients_evaluate_test())
        worst_acc = 1
        worst_loss = -1
        for i in range(len(sss.lambda_var)):
            if client_losses[i] > worst_loss:
                worst_loss_idx = i
            if client_accs[i] < worst_acc:
                worst_acc_idx = i
            wandb.log({f"lambda/client_{i}": sss.lambda_var[i].item()})
            wandb.log({f"perturbation/client_{i}": sss.perturbation[i].item()}) 
            wandb.log({f"loss/train/client_{i}": client_losses[i].item()}) 
            wandb.log({f"accuracy/train/client_{i}": client_accs[i].item()})
            wandb.log({f"loss/test/client_{i}": client_losses_test[i].item()}) 
            wandb.log({f"accuracy/test/client_{i}": client_accs_test[i].item()})
        wandb.log({f"worst_loss/train": client_losses[worst_loss_idx].item(),"worst_loss_idx" : worst_loss_idx,
                     "worst_acc/train":client_accs[worst_acc_idx], "worst_acc_idx":worst_acc_idx})
        wandb.log({f"worst_loss/test": client_losses_test[worst_loss_idx].item(),
                    "worst_acc/test": client_accs_test[worst_acc_idx]})
        wandb.log({f"worst_lambda": sss.lambda_var[worst_loss_idx].item(),
                    "worst_perturbation": sss.perturbation[worst_loss_idx].item()})   
        wandb.log({f"lambda/mean": sss.lambda_var.mean().item()}) 
        wandb.log({f"perturbation/mean": sss.perturbation.mean().item()})
        wandb.log({f"loss/train/mean": client_losses.mean().item()}) 
        wandb.log({f"accuracy/train/mean": client_accs.mean().item()})
        wandb.log({f"loss/test/mean": client_losses_test.mean().item()}) 
        wandb.log({f"accuracy/test/mean": client_accs_test.mean().item()})              
