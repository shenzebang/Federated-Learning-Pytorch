from api import PrimalDualFedAlgorithm, FedAlgorithm
from tqdm import trange
from collections import namedtuple
import torch

RLFL_server_state = namedtuple("ImFL_server_state", ['global_round', 'model'])


class RatioLossFL(PrimalDualFedAlgorithm):
    def __init__(self, fed_algorithm: FedAlgorithm, config, logger, auxiliary_data):
        super(RatioLossFL, self).__init__(fed_algorithm, config, logger, auxiliary_data)
        self.alpha = 1.
        self.beta = .1

    def server_init(self) -> RLFL_server_state:
        model = self.primal_fed_algorithm.server_state.model
        return RLFL_server_state(global_round=0, model=model)

    def step(self):
        Ra_p = compute_Ra_p(self.server_state.model, self.auxiliary_data)
        loss_weights = self.alpha + self.beta * Ra_p
        self.primal_fed_algorithm.loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
        self.primal_fed_algorithm.fit([1.] * self.config.n_workers, self.config.n_p_steps)
        self.server_state = RLFL_server_state(self.server_state.global_round+1, self.primal_fed_algorithm.server_state.model)

def compute_Ra_p(model, data_label):
    # data_label should be a list, i th element of the list should be tuple (data, label) for class i
    model.requires_grad_(True)
    n_classes = len(data_label)
    loss_fn = torch.nn.CrossEntropyLoss()
    Delta_W = []

    for c in range(n_classes):
        data, label = data_label[c]
        data, label = data.to(model.device), label.to(model.device)
        f_data = model(data)
        loss = loss_fn(f_data, label)
        grad_c = torch.autograd.grad(loss, model.parameters())
        Delta_W.append(grad_c[-2])
    # compute Ra_p from Delta_W
    Delta_W_sum = torch.sum(torch.stack(Delta_W, dim=0), dim=0)
    Ra_p = [torch.abs(torch.mean(
                (n_classes-1)*Delta_W[p] / (Delta_W_sum - Delta_W[p]),
                dim=1)[p]) for p in range(n_classes)]
    Ra_p = torch.stack(Ra_p)

    model.requires_grad_(False)
    return Ra_p

