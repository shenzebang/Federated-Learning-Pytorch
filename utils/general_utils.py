import torch
import numpy as np
import ray
import copy
from pathlib import Path
import wandb




def get_flat_grad_from(grad):
    flat_grad = torch.cat([torch.flatten(p) for p in grad])
    return flat_grad


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def average_grad(grads):
    # flatten the grads to tensors
    flat_grads = []
    for grad in grads:
        flat_grads.append(get_flat_grad_from(grad))

    average_flat_grad = torch.mean(torch.stack(flat_grads), dim=0)
    grad_0 = grads[0]
    average_grad_a = []
    for p in grad_0:
        average_grad_a.append(torch.zeros_like(p))

    set_flat_params_to(average_grad_a, average_flat_grad)
    return average_grad_a


def weighted_sum_functions(models, weights):
    # ensure "weights" has unit sum
    # sum_weights = sum(weights)
    # weights = [weight/sum_weights for weight in weights]

    if weights is None:
        weights = [1./len(models)]*len(models)

    average_model = copy.deepcopy(models[0])
    sds = [model.state_dict() for model in models]
    average_sd = sds[0]
    for key in sds[0]:
        # print(key, )
        if sds[0][key].dtype is torch.int64:
            average_sd[key] = torch.sum(torch.stack([sd[key].float() * weight for sd, weight in zip(sds, weights)]),
                                        dim=0).to(torch.int64)
        else:
            average_sd[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(sds, weights)]), dim=0)
    average_model.load_state_dict(average_sd)
    return average_model


def compute_model_delta(model_1, model_2):
    sd1 = model_1.state_dict()
    sd2 = model_2.state_dict()

    for key in sd1:
        sd1[key] = sd1[key] - sd2[key]
    model_1.load_state_dict(sd1)
    return model_1


def save_model(args, fed_learner):
    Path('./weights/').mkdir(exist_ok=True, parents=True)
    torch.save(fed_learner.server_state[1].state_dict(), f'./weights/{wandb.run.id}.pt')
    wandb.save(f'./weights/{wandb.run.id}.pt', policy = 'now')
    return


def _evaluate(loss_fn, device, model, dataloader):
    loss = torch.zeros(1).to(device)
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        loss += loss_fn(model(data), label)

    return loss.item()


@ray.remote(num_gpus=.3, num_cpus=4)
def _evaluate_ray(loss_fn, device, model, dataloader):
    loss = torch.zeros(1).to(device)
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        loss += loss_fn(model(data), label)
    return loss.item()

@ray.remote(num_gpus=.3, num_cpus=4)
def _acc_ray(device, model, dataloader):
    training = model.training
    model.eval()
    with torch.no_grad():
        n_data, n_correct = 0, 0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            f_data = model(data)
            pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            n_correct += pred.eq(label.view_as(pred)).sum().item()
            n_data += data.shape[0]
    if training:
        model.train()
    return n_correct/n_data