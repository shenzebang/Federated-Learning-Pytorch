import torch
import numpy as np


def make_monitor_fn():
    def evaluate_fn(model):
        param_norm = torch.norm(torch.stack([torch.norm(param) for param in model.parameters()]))
        return [param_norm]
    return evaluate_fn


def make_evaluate_fn(dataloader, device, eval_type="accuracy", n_classes=0, loss_fn=None):
    if eval_type == "accuracy":
        def evaluate_fn(model):
            with torch.autograd.no_grad():
                n_data = 0
                n_correct = 0
                loss = 0
                for data, label in dataloader:
                    data = data.to(device)
                    label = label.to(device)
                    f_data = model(data)
                    loss += loss_fn(f_data, label).item() * data.shape[0]
                    pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    n_correct += pred.eq(label.view_as(pred)).sum().item()
                    n_data += data.shape[0]
            return [np.true_divide(n_correct, n_data), np.true_divide(loss, n_data)]
    elif eval_type == "class_wise_accuracy":
        def evaluate_fn(model):
            correct_hist = torch.zeros(n_classes).to(device)
            label_hist = torch.zeros(n_classes).to(device)
            for data, label in dataloader:
                data = data.to(device)
                label = label.to(device)
                label_hist += torch.histc(label, n_classes, max=n_classes)
                f_data = model(data)
                pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_index = pred.eq(label.view_as(pred)).squeeze()
                label_correct = label[correct_index]
                correct_hist += torch.histc(label_correct, n_classes, max=n_classes)

            correct_rate_hist = correct_hist / label_hist
            return [correct_rate_hist.cpu().numpy()]
    else:
        raise NotImplementedError

    return evaluate_fn

