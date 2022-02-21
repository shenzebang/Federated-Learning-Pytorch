import torch
import numpy as np

class Logger:
    def __init__(self, writer, test_fn, test_metric='accuracy'):
        self.writer = writer
        self.test_metric = test_metric
        self.test_fn = test_fn

    def log(self, step, model):
        metric = self.test_fn(model)
        if len(metric) == 1:
            t_accuracy = metric[0]
        elif len(metric) == 2:
            t_accuracy, t_loss = metric
        elif len(metric) == 4:
            t_accuracy, t_loss, tr_accuracy, tr_loss = metric
        else:
            raise NotImplementedError

        if self.test_metric == 'accuracy':
            self.writer.add_scalar("correct rate vs round/test", t_accuracy, step)
            if 't_loss' in locals(): self.writer.add_scalar("loss vs round/test", t_loss, step)
            if 'tr_accuracy' in locals(): self.writer.add_scalar("correct rate vs round/train", tr_accuracy, step)
            if 'tr_loss' in locals(): self.writer.add_scalar("loss vs round/train", tr_loss, step)

        elif self.test_metric == 'class_wise_accuracy':
            n_classes = len(t_accuracy)
            for i in range(n_classes):
                # the i th element is the accuracy for the test data with label i
                self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", t_accuracy[i], step)
                if 't_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", t_loss[i], step)
                if 'tr_accuracy' in locals(): self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", tr_accuracy[i], step)
                if 'tr_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", tr_loss[i], step)
        elif self.test_metric == 'model_monitor':
            self.writer.add_scalar("model param norm vs round", metric[0], step)
        else:
            raise NotImplementedError

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