import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from model import convnet, mlp

DATASETS = {
    "cifar": datasets.CIFAR10,
    "mnist": datasets.MNIST,
    "emnist": datasets.EMNIST
}


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


def average_functions(models):
    average_model = models[0]
    sds = [model.state_dict() for model in models]
    average_sd = sds[0]
    for key in sds[0]:
        average_sd[key] = torch.mean(torch.stack([sd[key] for sd in sds]), dim=0)
    average_model.load_state_dict(average_sd)
    return average_model



def split_dataset(args, dataset: VisionDataset, transform=None):

    data = dataset.data
    label = dataset.targets

    # centralized case, no need to split
    if args.n_workers == 1:
        return [make_dataset(data, label, dataset.train, transform)]


    homo_ratio = args.homo_ratio
    n_workers = args.n_workers

    n_data = data.shape[0]

    n_homo_data = int(n_data * homo_ratio)

    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
        data_homo_list, label_homo_list = np.split(data_homo, n_workers), label_homo.chunk(n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        label_hetero_sorted, index = torch.sort(label_hetero)
        data_hetero_sorted = data_hetero[index]

        data_hetero_list, label_hetero_list = np.split(data_hetero_sorted, n_workers), label_hetero_sorted.chunk(n_workers)

    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                     zip(data_homo_list, data_hetero_list)]
        label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                      zip(label_homo_list, label_hetero_list)]
    elif n_homo_data < n_data:
        data_list = data_hetero_list
        label_list = label_hetero_list
    else:
        data_list = data_homo_list
        label_list = label_homo_list

    return [make_dataset(data, label, dataset.train, transform) for data, label in zip(data_list, label_list)]


class LocalDataset(VisionDataset):
    def __init__(self, data, label, train, transform=None, root: str = ""):
        super().__init__(root, transform)
        self.data = data
        self.label = label
        self.transform = transform
        self.train = train
        assert data.shape[0] == label.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.label[item]

def make_dataset(data, label, train, transform):
    return LocalDataset(data, label, train, transform)


def make_transforms(args, train=True):
    if args.dataset == "cifar10":
        if train:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        transform = None

    return transform


def make_dataloader(args, dataset: LocalDataset):
    if dataset.train is True:
        batch_size = dataset.data.shape[0] // args.client_step_per_epoch
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

    return dataloader


def make_evaluate_fn(dataloader, device, eval_type="accuracy"):
    if eval_type == "accuracy":
        def evaluate_fn(model):
            n_data = 0
            n_correct = 0
            for data, label in dataloader:
                data.to(device)
                label.to(device)
                f_data = model(data)
                pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                n_correct += pred.eq(label.view_as(pred)).sum().item()
                n_data += data.shape[0]

            return np.true_divide(n_correct/n_data)
    else:
        raise NotImplementedError

    return evaluate_fn


def save_model(args, fed_learner):
    return



def load_dataset(args):
    if args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR10(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 3
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, n_classes, n_channels


def make_model(args, n_classes, n_channels, device):
    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    if args.model == "convnet":
        model = convnet.LeNet5(n_classes, n_channels, conv_hidden_size, dense_hidden_size, device)
    elif args.model == "mlp":
        model = mlp.MLP(n_classes, dense_hidden_size, device)
    else:
        raise NotImplementedError

    return model


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:min(len(lst), i + n)]


class Logger:
    def __init__(self, writer):
        self.writer = writer

    def log(self, step, accuracy):
        self.writer.add_scalar("correct rate vs round/test", accuracy, step)
