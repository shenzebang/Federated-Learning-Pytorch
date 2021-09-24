import torch

def get_auxiliary_data(config, transforms, dataset, n_classes, n_aux):
    # output an auxiliary dataset with n_aux sample per class
    # the output is a list of tuples (data, label)
    data = dataset.data
    label = dataset.targets

    aux_data = []
    for c in range(n_classes):
        mask_c = label == c
        data_c = data[mask_c.numpy()]
        label_c = label[mask_c]
        assert(len(data_c) > n_aux)
        aux_data_c = data_c[:n_aux]
        aux_label_c = label_c[:n_aux]

        aux_data_c = torch.stack([transforms(aux_data_c[i]) for i in range(n_aux)])
        # if config.dataset == "cifar10" or config.dataset == "cifar100":
        #     aux_data_c = aux_data_c.permute(0, 3, 1, 2)
        aux_data.append((aux_data_c, aux_label_c))

    return aux_data