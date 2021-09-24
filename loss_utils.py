import torch
import torch.nn.functional as F

_softmax = torch.nn.Softmax(dim=1)

def focal_loss(f_data, label, gamma=2, num_classes=10):
    p = _softmax(f_data)
    label = F.one_hot(label, num_classes=num_classes)
    p_c = torch.sum(p*label, dim=1)
    return torch.mean( - (1-p_c) ** gamma * p_c.log())
