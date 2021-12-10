import torch
import torch.nn.functional as F

_softmax = torch.nn.Softmax(dim=1)

def focal_loss(f_data, label, gamma=2, num_classes=10):
    p = _softmax(f_data)
    label = F.one_hot(label, num_classes=num_classes)
    p_c = torch.sum(p*label, dim=1)
    return torch.mean( - (1-p_c) ** gamma * p_c.log())

def alpha_loss(f_data,label,alpha=0.8):
    if alpha == 1.0:
        loss = torch.mean(torch.sum(-target*torch.log(torch.softmax(f_data,dim=1) + 1e-8),dim=1))
    else:
        alpha_cuda = torch.FloatTensor([my_alpha]).cuda()
        one_cuda = torch.FloatTensor([1.0]).cuda()
        loss = (alpha_cuda/(alpha_cuda-one_cuda))*torch.mean(torch.sum(label*(one_cuda - torch.softmax(f_data,dim=1).pow(one_cuda - (one_cuda/alpha_cuda))),dim=1))
    return loss