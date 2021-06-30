import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_class, hidden_size=(128, 128), device='cuda'):
        super(MLP, self).__init__()
        self.device = device
        self.n_class = n_class
        self.fc1 = nn.Linear(28 * 28, hidden_size[0])
        # nn.init.normal_(self.fc1.weight.data, 0.0, 1 / self.width / self.height / self.n_channel)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        # nn.init.normal_(self.fc2.weight.data, 0.0, 1 / hidden_size[0])
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_size[1], n_class)
        # nn.init.normal_(self.fc3.weight.data, 0.0, 1 / hidden_size[1])
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)
        self.activation = F.leaky_relu

        self.to(self.device)
        self.requires_grad_(False)
        # self.apply(weights_init)

    def forward(self, x):
        # flatten image input
        x = torch.flatten(x, start_dim=1)
        # add hidden layer, with relu activation function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x