import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet5(nn.Module):
    def __init__(self, n_class=10, n_channels=3, conv_size=(6, 16),
                 hidden_size=(120, 84), device='cuda'):
        super(LeNet5, self).__init__()
        self.device = device
        self.activation = F.leaky_relu
        self.conv1 = nn.Conv2d(n_channels, conv_size[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_size[0], conv_size[1], 5)
        self.fc1 = nn.Linear(conv_size[1] * 5 * 5, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_class)

        self.requires_grad_(False)
        self.to(device)

        # By default, layers are initialized using Kaiming Uniform method.
        # self.apply(weight_init)

    def forward(self, x, debug=False):
        if debug:
            x = self.pool(self.activation(self.conv1(x)))
            print("1", torch.isnan(x).any())
            x = self.pool(self.activation(self.conv2(x)))
            print("2", torch.isnan(x).any())
            x = x.view(x.shape[0], -1)
            x = self.activation(self.fc1(x))
            print("3", torch.isnan(x).any())
            x = self.activation(self.fc2(x))
            print("4", torch.isnan(x).any())
            x = self.fc3(x)
            print("5", torch.isnan(x).any())
        else:
            x = self.pool(self.activation(self.conv1(x)))
            x = self.pool(self.activation(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
        return x

# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         nn.init.zeros_(m.bias)