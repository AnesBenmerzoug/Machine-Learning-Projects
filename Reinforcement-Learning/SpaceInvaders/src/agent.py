import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Linear
import torch.nn.functional as F
from collections import namedtuple


class DQN(Module):
    def __init__(self, params, num_actions):
        super(DQN, self).__init__()
        # Module Parameters
        self.params = params
        self.conv1 = Conv2d(4, 16, kernel_size=8, stride=4)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = BatchNorm2d(32)
        self.fc1 = Linear(in_features=2816,
                          out_features=512)
        self.fc2 = Linear(in_features=512,
                          out_features=num_actions)

        # Initialize Parameters
        self.initialize_parameters()

    def forward(self, x, temperature=1.0):
        # First Conv Layer
        x = F.relu(self.bn1(self.conv1(x)))
        # Second Conv Layer
        x = F.relu(self.bn2(self.conv2(x)))
        # Change view
        x = x.view(x.size(0), -1)
        # First FC layer
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        # Softmax
        x = F.softmax(x / temperature, dim=1)
        return x

    def initialize_parameters(self):
        for parameter in self.parameters():
            if len(parameter.size()) == 2:
                torch.nn.init.xavier_uniform(parameter, gain=1.0)

    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num

    @classmethod
    def load_model(cls, package, num_actions, useGPU=False):
        parameters = package['params']
        params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())
        model = cls(params, num_actions)
        model.load_state_dict(package['state_dict'])
        if useGPU is True:
            model = model.cuda()
        return model
