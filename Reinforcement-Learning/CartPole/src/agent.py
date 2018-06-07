from torch.nn import Module, Conv2d, BatchNorm2d, Linear
import torch.nn.functional as F
from collections import namedtuple


class DQN(Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        # Module Parameters
        self.conv1 = Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = Conv2d(16, 32, kernel_size=5, stride=2)
        self.head = Linear(in_features=3808,
                           out_features=num_actions)

    def forward(self, x):
        # First Conv Layer
        x = F.relu(self.conv1(x))
        # Second Conv Layer
        x = F.relu(self.conv2(x))
        # Change view
        x = x.view(x.size(0), -1)
        # Output layer
        x = self.head(x)
        return x

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
