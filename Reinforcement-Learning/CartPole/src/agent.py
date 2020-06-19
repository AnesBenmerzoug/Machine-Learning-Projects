import torch.nn as nn
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


screen_width = 600
screen_height = 400


class AgentNetwork(DQNTorchModel, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        dueling = model_config["custom_options"].get("dueling", False)
        kwargs.update({"dueling": dueling})
        nn.Module.__init__(self)
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        shape = model_config["custom_options"]["shape"]
        in_channels = shape[2]
        # Module Parameters
        self.module = nn.Sequential()
        # First Convolutional layer
        self.module.add_module(
            "conv_1", nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        )
        # Followed by Relu activation
        self.module.add_module("relu_1", nn.ReLU())
        # Second Convolutional layer
        self.module.add_module("conv_2", nn.Conv2d(16, 16, kernel_size=5, stride=2))
        # Followed by Relu activation
        self.module.add_module("relu_2", nn.ReLU())
        # Followed by Relu activation
        self.module.add_module("relu_2", nn.ReLU())
        # Layer to flatten output from convolutional layer
        self.module.add_module("flatten", Flatten())
        # Fully Connected output layer
        self.module.add_module(
            "output", nn.Linear(in_features=1344, out_features=num_outputs)
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]["pixels"]
        inputs = inputs.permute(0, 3, 1, 2)
        output = self.module(inputs)
        return output, state

    @property
    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num

    @classmethod
    def load_model(cls, package, num_actions):
        model = cls(num_actions)
        model.load_state_dict(package["state_dict"])
        return model


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
