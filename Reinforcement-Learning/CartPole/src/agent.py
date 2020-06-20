import torch
import torch.nn as nn
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from loguru import logger

torch.autograd.set_detect_anomaly(True)


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
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("conv_2", nn.Conv2d(16, 16, kernel_size=5, stride=2))
        self.module.add_module("relu_2", nn.ReLU())
        self.module.add_module("flatten", Flatten())
        self.module.add_module(
            "output", nn.Linear(in_features=1344, out_features=num_outputs)
        )

        self.prev_output_module = nn.Sequential()
        self.prev_output_module.add_module(
            "fc", nn.Linear(in_features=num_outputs, out_features=num_outputs)
        )
        self.prev_output_module.add_module("relu", nn.ReLU())

        self.prev_output = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]["pixels"]
        inputs = inputs.permute(0, 3, 1, 2)
        output_1 = self.module(inputs)
        if self.prev_output is None:
            self.prev_output = output_1.clone().zero_()
        else:
            self.prev_output = self.prev_output.detach()
        output_2 = self.prev_output_module(self.prev_output)
        final_output = output_1 + output_2
        self.prev_output = output_1.clone()
        return final_output, state


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
