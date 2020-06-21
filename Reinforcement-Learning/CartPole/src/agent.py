from typing import TYPE_CHECKING

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
import torch.nn as nn


if TYPE_CHECKING:
    from typing import Tuple, Optional
    import torch


class AgentNetwork(DQNTorchModel, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        nn.Module.__init__(self)
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        self.shape = model_config["custom_options"]["shape"]
        self.num_stack = model_config["custom_options"]["num_stack"]
        in_channels = self.shape[2]
        # Module Parameters
        self.module = nn.Sequential()
        # First Convolutional layer
        self.module.add_module(
            "conv_1", nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=2)
        )
        self.module.add_module("relu_1", nn.ReLU())
        self.module.add_module("conv_2", nn.Conv3d(16, 16, kernel_size=3, stride=2))
        self.module.add_module("relu_2", nn.ReLU())
        self.module.add_module("flatten", Flatten())
        self.module.add_module(
            "output", nn.Linear(in_features=2400, out_features=num_outputs)
        )

    @override(TorchModelV2)
    def forward(
        self, input_dict: dict, state: list, seq_lens: "Optional[float]" = None
    ) -> "Tuple[torch.Tensor, list]":
        # Shape: (Batch, Depth, Height * Weight * Channels)
        inputs = input_dict["obs"]
        # Shape: (Batch, Depth, Channels, Height, Weight)
        height, width, channels = self.shape
        inputs = inputs.reshape(inputs.size(0), inputs.size(1), height, width, channels)
        # Shape: (Batch, Channel, Depth, Height, Weight)
        inputs = inputs.permute(0, 4, 1, 2, 3)
        # Shape: (Batch, Num_Outputs)
        output = self.module(inputs)
        return output, state


class Flatten(nn.Module):
    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        return input.reshape(input.size(0), -1)
