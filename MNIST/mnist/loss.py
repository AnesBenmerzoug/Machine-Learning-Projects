import torch
from torch.nn import Module


class Entropy(Module):
    r"""
    Adapted from "Regularizing Neural Networks by Penalizing Confident Output Distributions"
    (https://arxiv.org/abs/1701.06548) Which is used to penalize low entropy output distributions,
    which acts as a regularizer for the network
    """
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, input_):
        output = - torch.sum(torch.exp(input_) * input_) / (input_.size(0))
        return output
