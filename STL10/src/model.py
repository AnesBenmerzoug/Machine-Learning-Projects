import torch
from torch.nn import Module, Linear, Conv2d, ConvTranspose2d,Sequential, ReLU, BatchNorm2d
import torch.nn.functional as F
from collections import namedtuple


class STL10_Network(Module):
    r"""
    A Deep Neural Network implementation inspired by the work described in the article:
    "STACKED WHAT-WHERE AUTO-ENCODERS"
    """
    def __init__(self, params):
        super(STL10_Network, self).__init__()
        # Module Parameters
        self.params = params
        image_height = self.params.image_size[0]
        image_width = self.params.image_size[1]
        conv1_in_size = self.params.conv1_in_size
        conv1_out_size = self.params.conv1_out_size
        conv2_out_size = self.params.conv2_out_size
        fc1_out_size = self.params.fc1_out_size
        fc2_out_size = self.params.fc2_out_size
        output_size = self.params.output_size

        # Conv + BN + Relu Layers
        self.ConvBNReLU1 = Sequential(
            Conv2d(in_channels=conv1_in_size, out_channels=conv1_out_size,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv1_out_size, out_channels=conv1_out_size,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True)
        )

        self.ConvBNReLU2 = Sequential(
            Conv2d(in_channels=conv1_out_size, out_channels=conv2_out_size,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv2_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv2_out_size, out_channels=conv2_out_size,
                   kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv2_out_size),
            ReLU(inplace=True)
        )

        # DeConv + BN + Relu Layers
        self.DeConvBNReLU1 = Sequential(
            ConvTranspose2d(in_channels=conv2_out_size, out_channels=conv1_out_size,
                            kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=conv1_out_size, out_channels=conv1_out_size,
                            kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True)
        )

        self.DeConvBNReLU2 = Sequential(
            ConvTranspose2d(in_channels=conv1_out_size, out_channels=conv1_in_size,
                            kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_in_size),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=conv1_in_size, out_channels=conv1_in_size,
                            kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_in_size),
            ReLU(inplace=True)
        )

        # Fully Connected Layers
        in_features = image_height * image_width * conv2_out_size // 16
        self.fc1 = Linear(in_features=in_features,
                          out_features=fc1_out_size)
        self.fc2 = Linear(in_features=fc1_out_size,
                          out_features=fc2_out_size)
        self.output_layer = Linear(in_features=fc2_out_size,
                                   out_features=output_size)

        # Initialize Parameters
        self.initialize_parameters()

    def forward(self, input_image, unsupervised=False):
        # Convolutional Layers
        output1, indices1 = F.max_pool2d(self.ConvBNReLU1(input_image), kernel_size=2, stride=2, return_indices=True)
        output2, indices2 = F.max_pool2d(self.ConvBNReLU2(output1), kernel_size=2, stride=2, return_indices=True)
        # DeConvolutional Layers
        output_rec1 = self.DeConvBNReLU1(F.max_unpool2d(output2, indices2, kernel_size=2, stride=2))
        output_rec2 = self.DeConvBNReLU2(F.max_unpool2d(output_rec1, indices1, kernel_size=2, stride=2))
        # Skip some computations in the case of the unsupervised training
        if unsupervised is True:
            return None, ((output_rec2, None), (output_rec1, output1))
        # Fully Connected Layers
        output = output2.view(output2.size(0), -1)
        output = F.relu(self.fc1(output), inplace=True)
        output = F.relu(self.fc2(output), inplace=True)
        output = self.output_layer(output)
        if self.training is True:
            # LogSoftmax
            output = F.log_softmax(output, dim=1)
        else:
            # Softmax
            output = F.softmax(output, dim=1)
        return output, ((output_rec2, None), (output_rec1, output1))

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
    def load_model(cls, package, useGPU=False):
        parameters = package['params']
        params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())
        model = cls(params)
        model.load_state_dict(package['state_dict'])
        if useGPU is True:
            model = model.cuda()
        return model
