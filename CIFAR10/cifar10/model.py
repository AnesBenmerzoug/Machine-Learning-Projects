import torch
from torch.nn import Module, Linear, Conv2d, Sequential, ReLU, BatchNorm2d
import torch.nn.functional as F


class CIFAR10_Network(Module):
    r"""
    A Deep Neural Network implementation inspired by the one described in the article:
    "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION"
    """
    def __init__(self, params):
        super(CIFAR10_Network, self).__init__()
        # Module Parameters
        self.params = params
        input_height = self.params.image_size[0]
        input_width = self.params.image_size[1]
        conv1_in_size = self.params.conv1_in_size
        conv1_out_size = self.params.conv1_out_size
        conv2_out_size = self.params.conv2_out_size
        conv3_out_size = self.params.conv3_out_size
        conv4_out_size = self.params.conv4_out_size
        fc1_out_size = self.params.fc1_out_size
        fc2_out_size = self.params.fc2_out_size
        output_size = self.params.output_size

        # Conv + BN + Relu Layers
        self.ConvBNReLU1 = Sequential(
            Conv2d(in_channels=conv1_in_size, out_channels=conv1_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv1_out_size, out_channels=conv1_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv1_out_size),
            ReLU(inplace=True)
        )

        self.ConvBNReLU2 = Sequential(
            Conv2d(in_channels=conv1_out_size, out_channels=conv2_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv2_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv2_out_size, out_channels=conv2_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv2_out_size),
            ReLU(inplace=True)
        )

        self.ConvBNReLU3 = Sequential(
            Conv2d(in_channels=conv2_out_size, out_channels=conv3_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv3_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv3_out_size, out_channels=conv3_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv3_out_size),
            ReLU(inplace=True),
        )

        self.ConvBNReLU4 = Sequential(
            Conv2d(in_channels=conv3_out_size, out_channels=conv4_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv4_out_size),
            ReLU(inplace=True),
            Conv2d(in_channels=conv4_out_size, out_channels=conv4_out_size, kernel_size=3, padding=1, stride=1),
            BatchNorm2d(num_features=conv4_out_size),
            ReLU(inplace=True),
        )

        # Fully Connected Layers
        self.fc1 = Linear(in_features=2 * 2 * conv4_out_size,
                          out_features=fc1_out_size)
        self.fc2 = Linear(in_features=fc1_out_size,
                          out_features=fc2_out_size)
        self.output_layer = Linear(in_features=fc2_out_size,
                                   out_features=output_size)

        # Initialize Parameters
        self.initialize_parameters()

    def forward(self, input_image):
        # Convolutional Layers
        output = F.max_pool2d(self.ConvBNReLU1(input_image), kernel_size=2, stride=2)
        output = F.max_pool2d(self.ConvBNReLU2(output), kernel_size=2, stride=2)
        output = F.max_pool2d(self.ConvBNReLU3(output), kernel_size=2, stride=2)
        output = F.max_pool2d(self.ConvBNReLU4(output), kernel_size=2, stride=2)
        # Fully Connected Layers
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output), inplace=True)
        output = F.relu(self.fc2(output), inplace=True)
        output = self.output_layer(output)
        if self.training is True:
            # LogSoftmax
            output = F.log_softmax(output, dim=1)
        else:
            # Softmax
            output = F.softmax(output, dim=1)
        return output

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
        params = package['params']
        model = cls(params)
        model.load_state_dict(package['state_dict'])
        # Replace NaN parameters with random values
        for parameter in model.parameters():
            if len(parameter.size()) == 2:
                if float(parameter.data[0, 0]) != float(parameter.data[0, 0]):
                    torch.nn.init.xavier_uniform(parameter, gain=1.0)
            else:
                if float(parameter.data[0]) != float(parameter.data[0]):
                    torch.nn.init.uniform(parameter, -1.0, 1.0)
        if useGPU is True:
            model = model.cuda()
        return model
