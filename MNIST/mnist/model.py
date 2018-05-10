import torch
from torch.nn import Module, Linear, Conv2d
import torch.nn.functional as F


class MNIST_2DLSTM(Module):
    r"""
    A Deep Neural Network implementation based on the network described in:
    'Multi-column Deep Neural Networks for Image Classification'
    """
    def __init__(self, params):
        super(MNIST_2DLSTM, self).__init__()
        # Module Parameters
        self.params = params
        # Convolutional Layers
        self.first_conv_layer = Conv2d(in_channels=self.params.first_conv_in,
                                       out_channels=self.params.first_conv_out,
                                       kernel_size=self.params.first_conv_kernel)
        self.second_conv_layer = Conv2d(in_channels=self.params.first_conv_out,
                                        out_channels=self.params.second_conv_out,
                                        kernel_size=self.params.second_conv_kernel,)
        # Fully Connected Layers
        self.first_fc_layer = Linear(in_features=self.params.second_conv_out * 3 * 3,
                                     out_features=self.params.first_fc_output)
        self.output_fc_layer = Linear(in_features=self.params.first_fc_output,
                                      out_features=self.params.second_fc_output)

        # Initialize Parameters
        self.reset_parameters()

    def forward(self, input_image):
        # First Convolutional Layer
        output = F.max_pool2d(F.tanh(self.first_conv_layer(input_image)), (2, 2))
        # Second Convolutional Layer
        output = F.max_pool2d(F.tanh(self.second_conv_layer(output)), (3, 3))
        # Change the view to process all outputs at once
        output = output.view(output.size(0), -1)
        # Fist Fully Connected Layer
        output = F.tanh(self.first_fc_layer(output))
        if self.training is True:
            # Linear + LogSoftmax
            output = F.log_softmax(self.output_fc_layer(output), dim=1)
        else:
            # Linear + Softmax
            output = F.softmax(self.output_fc_layer(output), dim=1)
        return output

    def reset_parameters(self):
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
