import torch
from torch.nn import Module, Linear, LSTM
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
        image_height = self.params.image_size[0]
        image_width = self.params.image_size[1]
        hidden_size = self.params.hidden_size
        output_size = self.params.output_size
        # LSTM Layers
        self.horizontal_layer = LSTM(input_size=image_height, hidden_size=hidden_size,
                                     num_layers=2, bidirectional=True, batch_first=True, bias=True)
        self.vertical_layer = LSTM(input_size=image_width, hidden_size=hidden_size,
                                   num_layers=2, bidirectional=True, batch_first=True, bias=True)
        # Output Layer
        self.output_layer = Linear(in_features=4 * hidden_size * image_height,
                                   out_features=output_size)

        # Initialize Parameters
        self.reset_parameters()

    def forward(self, input_image):
        # Horizontal LSTM Layer
        output_horizontal, _ = self.horizontal_layer(input_image)
        # Vertical LSTM Layer
        output_vertical, _ = self.vertical_layer(input_image.transpose(1, 2))
        # Concatenate the outputs of both layers
        output = torch.cat((output_horizontal, output_vertical), dim=2)
        # Change the view to process all outputs at once
        output = output.view(output.size(0), -1)
        if self.training is True:
            # Linear + LogSoftmax
            output = F.log_softmax(self.output_layer(output), dim=1)
        else:
            # Linear + Softmax
            output = F.softmax(self.output_layer(output), dim=1)
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
