from __future__ import print_function, division
import torch


class ImageTransform():
    def __init__(self, parameters):
        super(ImageTransform, self).__init__()
        self.params = parameters

    def __call__(self, sample):
        height = sample.size(1)
        width = sample.size(2)
        image = torch.zeros((height, width, self.params.num_channels))
        for row in range(height):
            for col in range(width):
                image[row, col, :] = sample[:, row, col]
        return image
