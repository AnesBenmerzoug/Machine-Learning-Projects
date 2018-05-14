from __future__ import print_function, division
import torch


class ImageTransform():
    def __init__(self, parameters):
        super(ImageTransform, self).__init__()
        self.params = parameters

    def __call__(self, sample):
        print(sample.size())
        height = sample.size(1)
        width = sample.size(2)
        image = torch.zeros((height*self.params.num_channels, width))
        print(image.size())
        for row in range(height):
            for col in range(width):
                print(sample[:, row, col])
                print(image[row:row+3, col])
                image[row:row+3, col] = sample[:, row, col]
        return image
