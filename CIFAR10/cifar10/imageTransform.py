from __future__ import print_function, division
import torch


class ImageTransform():
    def __init__(self, parameters):
        super(ImageTransform, self).__init__()
        self.params = parameters

    def __call__(self, sample):
        return sample
