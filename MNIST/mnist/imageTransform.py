import torch


class ImageTransform():
    def __init__(self, parameters):
        super(ImageTransform, self).__init__()
        self.params = parameters

    def __call__(self, sample):
        #image = sample.squeeze(0)
        image = torch.zeros(sample.size(0), sample.size(1)+1, sample.size(2)+1)
        height, width = sample.size(1), sample.size(2)
        for row in range(height):
            for col in range(width):
                image[:, row, col] = (sample[:, row, col])
        return image
