from __future__ import print_function, division
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def imgshow(img):
    plt.imshow(np.transpose(make_grid(img).numpy(), (1, 2, 0)))
    plt.show()


def plotlosses(losses, title='', xlabel='', ylabel=''):
    epochs = np.arange(losses.size) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plotaccuracy(accuracy, classes, title='', xlabel='', ylabel=''):
    indices = np.arange(len(classes))
    width = 0.35
    plt.bar(indices, accuracy, width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(indices, classes)
    plt.show()

