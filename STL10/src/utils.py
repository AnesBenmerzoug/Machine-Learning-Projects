from __future__ import print_function, division
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def imgshow(img):
    img = make_grid(img).numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
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
    width = 0.2
    bar = plt.bar(indices, accuracy, width)
    for idx, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 '{:.2f}'.format(accuracy[idx]), ha='center', va='bottom')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(indices, classes)
    plt.show()

def plotconfusion(confusion_matrix, classes, title='', xlabel='', ylabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    fig.colorbar(cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([''] + list(classes), rotation=90)
    ax.set_yticklabels([''] + list(classes))
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
