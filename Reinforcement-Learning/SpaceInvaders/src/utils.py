from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import os


def imgshow():
    fig, ax = plt.subplots()

    def draw(img):
        ax.imshow(np.transpose(make_grid(img).numpy(), (1, 2, 0)))
        plt.pause(0.001)
    return draw


def plot_losses(losses, title='', xlabel='', ylabel=''):
    epochs = np.arange(losses.size) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_accuracy(accuracy, classes, title='', xlabel='', ylabel=''):
    indices = np.arange(len(classes))
    width = 0.35
    bar = plt.bar(indices, accuracy, width)
    for idx, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{:.2f}'.format(accuracy[idx]), ha='center', va='bottom')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(indices, classes)
    plt.show()


def plot_scores(scores, title='', xlabel='', ylabel='', bins=10):
    plt.hist(scores, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def save_animation(path, image_list, fps=12):
    clip = mpy.ImageSequenceClip(image_list, fps=fps)
    clip.write_gif(os.path.join(path, 'animation.gif'), fps=fps)
