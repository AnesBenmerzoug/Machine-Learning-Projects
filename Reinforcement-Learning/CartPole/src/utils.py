import matplotlib.pyplot as plt
import moviepy.editor as mpy
import os


def plot_box(rewards, title="Rewards"):
    plt.boxplot(rewards)
    plt.title(title)
    plt.show()


def save_animation(path, image_list, fps=12):
    clip = mpy.ImageSequenceClip(image_list, fps=fps)
    clip.write_gif(os.path.join(path, "animation.gif"), fps=fps)
