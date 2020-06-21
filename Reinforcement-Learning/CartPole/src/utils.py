import matplotlib.pyplot as plt
import moviepy.editor as mpy
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    from pathlib import Path


def plot_box(rewards: list, title: str = ""):
    plt.boxplot(rewards)
    plt.title(title)
    plt.show()


def save_animation(filepath: "Union[str, Path]", image_list: list, fps: int = 12):
    clip = mpy.ImageSequenceClip(image_list, fps=fps)
    clip.write_gif(os.fspath(filepath), fps=fps)
