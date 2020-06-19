import gym
from gym import spaces
from gym.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
import collections
from typing import Optional


class ResizePixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape, pixel_key: str = "pixels"):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)
        self._pixel_key = pixel_key
        self.observation_space = spaces.Dict()

        pixels = self.env.render(mode="rgb_array")
        self.env.close()

        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float("inf"), float("inf"))
        else:
            raise TypeError(pixels.dtype)

        pixels_space = spaces.Box(
            shape=self.shape, low=low, high=high, dtype=pixels.dtype
        )

        self.observation_space.spaces[pixel_key] = pixels_space

    def observation(self, observation):
        import cv2

        new_observation = collections.OrderedDict()
        new_observation[self._pixel_key] = self.env.render(mode="rgb_array")
        reshape_height, reshape_width = self.shape[:-1]
        new_observation[self._pixel_key] = cv2.resize(
            new_observation[self._pixel_key],
            (reshape_width, reshape_height),
            interpolation=cv2.INTER_AREA,
        )
        if new_observation[self._pixel_key].ndim == 2:
            new_observation[self._pixel_key] = np.expand_dims(
                new_observation[self._pixel_key], -1
            )
        return new_observation


def cartpole_pixel_env_creator(env_config: Optional[dict] = None):
    if env_config is None:
        env_config = dict()
    env = gym.make("CartPole-v1").unwrapped
    env.reset()
    env = PixelObservationWrapper(env, render_kwargs={"mode": "rgb_array"})
    env.reset()
    env = ResizePixelObservationWrapper(env, shape=env_config["shape"])
    env.reset()
    env.close()
    return env
