import gym
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


class SpaceInvadersEnvironment:
    def __init__(self, parameters):
        self.env = gym.make('SpaceInvadersDeterministic-v4')
        self.params = parameters
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((105, 80), interpolation=Image.CUBIC),
            T.ToTensor()
        ])
        self.last_frame = None

    def get_frames(self, count):
        frames = []
        original_frames = []
        for i in range(count):
            # transpose into torch order (CHW)
            frame = self.render(mode='rgb_array')
            original_frames.append(frame)
            frame = frame.transpose((2, 0, 1))
            # Convert it to grayscale
            frame = frame.mean(axis=0, keepdims=True)
            # Convert to float, rescale, convert to torch tensor
            frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
            frame = torch.from_numpy(frame)
            frame = self.transform(frame).unsqueeze(0)
            if self.last_frame is None:
                self.last_frame = frame
            else:
                frame = frame + frame.lt(self.last_frame).float() * self.last_frame
            frames.append(frame)
        return frames, original_frames

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
