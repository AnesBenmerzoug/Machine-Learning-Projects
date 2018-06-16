import gym
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


class CartPoleEnvironment:
    def __init__(self, parameters):
        self.env = gym.make('CartPole-v0').unwrapped
        self.params = parameters
        self.resize = T.Compose([
            T.ToPILImage(),
            T.Resize(40, interpolation=Image.CUBIC),
            T.ToTensor()
        ])
        self.frame_width = 600

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.frame_width / world_width
        return int(self.env.state[0] * scale + self.frame_width / 2.0)  # MIDDLE OF CART

    def get_frame(self):
        original_frame = self.render(mode='rgb_array')
        frame = original_frame.transpose((2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the frame
        frame = frame[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.frame_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        frame = frame[:, :, slice_range]
        # Convert to float, rescare, convert to torch tensor
        # (this doesn't require a copy)
        frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
        frame = torch.from_numpy(frame)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(frame).unsqueeze(0), original_frame

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
