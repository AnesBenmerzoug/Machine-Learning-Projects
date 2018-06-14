import torch
from torch.autograd import Variable
from .agent import DQN
from collections import namedtuple
from .environment import SpaceInvadersEnvironment
from .utils import imgshow
import random
import time


class AgentTester(object):
    def __init__(self, parameters):
        self.params = parameters

        # Initialize environment
        self.env = SpaceInvadersEnvironment(self.params)

        # Define the transitions
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))
        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Load Trained Model
        self.load_model(self.env.action_space.n)
        self.model.eval()

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.useGPU is True:
            print("Using GPU")
            try:
                self.model.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
        else:
            print("Using CPU")

    def test_model(self):
        self.model.eval()
        #draw_fn = imgshow()
        # Initialize the environment and state
        self.env.reset()
        # Get the first 4 frames and initialize the state
        frames = self.env.get_frames(4)
        state = torch.cat(frames, dim=1)
        done = False
        score = 0.0
        while not done:
            #draw_fn(frames[0])
            self.env.render()
            # Wrap the state in a Variable
            if self.useGPU is True:
                state = state.cuda()
            state = Variable(state, volatile=True)
            # Select and perform an action
            action = self.select_action_epsilon_greedy(state)
            _, reward, done, _ = self.env.step(action.data[0])
            score += reward

            # Observe new state
            if not done:
                frames = frames[1:] + self.env.get_frames(1)
                next_state = torch.cat(frames, dim=1)
            else:
                next_state = None

            # Move to the next state
            state = next_state
        print("Score = {}".format(score))
        # Close the environment
        self.env.close()

    def select_action_epsilon_greedy(self, state):
        # Choose the action
        if random.random() < 0.05:
            action = self.env.action_space.sample()
            action = Variable(torch.LongTensor([[action]]))
        else:
            action = self.model(state).max(1)[1].view(1, 1)
        return action

    def load_model(self, num_actions, useGPU=False):
        package = torch.load(self.params.testModelPath, map_location=lambda storage, loc: storage)
        self.model = DQN.load_model(package, num_actions, useGPU)
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

