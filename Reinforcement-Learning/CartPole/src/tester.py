import torch
from torch.autograd import Variable
from .agent import DQN
from collections import namedtuple
from .environment import CartPoleEnvironment
from .utils import save_animation
import random


class AgentTester:
    def __init__(self, parameters):
        self.params = parameters

        # Initialize environment
        self.env = CartPoleEnvironment(self.params)

        # Define the transitions
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )
        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()

        # Load Trained Model
        path = self.params.model_dir / "trained_model.pt"
        self.model = self.load_model(path, self.env.action_space.n)

        self.model.eval()

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.use_gpu is True:
            print("Using GPU")
            try:
                self.model.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.use_gpu = False
                self.model.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.use_gpu = False
                self.model.cpu()
        else:
            print("Using CPU")

    def test_model(self):
        self.model.eval()
        # Initialize list to hold screens
        screens = []
        # Initialize the environment and state
        self.env.reset()
        last_screen, _ = self.env.get_frame()
        current_screen, original_screen = self.env.get_frame()
        screens.append(original_screen)
        state = current_screen - last_screen
        done = False
        duration = 0
        while not done:
            duration += 1
            self.env.render()
            # Wrap the state in a Variable
            if self.use_gpu is True:
                state = state.cuda()
            state = Variable(state, volatile=True)
            # Select and perform an action
            action = self.select_action(state)
            _, _, done, _ = self.env.step(action.item())

            # Observe new state
            last_screen = current_screen
            current_screen, original_screen = self.env.get_frame()
            screens.append(original_screen)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Move to the next state
            state = next_state
        print("Duration = {}".format(duration))
        save_animation("static", screens, 24)
        # Close the environment
        self.env.close()

    def select_action(self, state):
        # Choose the action
        if random.random() < 0.05:
            action = self.env.action_space.sample()
            action = Variable(torch.LongTensor([[action]]))
        else:
            action = self.model(state).max(1)[1].view(1, 1)
        return action

    def load_model(self, path, num_actions):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        return DQN.load_model(package, num_actions)
