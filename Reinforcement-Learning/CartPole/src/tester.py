import torch
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
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Load Trained Model
        path = self.params.model_dir / "trained_model.pt"
        self.model = self.load_model(path, self.env.action_space.n)
        self.model.eval()
        self.model.to(self.device)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

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
        state = state.to(self.device)
        done = False
        duration = 0
        while True:
            duration += 1
            self.env.render()
            # Select and perform an action
            action = self.select_action(state)
            _, _, done, _ = self.env.step(action.item())

            # Observe new state
            last_screen = current_screen
            current_screen, original_screen = self.env.get_frame()
            screens.append(original_screen)

            if done:
                break

            next_state = current_screen - last_screen
            next_state = next_state.to(self.device)

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
            action = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        else:
            action = self.model(state).max(1)[1].view(1, 1)
        return action

    def load_model(self, path, num_actions):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        return DQN.load_model(package, num_actions)
