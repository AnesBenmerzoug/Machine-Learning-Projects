import torch
import numpy as np
import torch.optim as optim
from .agent import DQN
from .environment import CartPoleEnvironment
from .replaymemory import ReplayMemory
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
from collections import namedtuple
import random


class AgentTrainer:
    def __init__(self, parameters):
        self.params = parameters

        # Initialize action selection probability
        self.epsilon = self.params.epsilon_start

        # Initialize environment
        self.env = CartPoleEnvironment(self.params)

        # Define the transitions
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

        # Initialize Replay Memory
        self.memory = ReplayMemory(self.params.replay_memory_capacity, self.transition)

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()

        # Initialize model
        self.model = DQN(self.params, self.env.action_space.n)

        # Initilize Target Network
        self.target_network = DQN(self.params, self.env.action_space.n)
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.use_gpu is True:
            print("Using GPU")
            try:
                self.model.cuda()
                self.target_network.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.use_gpu = False
                self.model.cpu()
                self.target_network.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.use_gpu = False
                self.model.cpu()
                self.target_network.cpu()
        else:
            print("Using CPU")

        # Setup optimizer
        self.optimizer = self.optimizer_select()

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.params.step_size,
            gamma=self.params.decay_coeff,
        )

        # Criterion, Huber Loss
        self.criterion = SmoothL1Loss()

    def update_epsilon(self, episode):
        self.epsilon = self.params.epsilon_end + (
            self.params.epsilon_start - self.params.epsilon_end
        ) * np.exp(-episode / self.params.epsilon_decay)

    def train_model(self):
        avg_losses = np.zeros(self.params.num_episodes)
        episode_durations = np.zeros(self.params.num_episodes)
        model_path = self.params.model_dir / "trained_model.pt"
        for episode in range(self.params.num_episodes):
            try:
                print("Episode {}".format(episode + 1))

                # Update learning rate
                self.scheduler.step(episode)

                # Update epsilon
                self.update_epsilon(episode)

                print("Learning Rate = {}".format(self.optimizer.param_groups[0]["lr"]))

                print("Epsilon = {}".format(self.epsilon))

                # Set mode to training
                self.model.train()

                # Go through the training set
                avg_losses[episode], episode_durations[episode] = self.train_episode()

                print("Average loss= {}".format(avg_losses[episode]))

                if (episode + 1) % 100 == 0:
                    self.save_model(model_path, self.model.cpu().state_dict())
                    if self.use_gpu:
                        self.model = self.model.cuda()
            except KeyboardInterrupt:
                print("Training was interrupted")
                break
        # Close the environment
        self.env.close()
        # Saving trained model
        self.save_model(model_path, self.model.state_dict())
        return avg_losses, episode_durations

    def train_episode(self):
        # Initialize losses
        losses = 0.0
        score = 0.0
        # Initialize the environment and state
        self.env.reset()
        last_screen, _ = self.env.get_frame()
        current_screen, _ = self.env.get_frame()
        state = current_screen - last_screen
        for step_index in range(1, self.params.num_steps_per_episode + 1):
            # Wrap the state in a Variable
            if self.use_gpu is True:
                state = state.cuda()
            state = Variable(state)
            # Select and perform an action
            action = self.select_action(state)
            _, reward, done, info = self.env.step(action.item())
            score += reward
            reward = score

            # Observe new state
            last_screen = current_screen
            current_screen, _ = self.env.get_frame()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
                reward = 0.0

            reward = torch.Tensor([reward])
            if self.use_gpu is True:
                reward = reward.cuda()
            reward = Variable(reward, requires_grad=False)
            reward = F.tanh(reward)

            # Store the transition in memory
            self.memory.push(state.data, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = self.optimize_model()
            losses += loss

            # Update the target network
            if (step_index + 1) % self.params.target_update == 0:
                self.target_network.load_state_dict(self.model.state_dict())
                if self.use_gpu:
                    self.target_network = self.target_network.cuda()

            if done is True:
                break

        # Compute the average loss for this epoch
        episode_duration = step_index + 1
        avg_loss = losses / episode_duration
        print(f"Episode duration = {episode_duration}")
        return avg_loss, episode_duration

    def optimize_model(self):
        if len(self.memory) < self.params.batch_size:
            return 0.0
        transitions = self.memory.sample(self.params.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        if self.use_gpu is True:
            non_final_mask = non_final_mask.cuda()
            non_final_next_states = non_final_next_states.cuda()
        non_final_next_states = Variable(non_final_next_states)

        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.use_gpu is True:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
        state_batch = Variable(state_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Determine the greedy batch policies using the policy network - argmax Q(s_t+1, a)
        # This is Double Q-Learning
        action_selection = (
            self.model(non_final_next_states).max(1, keepdim=True)[1].detach()
        )

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.params.batch_size))
        if self.use_gpu is True:
            next_state_values = next_state_values.cuda()
        # next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        next_state_values[non_final_mask] = (
            self.target_network(non_final_next_states)
            .gather(1, action_selection)
            .squeeze()
            .detach()
        )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.params.gamma
        ) + reward_batch

        # Compute Huber loss
        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Zero the optimizer gradient
        self.optimizer.zero_grad()

        # Backward step
        loss.backward()

        # Clip gradients
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        # Weight Update
        self.optimizer.step()

        if self.use_gpu is True:
            torch.cuda.synchronize()

        return loss.data.item()

    def select_action(self, state):
        # Choose the action
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            action = Variable(torch.LongTensor([[action]]))
            if self.use_gpu:
                action = action.cuda()
        else:
            self.model.eval()
            action = self.model(state).max(1)[1].view(1, 1)
            self.model.train()
        return action

    def save_model(self, path, model_parameters):
        self.model.load_state_dict(model_parameters)
        torch.save(self.serialize(), path)

    def serialize(self):
        model_is_cuda = next(self.model.parameters()).is_cuda
        model = self.model.cpu() if model_is_cuda else self.model
        package = {
            "state_dict": model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
        }
        return package

    def optimizer_select(self):
        if self.params.optimizer == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                nesterov=self.params.nesterov,
            )
        else:
            raise NotImplementedError
