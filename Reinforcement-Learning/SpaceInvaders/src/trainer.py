import torch
import numpy as np
import torch.optim as optim
from .agent import DQN
from .environment import SpaceInvadersEnvironment
from .replaymemory import ReplayMemory
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
from collections import namedtuple
import random
import time
import os


class AgentTrainer(object):
    def __init__(self, parameters):
        self.params = parameters

        # Initialize action selection probability
        self.epsilon = self.params.epsilon_start

        # Initialize environment
        self.env = SpaceInvadersEnvironment(self.params)

        # Define the transitions
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))

        # Initialize Replay Memory
        self.memory = ReplayMemory(self.params, self.transition)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Initialize model
        if self.params.resumeTraining is False:
            print("Training New Model")
            self.model = DQN(self.params, self.env.action_space.n)
        else:
            print("Resuming Training")
            self.load_model(self.env.action_space.n)

        # Initilize Target Network
        self.target_network = DQN(self.params, self.env.action_space.n)
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.useGPU is True:
            print("Using GPU")
            try:
                self.model.cuda()
                self.target_network.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                self.target_network.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                self.target_network.cpu()
        else:
            print("Using CPU")

        # Setup optimizer
        self.optimizer = self.optimizer_select()

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=self.params.step_size,
                                                   gamma=self.params.decay_coeff)

        # Criterion, Huber Loss
        self.criterion = SmoothL1Loss()

    def update_epsilon(self, episode):
        self.epsilon = self.params.epsilon_end + (self.params.epsilon_start  - self.params.epsilon_end) * \
            np.exp(- episode / self.params.epsilon_decay)

    def train_model(self):
        avg_losses = np.zeros(self.params.num_episodes)
        episode_durations = np.zeros(self.params.num_episodes)
        for episode in range(self.params.num_episodes):
            print("Episode {}".format(episode + 1))

            # Update learning rate
            self.scheduler.step(episode)

            # Update epsilon
            self.update_epsilon(episode)

            print("Learning Rate = {}".format(self.optimizer.param_groups[0]['lr']))

            print("Epsilon = {}".format(self.epsilon))

            # Set mode to training
            self.model.train()

            # Go through the training set
            avg_losses[episode], episode_durations[episode] = self.train_episode()

            print("Average loss= {}".format(avg_losses[episode]))

            if (episode + 1) % 100 == 0:
                self.save_model(self.model.state_dict())
        # Close the environment
        self.env.close()
        # Saving trained model
        self.save_model(self.model.state_dict())
        return avg_losses, episode_durations

    def train_episode(self):
        # Initialize losses
        losses = 0.0
        # Initialize score
        score = 0
        # Initialize the environment and state
        self.env.reset()
        # Get the first 4 frames and initialize the state
        frames = self.env.get_frames(4)
        state = torch.cat(frames, dim=1)
        # Initiliaze the number of lives
        current_lives = 3
        previous_lives = current_lives
        # value by which the reward gets multiplied
        reward_multiplier = 1.0 / 30
        for step_index in range(1, self.params.num_steps_per_episode+1):
            #self.env.render()
            # Wrap the state in a Variable
            if self.useGPU is True:
                state = state.cuda()
            state = Variable(state)
            # Select and perform an action
            action = self.select_action_boltzmann(state)
            _, reward, done, info = self.env.step(action.data[0])
            score += reward
            reward = reward * reward_multiplier

            previous_lives = current_lives
            current_lives = info['ale.lives']

            if current_lives != previous_lives:
                reward_multiplier /= 2.0
                reward = -10.0

            # Observe new state
            if not done:
                frames = frames[1:] + self.env.get_frames(1)
                next_state = torch.cat(frames, dim=1)
            else:
                next_state = None

            reward = torch.Tensor([reward])
            if self.useGPU is True:
                reward = reward.cuda()
            reward = Variable(reward)
            reward = F.tanh(reward)

            # Store the transition in memory
            self.memory.push(state.data, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            losses += self.optimize_model()

            # Update the target network
            if (step_index + 1) % self.params.target_update == 0:
                self.target_network.load_state_dict(self.model.state_dict())

            if done is True:
                break

        print("Score = {}".format(score))
        # Compute the average loss for this epoch
        episode_duration = step_index + 1
        avg_loss = losses / episode_duration
        return avg_loss, episode_duration

    def optimize_model(self):
        if len(self.memory) < self.params.batch_size:
            return 0
        transitions = self.memory.sample(self.params.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        if self.useGPU is True:
            non_final_next_states = non_final_next_states.cuda()
        non_final_next_states = Variable(non_final_next_states)

        state_batch = torch.cat(batch.state, dim=0)
        if self.useGPU is True:
            state_batch = state_batch.cuda()
        state_batch = Variable(state_batch)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Determine the greedy batch policies using the policy network - argmax Q(s_t+1, a)
        # This is Double Q-Learning
        action_selection = self.model(non_final_next_states).max(1, keepdim=True)[1].detach()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.params.batch_size))
        # next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).gather(1,
                                                                                              action_selection).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.params.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Zero the optimizer gradient
        self.optimizer.zero_grad()

        # Backward step
        loss.backward()

        # Clip gradients
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        # Weight Update
        self.optimizer.step()

        if self.useGPU is True:
            torch.cuda.synchronize()

        return loss.data[0]

    def select_action_epsilon_greedy(self, state):
        # Choose the action
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            action = Variable(torch.LongTensor([[action]]))
        else:
            self.model.eval()
            action = self.model(state).max(1)[1].view(1, 1)
            self.model.train()
        return action

    def select_action_boltzmann(self, state):
        r""" Inspired from the implementation shown in this article:
        https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
        """
        self.model.eval()
        # Using epsilon as temperature to avoid having too many variables
        action_probabilities = self.model(state, self.epsilon).data[0].numpy()
        action = int(np.random.choice(action_probabilities.shape[0], 1, p=action_probabilities))
        action = Variable(torch.LongTensor([[action]]))
        self.model.train()
        return action

    def save_model(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        torch.save(self.serialize(),
                   os.path.join(self.params.savedModelDir, 'Trained_Model'
                                + '_' + time.strftime("%d.%m.20%y_%H.%M")))

    def load_model(self, num_actions, useGPU=False):
        package = torch.load(self.params.testModelPath, map_location=lambda storage, loc: storage)
        self.model = DQN.load_model(package, num_actions, useGPU)
        self.optimizer = self.optimizer_select()
        #parameters = package['params']
        #self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

    def serialize(self):
        model_is_cuda = next(self.model.parameters()).is_cuda
        model = self.model.cpu() if model_is_cuda else self.model
        package = {
            'state_dict': model.state_dict(),
            'params': self.params._asdict(),
            'optim_dict': self.optimizer.state_dict()
        }
        return package

    def optimizer_select(self):
        if self.params.optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'Adadelta':
            return optim.Adadelta(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.params.learning_rate,
                             momentum=self.params.momentum, nesterov=self.params.nesterov)
        else:
            raise NotImplementedError

