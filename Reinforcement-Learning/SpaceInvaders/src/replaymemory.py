import random


class ReplayMemory(object):
    def __init__(self, parameters, transition):
        self.params = parameters
        self.capacity = self.params.replayMemoryCapacity
        self.transition = transition
        self.memory = []
        self.position = 0

    def push(self, *args):
        r"""Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        for i in range(len(self.memory)):
            yield self.memory[i]
