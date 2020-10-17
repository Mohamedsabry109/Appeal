import random
from collections import deque


class ReplayMemory:
    """
    A buffer for the replay memory.
    """

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def clear(self):
        self.buffer.clear()

    def add(self, state, vark_action, taxonomies_action, reward, next_state, done):
        self.buffer.appendleft((state, vark_action, taxonomies_action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    @property
    def size(self):
        return len(self.buffer)
