import random
from collections import deque


class ReplayMemory:
    """
    A buffer for the replay memory.
    """

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def clear(self):
        """
        Clear the buffer
        """
        self.buffer.clear()

    def add(self, state, vark_action, taxonomies_action, reward, next_state, done):
        """
        add state, action, reward, next state and done tuple to the buffer
        """
        self.buffer.appendleft((state, vark_action, taxonomies_action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample amini batch of a certain size from the buffer

        Paramters
        ---------

        batch_size : size of the mini batch to be sampled 
        """
        return random.sample(self.buffer, batch_size)

    @property
    def size(self):
        return len(self.buffer)
