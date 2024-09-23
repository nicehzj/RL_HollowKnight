import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Buffer:
    def __init__(self, size: int):
        assert size > 0
        self.buffer = deque([], maxlen=size)
        self.maxlen = size
    
    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def is_full(self):
        return self.maxlen == len(self.buffer)

    def __len__(self):
        return len(self.buffer)

