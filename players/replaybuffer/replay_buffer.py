"""handles the replay buffer, writing and reading episodes to/from disk for
training on"""
import os

import tensorflow as tf


class ReplayBuffer(object):
    """A persistent replay buffer for deep Q learning."""

    def __init__(self, logdir, actions_per_file=1000, max_files=100):
        """Sets up a buffer using a given directory"""
        pass

    def store(self, current_state, action, reward, previous_state):
        """Stores an action in the buffer. When the buffer is big enough, they
        are written to disk. If too many files are present, deletes the oldest
        """
        pass

    def get_read_tensors(self, batch_size):
        """Gets a pipeline set up to read random batches from the buffer"""
        pass
