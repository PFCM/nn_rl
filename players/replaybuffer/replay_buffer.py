"""handles the replay buffer, writing and reading episodes to/from disk for
training on"""
import os
import logging

import tensorflow as tf


def get_next_filename(logdir):
    """gets the filename for the next set of actions.

    Args:
        logdir (str): where the files are stored.

    Returns:
        str: the next logical name.
    """
    files = sorted(os.listdir(logdir))
    # get rid of the extension
    fname = files[-1].split('.')[0]
    # and pull the number off the end
    num = int(fname.split('-')[-1])
    return os.path.join(logdir, 'transitions-{:05d}.tfrecords'.format(num+1))


class ReplayBuffer(object):
    """A persistent replay buffer for deep Q learning."""

    def __init__(self, logdir, actions_per_file=1000, max_files=100):
        """Sets up a buffer using a given directory"""
        self.logdir = logdir
        self.actions_per_file = actions_per_file
        self.max_files = max_files
        self._buffer = []

    def store(self, current_state, action, reward, previous_state):
        """Stores an action in the buffer. When the buffer is big enough, they
        are written to disk. If too many files are present, deletes the oldest
        """
        self._buffer.append((current_state, action, reward, previous_state))
        if len(self._buffer) >= self.actions_per_file:
            logging.info('saving %d actions', len(self._buffer))
            self._save_and_clear_buffer()

    def _save_and_clear_buffer(self):
        """saves the contents of the file to disk and clears out the buffer"""
        self._buffer = []

    def get_read_tensors(self, batch_size):
        """Gets a pipeline set up to read random batches from the buffer"""
        pass
