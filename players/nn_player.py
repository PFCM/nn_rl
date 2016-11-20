"""Contains some things we can use to play a game using neural networks.
"""
import logging

import tensorflow as tf

import nets


class NNPlayer(object):
    """Neural network player"""

    def __init__(self, model_dir, model, env, save_summaries=False):
        """Callable neural network policy.

        Args:
            model_dir (string): where the checkpoints to load are
            model (string): which model to use. For options, see `nets.py`.
            env (Environment): the gym environment in which we are to operate.
            save_summaries (Optional[bool]): whether to save summaries as we
                go.
        """
        self._model_dir = model_dir
        self._model = model

        self._input_var = get_input_for(env, 1)
        self._action_var = get_net(model, self._input_var, env)
        self._action_var = tf.squeeze(self._action_var)

        # now we have a model, get a managed session
        self._supervisor = tf.train.Supervisor(
            saver=None, summary_op=0 if save_summaries else None)

        logging.info('initializing/loading')
        self._sess = self._supervisor.managed_session()

    def __call__(self, obs):
        """Get a move given an observation"""
        return self._sess.run(self._action_var, {self._input_var: obs})
