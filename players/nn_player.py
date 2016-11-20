"""Contains some things we can use to play a game using neural networks.
"""
import logging
from contextlib import contextmanager

import tensorflow as tf

import players.nets as nets


@contextmanager
def nn_player(model_dir, model, env, save_summaries=False):
    """Context manager, returns a callable neural network policy.

    Args:
        model_dir (string): where the checkpoints to load are
        model (string): which model to use. For options, see `nets.py`.
        env (Environment): the gym environment in which we are to operate.
        save_summaries (Optional[bool]): whether to save summaries as we
            go.
    """
    input_var = nets.get_input_for(env, 1)
    action_var = nets.get_net(model, input_var, env)
    action_var = tf.squeeze(action_var)

    # now we have a model, get a managed session
    supervisor = tf.train.Supervisor(
        logdir=model_dir,
        saver=None, summary_op=0 if save_summaries else None)

    logging.info('initializing/loading')
    with supervisor.managed_session() as sess:

        def act(obs):
            """Get a move given an observation"""
            return sess.run(action_var, {input_var: obs.reshape((1, -1))})

        yield act
