"""Contains some things we can use to play a game using neural networks.
"""
import logging

import tensorflow as tf

import players.nets as nets


USE_DEFAULT = 0


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


class NNPlayer(object):
    """Neural network player, loads up a net and maybe remembers how it's been
    going"""

    def __init__(self, model_dir, model, env, trajectory_saver=USE_DEFAULT):
        """Makes a new player.

        Args:
            model_dir (string): where the checkpoints to load are
            model (string): which model to use. For options, see `nets.py`.
            env (Environment): the gym environment in which we are to operate.
            trajectory_saver (Optional): something we can use to save
                transitions as we observe them. If None, transitions are not
                saved, if 0 then a default ReplayBuffer is created.
        """
        self.input_var = nets.get_input_for(env, 1)
        self.action_var = tf.squeeze(nets.get_net(model, input_var, env))


    def act(self, observation, session):
        """act on an observation"""
        pass

    def reward(self, reward):
        """receive a reward for the last executed action"""
        pass
