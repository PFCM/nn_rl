"""Contains some things we can use to play a game using neural networks.
"""
import logging

import tensorflow as tf

import players.nets as nets
import players.replaybuffer as replaybuffer


USE_DEFAULT = 0


class NNPlayer(object):
    """Neural network player, loads up a net and maybe remembers how it's been
    going"""

    def __init__(self, model, env, trajectory_saver=USE_DEFAULT):
        """Makes a new player.

        Args:
            model (string): which model to use. For options, see `nets.py`.
            env (Environment): the gym environment in which we are to operate.
            trajectory_saver (Optional): something we can use to save
                transitions as we observe them. If None, transitions are not
                saved, if 0 then a default ReplayBuffer is created.
        """
        self.input_var = nets.get_input_for(env, 1)
        self.action_var = tf.squeeze(nets.get_net(model, self.input_var, env))

        if trajectory_saver == USE_DEFAULT:
            self.trajectory_saver = replaybuffer.ReplayBuffer(
                '/tmp/rl/replays')
        else:
            self.trajectory_saver = trajectory_saver
        self._current_state = None

    def act(self, obs, session):
        """act on an observation"""
        self._last_action = session.run(
            self.action_var, {self.input_var: obs.reshape((1, -1))})
        self._last_state = self._current_state
        self._current_state = obs.reshape((1, -1))
        return self._last_action

    def reward(self, reward):
        """receive a reward for the last executed action"""
        if self.trajectory_saver:
            self.trajectory_saver.store(self._current_state,
                                        self._last_action, reward,
                                        self._last_state)

nn_player = NNPlayer
