"""general things"""
import random


def epsilon_wrapper(player, epsilon, env):
    """Gets a wrapper around a player which may just choose a random action
    instead.

    Args:
        player (callable): the player we want to actually use.
        epsilon (float): the probability of choosing uniformly at random.

    Returns:
        callable: the wrapped policy.
    """
    action_space = env.action_space

    def _with_uniform_random_chance(obs):
        if random.random() <= epsilon:
            return action_space.sample()
        return player(obs)

    return _with_uniform_random_chance
