"""play some games, maybe even learn from them"""
import logging

import tensorflow as tf
import gym


import players


def play_loop(env):
    """play for a while"""
    logging.info('getting player')
    player = players.nn_player('mlp', env)
    sv = tf.train.Supervisor(logdir='/tmp/rl')
    with sv.managed_session() as session:
        current_state = env.reset()
        for episode in range(1000):
            env.render()
            action = player.act(current_state, session)
            next_state, reward, done, info = env.step(action)
            player.reward(reward)

            logging.debug('transition:')
            logging.debug('  current_state: %s', current_state)
            logging.debug('         action: %s', action)
            logging.debug('         reward: %s', reward)
            logging.debug('     next_state: %s', next_state)

            current_state = next_state


def main():
    logging.basicConfig(level=logging.INFO)
    gym.logger.setLevel(logging.INFO)
    env = gym.make('Pong-v0')
    env.reset()

    play_loop(env)


if __name__ == '__main__':
    main()
