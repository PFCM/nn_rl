"""play some games, maybe even learn from them"""
import logging

import gym


import players


def record_transition(current_state, action, reward, next_state, terminal):
    """Records a transition into the replay buffer (this probably means
    waiting for enough of them and then writing to disk).

    Args:
        current_state: the starting state for the transition.
        action: the action taken by the agent.
        reward: the reward associated with this state/action pair.
        next_state: the subsequent state the world transitioned to.
        terminal: if the state turned out to be terminal.
    """
    logging.debug('transition:')
    logging.debug('  current_state: %s', current_state)
    logging.debug('         action: %s', action)
    logging.debug('         reward: %s', reward)
    logging.debug('     next_state: %s', next_state)

    players.replay_buffer.store(current_state, action, reward, next_state,
                                terminal)


def play_loop(env):
    """play for a while"""
    logging.info('getting player')
    with players.nn_player('/tmp/rl', 'mlp', env) as player:
        current_state = env.reset()
        for episode in range(1000):
            env.render()
            action = player(current_state)
            next_state, reward, done, info = env.step(action)
            # maybe record the transition
            record_transition(current_state, action, reward, next_state, done)
            current_state = next_state


def main():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pong-ram-v0')
    env.reset()

    play_loop(env)


if __name__ == '__main__':
    main()
