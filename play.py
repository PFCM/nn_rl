"""play some games, maybe even learn from them"""
import gym


import players


def play_loop(env):
    """play for a while"""
    for episode in range(1000):
        env.render()
        env.step(env.action_space.sample())


def main():
    env = gym.make('Pong-ram-v0')
    env.reset()

    play_loop(env)


if __name__ == '__main__':
    main()
