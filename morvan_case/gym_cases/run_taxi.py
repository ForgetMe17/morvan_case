import gym
from gym_cases.RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt


def run_maze():
    step = 0
    train_info = []
    for episode in range(2000):
        observation = env.reset()
        print('\rin episode %d' % episode, end='')
        ep_reward = 0
        ep_step = 0

        while True:
            ep_step += 1

            action = RL.choose_action(np.array(observation)[np.newaxis])

            observation_, reward, done, _ = env.step(action)
            ep_reward += reward

            RL.store_transition(np.array(observation)[np.newaxis], action, reward, np.array(observation_)[np.newaxis])

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                train_info.append((episode, ep_step, ep_reward/ep_step))
                break
            step += 1

    info = np.vstack(train_info)
    plt.figure()
    plt.subplot(121)
    plt.plot(info[:, 0], info[:, 1])
    plt.subplot(122)
    plt.plot(info[:, 0], info[:, 2])
    plt.show()


if __name__ == '__main__':
    env = gym.make('Taxi-v2', )
    RL = DeepQNetwork(
        n_states=1,
        n_actions=env.action_space.n,
        output_graph=False
    )
    run_maze()
