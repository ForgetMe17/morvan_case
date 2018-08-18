import gym
from gym_cases.RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import time


def run_maze():
    step = 0
    train_info = []
    for episode in range(n_episode):
        observation = env.reset()
        ep_reward = 0
        ep_step = 0
        print('\rin episode %d' % episode,  end='')

        while True:
            ep_step += 1
            if episode % 500 == 0:
                env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, _ = env.step(action)
            (pos, v) = observation_
            r1 = (pos-min_position)/(max_position-min_position)
            r2 = abs(v)/max_speed
            reward = r1 + r2

            ep_reward += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                train_info.append((episode, ep_step, ep_reward/ep_step, RL.e_greedy))
                break
            step += 1

    info = np.vstack(train_info)
    plt.figure()
    plt.subplot(221)
    plt.plot(info[:, 0], info[:, 1])
    plt.title('episode steps')
    plt.subplot(222)
    plt.plot(info[:, 0], info[:, 2])
    plt.title('ep average reward')
    plt.subplot(223)
    plt.plot(info[:, 0], info[:, 3])
    plt.title('ep policy')
    plt.subplot(224)
    plt.plot(RL.cost_list)
    plt.title('ep cost')
    plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    RL = DeepQNetwork(
        n_states=env.reset().shape[0],
        n_actions=env.action_space.n,
        output_graph=False,
        e_greedy_increment=1e-6
    )
    n_episode = 5000
    max_speed = env.env.max_speed
    min_position = env.env.min_position
    max_position = env.env.max_position
    run_maze()
