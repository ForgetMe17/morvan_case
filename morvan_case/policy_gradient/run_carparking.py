from policy_gradient.RL_brain import PolicyGradient
import gym
import numpy as np
import matplotlib.pyplot as plt


def run():
    cost_list = []
    step_list = []
    ep_step = 0
    for ep in range(1000):
        observation = env.reset()
        action_rate = []
        ep_step = 0
        while True:
            ep_step += 1
            # if ep_step > 1000:
            #     env.render()
            action, prob = RL.choose_action(np.array(observation)[np.newaxis])
            action_rate.append(prob)

            observation_, reward, done, _ = env.step(action)
            # (x, _, theta, _) = observation_
            # if abs(x) > 0.8*x_thre:
            #     r1 = -abs(x / x_thre)
            # else:
            #     r1 = 0
            # r2 = -abs(theta / theta_thre)
            # reward = r2 + r1

            RL.store_transition(observation, action, np.float32(reward))
            observation = observation_

            if done or ep_step > 10000:
                if ep_step < 10000:
                    _, ep_loss, _, _ = RL.learn()
                    cost_list.append(ep_loss)
                step_list.append(ep_step)
                print('in episode %d  ep steps %d   mean action rate %f' % (ep, ep_step, np.mean(action_rate)))
                break
    plt.figure()
    plt.plot(cost_list)
    plt.show()


if __name__ == '__main__':
    env = gym.make('Taxi-v2')
    env = env.unwrapped
    RL = PolicyGradient(n_features=env.observation_space.n,
                        n_actions=env.action_space.n,
                        time_decay=0.9,
                        learning_rate=0.01)
    run()