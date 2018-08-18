from policy_gradient.RL_brain import PolicyGradient
import gym
import numpy as np
import matplotlib.pyplot as plt


def run():
    cost_list = []
    step_list = []
    for ep in range(5000):
        observation = env.reset()
        action_rate = []
        while True:
            if ep % 100 == 0:
                env.render()

            action, prob = RL.choose_action(observation)
            action_rate.append(prob)

            observation_, reward, done, _ = env.step(action)
            # (x, _, theta, _) = observation_
            # if abs(x) > 0.8*x_thre:
            #     r1 = -abs(x / x_thre)
            # else:
            #     r1 = 0
            # r2 = -abs(theta / theta_thre)
            # reward = r2 + r1

            RL.store_transition(observation, action, reward)
            observation = observation_

            if done:
                _, ep_loss, ep_step, _ = RL.learn()
                cost_list.append(ep_loss)
                step_list.append(ep_step)
                print('in episode %d  ep steps %d  ep loss  %f  mean action rate %f' % (ep, ep_step, ep_loss, np.mean(action_rate)))
                break
    plt.figure()
    plt.plot(cost_list)
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.seed(1)
    RL = PolicyGradient(n_features=env.observation_space.shape[0],
                        n_actions=env.action_space.n)
    x_thre = env.env.x_threshold
    theta_thre = env.env.theta_threshold_radians
    run()