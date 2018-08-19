from Actor_critc.RL_brain import ActorCritic
import gym

env = gym.make('Taxi-v2')
env = env.unwrapped
env.seed(1)

RL = ActorCritic(
    n_features=env.observation_space.n,
    n_actions=env.action_space.n
)

for episode in range(1000):
    reward_count = 0
    action_rate = 0
    ep_steps = 0
    force_done = False
    observation = env.reset()
    while True:
        ep_steps += 1

        action, prob = RL.choose_action(observation)

        observation_, reward, done, _ = env.step(action)

        RL.store_transition(observation=observation,
                            action=action,
                            reward=reward,
                            observation_=observation_)
        reward_count += reward
        action_rate += prob
        if ep_steps > 10000:
            force_done = True
        if done or force_done:
            t_s = RL.total_step
            m_s = RL.memory_size
            b_s = RL.batch_size
            m_i = RL.memory_index
            if done:
                if t_s < m_s:
                    learn_time = 1 + int(t_s / b_s * 2)
                    batch_start = 0
                    batch_end = t_s
                else:
                    learn_time = min(int(ep_steps / b_s), 30)
                    batch_start = int(m_i - learn_time/2 * b_s)
            else:
                learn_time = 1
                batch_start = m_i - 2 * b_s
            for _ in range(int(learn_time)):
                RL.learn(batch_start, m_i)
            print('episode %d   ep_steps %d   mean_reward %f   mean_action_rate %f   ep_learn_time %d'
                  % (episode, ep_steps, reward_count/ep_steps, action_rate/ep_steps, learn_time))
            break
        observation = observation_
