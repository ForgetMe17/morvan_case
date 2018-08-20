import gym
from RL_brain import QLearning

env = gym.make('Taxi-v2')

RL = QLearning(n_features=env.observation_space.n,
				n_actions=env.action_space.n)
				
for episode in range(100):
	ep_step = 0
	ep_reward = 0
	ob = env.reset()
	
	while True:
		ep_step += 1
		
		action = RL.choose_action(ob)
		
		ob_, reward, done, _ = env.step(action)
		ep_reward += reward
		
		RL.learn(ob, action, reward, ob_)
		
		if done:
			print('episode %d   ep_step %d   ep_reward %f' % (episode, ep_step, ep_reward/ep_step))
			break