import numpy as np
import tensorflow as tf
np.random.seed(1)

class QLearning:
	def __init__(self,
				n_features,
				n_actions,
				e_max=0.9,
				e_increment=1e-4,
				time_decay=0.98,
				learning_rate=0.01
				):
		self.n_actions = n_actions
		self.n_features = n_features
		self.epsilo = 0
		self.e_max = e_max
		self.e_increment = e_increment
		self.gamma = time_decay
		self.lr = learning_rate
		
		self.q_table = np.zeros((n_features, n_actions))
	
	def choose_action(self, observation):
		if np.random.uniform() > self.epsilo:
			action = np.random.choice(self.n_actions)
		else
			action = np.argmax(q_table[observation, :])
		return action
		
	def learn(self, observation, action, reward, observation_):
		q_pred = self.q_table[observation, action]
		q_next = np.argmax(self.q_table[observation_, :])
		q_target = reward + self.gamma*q_next
		self.q_table[observation, action] += self.lr*(q_target-q_pred)
