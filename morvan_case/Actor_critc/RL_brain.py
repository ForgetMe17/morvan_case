import tensorflow as tf
import numpy as np


class ActorCritic:
    def __init__(self,
                 n_features,
                 n_actions,
                 time_decay=0.98,
                 learning_rate=0.005,
                 batch_size=50,
                 memory_size=2000):

        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = time_decay
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_index = 0
        self.memory = np.zeros((self.memory_size, 4))
        self.total_step = 0

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.variable_scope('Inputs'):
            self.ob = tf.placeholder(tf.int32, [None, ], name='state')
            s = tf.one_hot(self.ob, self.n_features)
            self.a = tf.placeholder(tf.int32, [None, ], name='action')
            self.r = tf.placeholder(tf.float32, [None, ], name='reward')
            # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
            self.v_ = tf.placeholder(tf.float32, [None, ], name='next_state_value')
        with tf.variable_scope('Actor'):
            actor_w_init, actor_b_init = tf.initializers.random_normal(0, 0.01), tf.initializers.constant(0)
            with tf.variable_scope('fc1'):
                l1 = tf.layers.dense(
                    inputs=s,
                    units=100,
                    kernel_initializer=actor_w_init,
                    bias_initializer=actor_b_init,
                    activation=tf.nn.tanh
                )
            with tf.variable_scope('fc2'):
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=self.n_actions,
                    kernel_initializer=actor_w_init,
                    bias_initializer=actor_b_init,
                    activation=None
                )
            with tf.variable_scope('action_dis'):
                self.action_dis = tf.nn.softmax(l2)
        with tf.variable_scope('Critic'):
            critic_w_init, critic_b_init = tf.initializers.random_normal(0, 0.01), tf.initializers.constant(0)
            with tf.variable_scope('l1'):
                critic_l1 = tf.layers.dense(
                    inputs=s,
                    units=20,
                    kernel_initializer=critic_w_init,
                    bias_initializer=critic_b_init,
                    activation=tf.nn.relu
                )
            with tf.variable_scope('state_value'):
                self.critic_value = tf.layers.dense(
                    inputs=critic_l1,
                    units=1,
                    kernel_initializer=critic_w_init,
                    bias_initializer=critic_b_init,
                    activation=None
                )
            with tf.variable_scope('critic_train'):
                self.td_error = self.r + self.gamma*self.v_ - self.critic_value
                critic_loss = tf.square(self.td_error)
                self.critic_train_op = tf.train.AdamOptimizer(self.lr).minimize(critic_loss)
        with tf.variable_scope('Actor_loss'):
            neg_prob_log = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.a, logits=l2)
            actor_loss = tf.reduce_mean(neg_prob_log*self.td_error)
            self.actor_train_op = tf.train.AdamOptimizer(self.lr).minimize(actor_loss)

    def choose_action(self, observation):
        action_dis = self.sess.run(self.action_dis, feed_dict={
            self.ob: np.array([observation])
        })
        action_dis = action_dis.ravel()
        action = np.random.choice(range(self.n_actions), p=action_dis.ravel())
        return action, action_dis[action]

    def learn(self, slice_start, slice_end):
        batch_slice = np.random.random_integers(slice_start, slice_end, self.batch_size)
        batch = self.memory[batch_slice]
        value = self.sess.run(self.critic_value, feed_dict={
            self.ob: batch[:, 3]
        })
        td_error, _, _ = self.sess.run([self.td_error, self.actor_train_op, self.critic_train_op],
                                       feed_dict={
                                          self.ob: batch[:, 0],
                                          self.a: batch[:, 1],
                                          self.r: batch[:, 2],
                                          self.v_: value[:, 0]
                                      })
        return td_error

    def store_transition(self, observation, action, reward, observation_):
        self.total_step += 1
        self.memory_index = (self.memory_index+1) % self.memory_size
        self.memory[self.memory_index] = (observation, action, reward, observation_)
