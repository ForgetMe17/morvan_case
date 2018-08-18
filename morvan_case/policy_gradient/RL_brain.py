import tensorflow as tf
import numpy as np


class PolicyGradient:
    def __init__(self,
                 n_features,
                 n_actions,
                 learning_rate=0.01,
                 time_decay=0.99
                 ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = time_decay

        self.ep_ob = []
        self.ep_a = []
        self.ep_r = []

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        pass

    def _build_net(self):
        # policy net
        with tf.variable_scope('Inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        with tf.variable_scope('Policy_net'):
            w_initializer, b_initializer = tf.initializers.random_normal(0, 0.05), tf.initializers.constant(0)
            with tf.variable_scope('fc1'):
                l1 = tf.layers.dense(self.s, 20, activation=tf.nn.tanh, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer)
                self.fc1_out = l1
            with tf.variable_scope('fc2'):
                l2 = tf.layers.dense(l1, self.n_actions, activation=None,
                                     kernel_initializer=w_initializer, bias_initializer=b_initializer)
                self.fc2_out = l2
            with tf.variable_scope('policy_distribution'):
                self.action_dis = tf.nn.softmax(l2)
        with tf.variable_scope('loss'):
            self.a = tf.placeholder(tf.int32, [None, 1], name='action')
            self.s_v = tf.placeholder(tf.float32, [None, 1], name='state_value')
            log_prob = tf.reduce_sum(-tf.log(self.action_dis)*tf.one_hot(self.a, self.n_actions), axis=1)
            self.log_prob = log_prob
            self.loss = tf.reduce_mean(log_prob*self.s_v, axis=1)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        step_prob = self.sess.run(self.action_dis, feed_dict={
            self.s: observation[np.newaxis, :]
        })
        # print(self.sess.run(self.fc1_out, feed_dict={
        #     self.s: observation[np.newaxis, :]}))
        if not (step_prob == step_prob).all():
            print(observation)
            print(self.sess.run(self.fc1_out, feed_dict={
                self.s: observation[np.newaxis, :]
            }))
            print(step_prob)
            exit()

        action = np.random.choice(range(step_prob.shape[1]), p=step_prob.ravel())
        return action, step_prob[0, action]

    def store_transition(self, ob, a, r):
        self.ep_ob.append(ob)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def _discounted_reward(self):
        dis_r = np.zeros_like(self.ep_r)
        running_add = 0
        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add*self.gamma + self.ep_r[t]
            dis_r[t] = running_add
        dis_r -= np.mean(dis_r)
        dis_r /= np.std(dis_r)
        return dis_r

    def learn(self):
        discounted_reward = self._discounted_reward()

        _, ep_loss, log_prob = self.sess.run(
            [self.train_op, self.loss, self.log_prob], feed_dict={
                self.s: np.vstack(self.ep_ob),
                self.a: np.vstack(self.ep_a),
                self.s_v: np.vstack(discounted_reward)
            })

        self.ep_ob = []
        self.ep_a = []
        self.ep_r = []
        return np.max(discounted_reward), np.mean(ep_loss), len(discounted_reward), log_prob
