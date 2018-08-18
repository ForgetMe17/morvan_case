import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(
            self,
            n_states,
            n_actions,
            reward_decay=0.9,
            learning_rate=1e-2,
            memomery_size=2000,
            e_greedy_max=0.9,
            replace_iter=300,
            batch_size=50,
            e_greedy_increment=0.000001,
            output_graph=False
            ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_decay = reward_decay
        self.lr = learning_rate
        self.memory_size = memomery_size
        self.e_greedy_max = e_greedy_max
        self.replace_iter = replace_iter
        self.learn_step_count = 0
        self.output_graph = output_graph
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.e_greedy = 0 if self.e_greedy_increment else self.e_greedy_max
        self._build_net()
        self.cost_list = []

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('soft_replacemet'):
            self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.memory = np.zeros((self.memory_size, self.n_states*2+2))

        self.sess = tf.Session()
        if self.output_graph:
            self.writer = tf.summary.FileWriter('./log', self.sess.graph)
            self.addsummary_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # inputs
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), \
                                       tf.constant_initializer(0)

        with tf.variable_scope('eval_net'):
            eval_l1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer, name='eval_l1')
            self.q_eval = tf.layers.dense(eval_l1, self.n_actions, tf.nn.relu,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_eval')

        with tf.variable_scope('target_net'):
            t_l1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='target_l1')
            self.q_next = tf.layers.dense(t_l1, self.n_actions, tf.nn.relu,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.reward_decay * tf.reduce_max(self.q_next, axis=1,
                                                                    name='q_target')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_index = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_index)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a),
                                       name='TD_error')
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.e_greedy:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        if self.e_greedy_increment:
            self.e_greedy += self.e_greedy_increment if self.e_greedy < self.e_greedy_max else 0

        return action

    def learn(self):
        if self.learn_step_count % self.replace_iter == 0:
            self.sess.run(self.replace_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_states],
                self.a: batch_memory[:, self.n_states],
                self.r: batch_memory[:, self.n_states+1],
                self.s_: batch_memory[:, -self.n_states:]
            })
        self.cost_list.append(cost)
        if self.output_graph:
            summary = self.sess.run(
                self.addsummary_op,
                feed_dict={
                    self.s: batch_memory[:, :self.n_states],
                    self.a: batch_memory[:, self.n_states],
                    self.r: batch_memory[:, self.n_states + 1],
                    self.s_: batch_memory[:, -self.n_states:]
                })
            self.writer.add_summary(summary, self.learn_step_count)

        self.learn_step_count += 1
