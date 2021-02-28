import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from train_env1_mini_16_16_16_s import working_space_train1
from train_env2_mini_16_16_16_c import working_space_train2
from train_env3_mini_16_16_16_n import working_space_train3

############################## Hyper Parameters ###############################

# For training
n_actions = 6
n_features = 14

n_envs = 1 # env_training1 + env_training2 + env_training3
max_episode = 300 # 100, 500, 2000, 5000
output_graph = True
training_print = 20

n_hidden = 10
learning_rate = 0.001
reward_decay = 0.9
e_greedy = 0.95
replace_target_iter = 300
memory_size = 10000
batch_size = 32
e_greedy_increment = 0.001

# Save network model
save_step_interval = 10000
save_path = r"tf_model\tf_model.ckpt"

##################### Class Deep Q Network (off-policy) #######################

class deep_q_network:
    def __init__(self):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        # Initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Two learning nets: target_net, evaluate_net
        self._build_net()
        target_parameters = tf.get_collection('target_net_parameters')
        evaluate_parameters = tf.get_collection('evaluate_net_parameters')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_parameters, evaluate_parameters)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("tf_logs/", self.sess.graph)
            
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(self.init)
        self.cost_history = []
    
    # Method: Fixed-Target
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # state s as input
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        with tf.variable_scope('eval_net'):
            collection_names = ['evaluate_net_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = self.n_hidden
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=collection_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=collection_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=collection_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=collection_names)
                self.q_eval = tf.matmul(l1, w2) + b2        

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # new state s_ as input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # collections_names are the collections to store variables
            collection_names = ['target_net_parameters', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer = w_initializer, collections = collection_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer = b_initializer, collections = collection_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer = w_initializer, collections = collection_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer = b_initializer, collections = collection_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
#            print('\n----------------[ target parameters replaced ]----------------\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_: batch_memory[:, -self.n_features:],  # fixed params
                                                  self.s: batch_memory[:, :self.n_features]})  # newest params

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_history.append(self.cost)
        
        # Save the training results every certain learning steps
        if (self.learn_step_counter + 1) % save_step_interval == 0:
            self.saver.save(self.sess, save_path)
            print('\n----------------[ model saved ]----------------\n')
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1                    
        
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.title('Training Loss Diagram',fontsize='large')
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.savefig(r'result_training\training_loss_diagram')
        plt.show()

############################## Class Training #################################

class training():
    
    def __init__(self):
        self.rl = deep_q_network()
        self.total_steps = 0
        self.accuracy = []
        self.fault = []
        self.envs_counter = 0
    
    def train_in(self, env):
        if env == 'env_training1':
            env_training = working_space_train1()
        elif env == 'env_training2':
            env_training = working_space_train2()
        elif env == 'env_training3':
            env_training = working_space_train3()
        else:
            print('There is no such environment found, please check the environemnt name.')
            return
        self.envs_counter += 1
        print('\n{ Training in %s }' % env)
        t0 = time.time()
        to_cut, to_keep = env_training.info()
        print('\n')
        for episode in range(max_episode):
            s = env_training.reset()
            episode_r = 0
            episode_steps = 0
            while True:
                a = self.rl.choose_action(s)
                s_, step_r, e_num, f_num, m_complete, m_incomplete, out, end, moves = env_training.step(a)
                self.rl.store_transition(s, a, step_r, s_)
                episode_r += step_r
                if self.total_steps > 1000:
                    self.rl.learn()
                if end:
                    a_rate = round((e_num / to_cut), 2)
                    f_rate = round((f_num / to_keep), 2)
                    self.accuracy.append(a_rate)
                    self.fault.append(f_rate)
                    if (episode + 1) * training_print % max_episode == 0:
                        print('Episode:', episode,'Steps:', episode_steps, 'Reward:', round(episode_r, 2),
                              'Accuracy:', a_rate, 'Fault:', f_rate)
                    break
                s = s_
                self.total_steps += 1
                episode_steps += 1
        t_ = time.time()
        delta_t = round((t_ - t0), 6)
        print('\nTraining time for %d episodes:' % max_episode, delta_t, 's')
        
    def draw_accuracy(self):
        x = np.arange(self.envs_counter * max_episode)
        y = self.accuracy
        plt.scatter(x, y, s = 8, alpha = 0.6)
        plt.title('Training Accuracy-rate Diagram',fontsize='large')
        plt.ylabel('accuracy rate')
        plt.xlabel('episodes')
        plt.savefig(r'result_training\training_accuracy_diagram')
        plt.show()
    
    def draw_fault(self):
        x = np.arange(self.envs_counter * max_episode)
        y = self.fault
        plt.scatter(x, y, s = 8, alpha = 0.6)
        plt.title('Training Fault-rate Diagram',fontsize='large')
        plt.ylabel('fault rate')
        plt.xlabel('episodes')
        plt.savefig(r'result_training\training_fault_diagram')
        plt.show()
    
    def train_result(self):
        self.rl.plot_cost()
        self.draw_accuracy()
        self.draw_fault()
        sum_a = 0
        for acc in self.accuracy:
            sum_a += acc
        average_accuracy = round((sum_a / (max_episode * n_envs)), 4)
        sum_f = 0
        for fau in self.fault:
            sum_f += fau
        average_fault = round((sum_f / (max_episode * n_envs)), 4)
        print('\nAverage accuracy rate:', average_accuracy,
              '\nAverage fault rate:', average_fault)
        
################################# Execution ###################################

tr = training()
tr.train_in('env_training1')
#tr.train_in('env_training2')
#tr.train_in('env_training3')
tr.train_result()