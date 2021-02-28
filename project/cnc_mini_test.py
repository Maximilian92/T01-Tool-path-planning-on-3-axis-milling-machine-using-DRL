import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
## This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from test_env1_mini_16_16_16 import working_space_test1
from test_env2_medium_40_40_20 import working_space_test2
from test_env3_large_60_60_30 import working_space_test3

############################## Hyper Parameters ###############################

n_actions = 6
n_features = 14
n_hidden = 10
n_l1 = n_hidden
restore_path = "tf_model/tf_model.ckpt"

env_number = 'env_test3'
max_test_steps = 1000
visulization = False
ani_interval = 200
test_num = 10 if not visulization else 1
print_test_detail = not visulization
test_print = 2 if not visulization else 1

################################# Class Test ##################################

class test():
    
    def __init__(self):
        self.sess = tf.Session()
        self.get_parameters()

    def get_parameters(self):
        self.w1 = tf.Variable(np.arange(n_features * n_l1).reshape(n_features, n_l1), dtype = tf.float32, name="w1")
        self.b1 = tf.Variable(np.arange(n_l1).reshape(1, n_l1), dtype = tf.float32, name="b1")
        self.w2 = tf.Variable(np.arange(n_l1 * n_actions).reshape(n_l1, n_actions), dtype = tf.float32, name="w2")
        self.b2 = tf.Variable(np.arange(n_actions).reshape(1, n_actions), dtype = tf.float32, name="b2")
        saver = tf.train.Saver({'eval_net/l1/w1': self.w1,
                                'eval_net/l1/b1': self.b1,
                                'eval_net/l2/w2': self.w2,
                                'eval_net/l2/b2': self.b2})
        saver.restore(self.sess, restore_path)
    
    def judge_goback(self, action_memory):
        if len(action_memory) > 2:
            a1 = action_memory[-2]
            a2 = action_memory[-1]
            goback = a1 + a2 == 1 or a1 + a2 == 9 or a1 * a2 == 6
        else:
            goback = False
        return goback      
    
    def choose_action_test(self, observation, action_memory):
        observation = observation[np.newaxis, :]
        state = tf.placeholder(tf.float32, [None, n_features], name='state')
        l1 = tf.nn.relu(tf.matmul(state, self.w1) + self.b1)
        action_value = tf.matmul(l1, self.w2) + self.b2
        a_value = self.sess.run(action_value, feed_dict = {state: observation})
        action = np.argmax(a_value)
        action_memory.append(action)
        goback = self.judge_goback(action_memory)
        if goback:
            action = np.argsort(a_value)[0, np.random.randint(3, n_actions - 2)]
#            action = 0
        return action
    
    def choose_action_random(self):
        action = np.random.randint(0, n_actions)
        return action
    
    def draw_accuracy(self, accuracy_test, average_accuracy_test):
        x = np.arange(test_num)
        y = accuracy_test
        y_average = average_accuracy_test
        plt.scatter(x, y, s = 8, alpha = 0.6)
        plt.plot(x, y_average, color = 'red')
        plt.title('Test Accuracy-rate Diagram',fontsize='large')
        plt.ylabel('accuracy rate')
        plt.xlabel('episodes')
        plt.savefig(r'result_test\test_accuracy_diagram')
        plt.show()
    
    def draw_fault(self, fault_test, average_fault_test):
        x = np.arange(test_num)
        y = fault_test
        y_average = average_fault_test
        plt.scatter(x, y, s = 8, alpha = 0.6)
        plt.plot(x, y_average, c = 'red')
        plt.title('Test_Fault-rate Diagram',fontsize='large')
        plt.ylabel('fault rate')
        plt.xlabel('episodes')
        plt.savefig(r'result_test\test_fault_diagram')
        plt.show()
    
    def test_result(self, accuracy_test, fault_test, time_test, average_accuracy_test, average_fault_test):
        if not visulization:
            self.draw_accuracy(accuracy_test, average_accuracy_test)
            self.draw_fault(fault_test, average_fault_test)
        sum_t_test = 0
        for t_test in time_test:
            sum_t_test += t_test
        average_time_test = round((sum_t_test / test_num), 6)
        sum_a_test = 0
        for a_test in accuracy_test:
            sum_a_test += a_test
        average_accuracy_test = round((sum_a_test / test_num), 4)
        sum_f_test = 0
        for f_test in fault_test:
            sum_f_test += f_test
        average_fault_test = round((sum_f_test / test_num), 4)
        print('\nAverage processing time: ', average_time_test, 's', 
              '\nAverage accuracy rate:', average_accuracy_test,
              '\nAverage fault rate:', average_fault_test)
    
    def visualize(self, env_test, file_name, *args):        
        size_x, size_y, size_z = env_test.result_space.shape
        x, y, z = np.indices((size_x, size_y, size_z))
        voxels_cut = env_test.result_space_c
        voxels_keep = env_test.result_space_p
        voxels = voxels_cut | voxels_keep
        colors = np.empty(voxels.shape, dtype=object)
        colors[voxels_cut] = '#FFD65DC0'
        colors[voxels_keep] = '#7A88CCC0'
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_title('Visualization of workspace (%s)' % file_name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z") 
        ax.voxels(voxels, facecolors = colors, edgecolor = 'k')
        if file_name == 'After processing':
            accuracy, fault = args
            result_text = 'Accuracy: %s\nFault: %s' % (accuracy, fault)
            ax.text(0, 0, size_z, result_text, color = 'black')
        plt.savefig(r'result_test\%s' % file_name)
        plt.show()
        
    def animate_voxel(self, tracks_test, env_test):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
#        ax.set_aspect('equal')
        size_x, size_y, size_z = env_test.result_space_ani.shape
        x, y, z = np.indices((size_x, size_y, size_z))
        def update(num):
            ax.cla()
            ax.set_title('Animation of processing')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")  
            voxels_cut = env_test.result_space_c_ani
            voxels_keep = env_test.result_space_p_ani
            voxels_cutter = np.zeros((size_x, size_y, size_z), dtype = bool)
            for h in range(env_test.cutter_h):
                track_x, track_y, track_z = tracks_test[num]
                voxels_cutter[track_x, track_y, track_z + h] = True
                if voxels_cut[track_x, track_y, track_z + h]:
                    voxels_cut[track_x, track_y, track_z + h] = False
                if voxels_keep[track_x, track_y, track_z + h]:
                    voxels_keep[track_x, track_y, track_z + h] = False
            voxels = voxels_cut | voxels_keep | voxels_cutter
            colors = np.empty(voxels.shape, dtype=object)
            colors[voxels_cut] = '#FFD65DC0' # yellow transparent
            colors[voxels_keep] = '#7A88CCC0' # purple transparent # '#1f77b430' # blue transparent
            colors[voxels_cutter] = 'gray' 
            ax.voxels(voxels, facecolors = colors, edgecolor='k')
        ani_steps = (tracks_test.shape)[0]
        ani_show = animation.FuncAnimation(fig, update, ani_steps, 
                                           interval = ani_interval, repeat = False)
        ani_show.save(r'result_test\test_animation.gif', writer = 'pillow')
        plt.show()
        
    def animate_track(self, tracks_test, env_test):
        size_x, size_y, size_z = env_test.result_space_ani.shape
        fig = plt.figure()
        ax = p3.Axes3D(fig)   
        frame = np.transpose(tracks_test)
        data = [frame]
        tracks = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
        ax.set_title('Animation of processing tracks')
        ax.set_xlim3d([0, size_x])
        ax.set_xlabel('X')
        ax.set_ylim3d([0, size_y])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0, size_z])
        ax.set_zlabel('Z')
        def update(num, data_tracks, tracks):
            for track, data in zip(tracks, data_tracks):
                track.set_data(data[0:2, :num])
                track.set_3d_properties(data[2, :num])
            return tracks
        ani_steps = (tracks_test.shape)[0]
        ani_show = animation.FuncAnimation(fig, update, ani_steps, fargs = (data, tracks),
                                           interval = ani_interval, repeat = False, blit = False)
        ani_show.save(r'result_test\test_track.gif', writer = 'pillow')
        plt.show()
    
    def test_in(self, env):
        if env == 'env_test1':
            env_test = working_space_test1()
        elif env == 'env_test2':
            env_test = working_space_test2()
        elif env == 'env_test3':
            env_test = working_space_test3()
        else:
            print('Environment doesn\'t exist. Please check the environment name.')
            return
        print('\n-------------------[ start of %s ]------------------' % env)
        to_cut_test, to_keep_test = env_test.info()
        if visulization:
            self.visualize(env_test, 'Before processing')
        print('\nProcessing ......')
        accuracy_test = []
        fault_test = []
        average_accuracy_test = []
        average_fault_test = []
        time_test = []
        sum_accuracy_test = 0
        sum_fault_test = 0
        for test in range(test_num):
            s_test = env_test.reset()
            test_r = 0
            test_steps = 0
            a_memory = []
            t_test0 = time.time()
            while True:
                a_test = self.choose_action_test(s_test, a_memory)
#                a_test = self.choose_action_random()
                a_memory.append(a_test)
                s_test_, step_r_test, e_num_test, f_num_test, m_complete_test, m_incomplete_test, \
                out_test, end_test, tracks_test = env_test.step(a_test)
                test_r += step_r_test
                if end_test or test_steps == max_test_steps:
                    t_test_ = time.time()
                    delta_t_test = t_test_ - t_test0
                    time_test.append(delta_t_test)
                    a_rate_test = round((e_num_test / to_cut_test), 4)
                    f_rate_test = round((f_num_test / to_keep_test), 4)
                    sum_accuracy_test += a_rate_test
                    sum_fault_test += f_rate_test
                    accuracy_test.append(a_rate_test)
                    fault_test.append(f_rate_test)
                    average_accuracy_test.append(sum_accuracy_test / (test + 1))
                    average_fault_test.append(sum_fault_test / (test + 1))
                    if print_test_detail:
                        if (test + 1) * test_print % test_num == 0:
                            print('\n[Test%d] ' % test, 'Steps:', test_steps, 
                                  'Accuracy:', a_rate_test, 'Fault:', f_rate_test, 'Out:', out_test)
                            print(a_memory)
                        else:
                            print('[Test%d]' % test, end = '')
                    elif (test + 1) % 10 == 0:
                        print('[%d]' % (test + 1), end = '')
                    else:
                        print('.', end = '')
                    break
                s_test = s_test_
                test_steps += 1
        self.test_result(accuracy_test, fault_test, time_test, average_accuracy_test, average_fault_test)
        if visulization:
            self.visualize(env_test, 'After processing', average_accuracy_test[-1], average_fault_test[-1])
            self.animate_voxel(tracks_test, env_test)
            self.animate_track(tracks_test, env_test)
        print('-------------------[ end of %s ]------------------' % env)

################################## Execution ##################################

te = test()
te.test_in(env_number)