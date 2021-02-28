# -*- coding: utf-8 -*-
'''
This file defines a 3-axis-cnc working space, including material, product, cutter and cutter's action::
1. Size: 
   Environments: 40 * 40 * 20
   Matched .binvox model: 15 * 15 * 10
2. Movement: 
   step-by-step movement
3. Settable parameters:
   material_name  - Name of material file
   product_name   - Name of product file
   space_x        - Size of the working space in x-coordinate
   space_y        - Size of the working space in y-coordinate
   space_z        - Size of the working space in z-coordinate
   s_neutral      - Observation (voxel: no material)
   s_false        - Observation (voxel: material to keep)
   s_effective    - Observation (voxel: material to cut)
   s_out          - Observation (voxel: out of working space)
   r_neutral      - Reward (no cut)
   r_false        - Reward (wrong cut)
   r_effective    - Reward (effective cut)
   r_out          - Reward (run out of working space)
   accuracy       - The accuracy of cutting
   tolerance      - The fault tolerance of cutting
'''

import numpy as np
import binvox_rw
import gym

class working_space_test2(gym.Env):
    '''  
    Funcitons:
        -------------------------
        [ Initialize parameters ]
        -------------------------
            __init__(self)
        ---------------------
        [ Build Environment ]
        ---------------------
            get_model(self)
            expansion(self, obj_name)
            info(self)
            top_point(self, obj_name)
        ----------------- 
        [ Define Cutter ]
        -----------------
            cut_points(self, position)
            sensor(self, position)
            scanner(self, position)
        ------------------------------------- 
        [ Define Action(move, cut & reward) ]
        -------------------------------------
            move(self, position, action)
            cut_and_reward(self, position)
        ----------------- 
        [ Execute steps ]
        -----------------
            step(self, action)
            reset(self)
    '''
    
    ######################## Initialize parameters ############################
    
    def __init__(self,
                 material_name = 'test_material2_16_16_10.binvox',
                 product_name = 'test_product2_16_16_10.binvox',
                 space_x = 40, 
                 space_y = 40, 
                 space_z = 20, 
                 s_neutral = 0,
                 s_false = -1,
                 s_effective = 1,
                 s_out = -2,
                 s_repeat = -0.1,
                 s_reduce = -0.001,
                 r_neutral = 0,
                 r_false = -1,
                 r_effective = 1,
                 r_out = -20,
                 r_vain = -0.5,
                 r_repeat = -0.1,
                 r_reduce = -0.001,
                 accuracy = 1,
                 tolerance = 0.3):
        
#------------- From here all the same for each test environment --------------#
        
        '''
        Initialized variables (global):
            action_option       'dict'            Options of actions
            sight_option        'dict'            Options of sights
            reward_option       'dict'            Options of rewards
            -------------------------------------------------------------------------
            space               'tuple'           Size of environment
            space_diagonal      'float'           length of the diagonal of the working space
            mzero               'np.ndarray'      Position of machanical origin point
            cutter_sl           'int'             Side length of the cutter head (assume that cutter head is a square)
            cutter_h            'int'             Height of cutter
            -------------------------------------------------------------------------
            material            'np.ndarray'      3D-Array of material
            product             'np.ndarray'      3D-Array of product
            result_space        'np.ndarray'      Working space for processing
            result_space_p      'np.ndarray'      Product space for processing
            result_space_c      'np.ndarray'      Cutting space for processing
            sight_space         'np.ndarray'      Space of observation
            reward_space        'np.ndarray'      Space of reward
            -------------------------------------------------------------------------
            product_size        'int'             Number of points in product space
            cutting_size        'int'             Number of points in cutting space
            false_num           'int'             Number of false cut
            false_max           'int'             Max. number of false cut allowed
            effective_num       'int'             Number of effective cut
            effective_max       'int'             Max. number of effective cut
            effective_standard  'int'             Number of effective cut before stop
            false_num           'int'             Number of false cut
            false_max           'int'             Max. number of false cut
            false_standard      'int'             Number of allowed false cut
            -------------------------------------------------------------------------
            mzero               'np.ndarray       Position of machine origin point
            position            'np.ndarray'      Position of starting point
            start               'np.ndarray'      Position of starting point (same with position)
            state               'np.ndarray'      Start state of the cutter (including sensor and scanner)
            -------------------------------------------------------------------------
            state               'np.ndarray'      Starting state (observations)
            -------------------------------------------------------------------------
            moves               'list'            The record of movements (action num)
            tracks              'np.ndarray'      The record of tracks (positions)
            result_space_ani    'np.ndarray'      Working space for processing (animation)
            result_space_p_ani  'np.ndarray'      Product space for processing (animation)
            result_space_c_ani  'np.ndarray'      Cutting space for processing (animation)
        ''' 
        
        self.action_option = {'up':0,
                              'down':1,
                              'left':2,
                              'right':3,
                              'front':4,
                              'back':5}
        self.sight_option = {'neutral':s_neutral,
                             'false':s_false,
                             'effective':s_effective,
                             'out':s_out,
                             'repeat':s_repeat,
                             'reduce':s_reduce}
        self.reward_option = {'neutral':r_neutral,
                              'false':r_false,
                              'effective':r_effective,
                              'out':r_out,
                              'repeat':r_repeat,
                              'reduce':r_reduce}
        
        self.space_x = space_x
        self.space_y = space_y
        self.space_z = space_z
        self.space = (self.space_x, self.space_y, self.space_z)
        
        self.diagonal = np.sqrt(np.square(self.space_x) + np.square(self.space_y) + np.square(self.space_z))

        self.x_mzero = 0
        self.y_mzero = 0
        self.z_mzero = self.space_z
        self.mzero = np.array([self.x_mzero, self.y_mzero, self.z_mzero])

        self.cutter_sl = 1
        self.cutter_h = 6
        
        self.material, self.product = self.get_model(material_name, product_name)
        self.result_space = self.expansion(self.material)
        self.result_space_p = self.expansion(self.product)
        self.result_space_c = self.result_space_p != self.result_space
        
        self.sight_space = self.sight_option['reduce'] * ~ self.result_space + \
                           self.sight_option['effective'] * self.result_space_c + \
                           self.sight_option['false'] * self.result_space_p
                           
        self.reward_space = self.reward_option['reduce'] * ~ self.result_space + \
                            self.reward_option['effective'] * self.result_space_c + \
                            self.reward_option['false'] * self.result_space_p
        
        self.product_size = self.result_space_p[self.result_space_p == True].size
        self.cutting_size = self.result_space_c[self.result_space_c == True].size

        self.effective_num = 0
        self.effective_max = self.cutting_size
        self.effective_standard = accuracy * self.cutting_size

        self.false_num = 0
        self.false_max = self.product_size
        self.false_standard = tolerance * self.product_size
        
        self.x_pzero = 0
        self.y_pzero = 0
        self.z_pzero = self.top_point() + 1
        self.mzero = np.array([self.x_pzero, self.y_pzero, self.z_pzero])
        
        self.top_z = self.top_point()
        self.position_x = np.random.randint(0, self.space_x - 1)
        self.position_y = np.random.randint(0, self.space_y - 1)
        self.position_z = np.random.randint(self.top_z + 1, self.space_z - self.cutter_h)
        self.position = np.array([self.position_x, self.position_y, self.position_z])
        self.start = self.position
        
        s_down, s_up, s_left0, s_right0, s_back0, s_front0 = self.sensor(self.position)
        scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8 = self.scanner(self.position)
        self.state = np.array([s_down, s_up, s_left0, s_right0, s_back0, s_front0,
                               scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8])
    
        self.moves = []
        self.tracks = self.start[np.newaxis, :]
        self.result_space_ani = self.expansion(self.material)
        self.result_space_p_ani = self.expansion(self.product)
        self.result_space_c_ani = self.result_space_p_ani != self.result_space_ani
    
    ########################## Build Environment ##############################
    
    def get_model(self, material_name, product_name):
        '''
        Get 3d-Array form of obj. file
        '''
        with open('model_binvox/%s'% material_name, 'rb') as f:
            material_binvox = binvox_rw.read_as_3d_array(f)
        material = np.array([material_binvox.data])[0,:,:,:]
        with open('model_binvox/%s'% product_name, 'rb') as f:
            product_binvox = binvox_rw.read_as_3d_array(f)
        product = np.array([product_binvox.data])[0,:,:,:]
        return material, product

    def expansion(self, obj_name):
        '''
        Expansion of model to build the complete environment
        '''
        j, k, l = obj_name.shape
        if l <= self.space_z:
            obj_deep = np.concatenate((obj_name,
                                       np.zeros((j, k, self.space_z - l), dtype = bool)), axis = 2)
        else:
            obj_deep = obj_deep = obj_name[:, :, :self.space_z]
        obj_horizon = np.concatenate((np.zeros((j, round((self.space_y - k)/2), self.space_z), dtype = bool), 
                                      obj_deep, 
                                      np.zeros((j, round((self.space_y - k)/2), self.space_z), dtype = bool)), axis = 1)
        obj_space = np.concatenate((np.zeros((round((self.space_x - j)/2), self.space_y, self.space_z), dtype = bool), 
                                    obj_horizon, 
                                    np.zeros((round((self.space_x - j)/2), self.space_y, self.space_z), dtype = bool)))
        return obj_space
    
    def info(self):
        '''
        Check the size of material and product and their spaces
        '''
        print('\nInformation of env_training1')
        to_cut = self.cutting_size
        to_keep = self.product_size
        print('Shape of original material:', self.material.shape)
        print('Shape of material_space:', self.result_space.shape)
        print('Shape of original product:', self.product.shape)
        print('Shape of product_space:', self.result_space_p.shape)
        print('To cut points:', to_cut)
        print('To keep points:', to_keep)
        return to_cut, to_keep

    def top_point(self):
        '''
        Get the top point's z coordinate
        '''
        x_max, y_max, z_max = self.material.shape
        for x in range(x_max):
            for y in range(y_max):
                for fall in range(1, z_max + 1):
                    z = z_max - fall
                    if self.material[x, y, z]:
                        top = z
                        return top
                    else:
                        pass
    
    ############################## Define Cutter ##############################
    
    def cut_points(self, position):
        '''
        Get the cut points according to the position of the cutter
        '''
        cut = []
        cut_points = []
        for z in range(self.cutter_h):
            cut.append(np.array([0,0,z]))
        for cut_point in cut:
            cut_points.append(position + cut_point)
        return cut_points
    
    def sensor(self, position):
        '''
        Get the observations around the cutter
        '''
        x, y, z = position
        sight_down = np.array([x, y, z -1])
        sight_up = np.array([x, y, z + self.cutter_h])
        sight_left0 = np.array([x, y - 1, z])
        sight_right0 = np.array([x, y + 1, z])
        sight_front0 = np.array([x + 1, y, z])
        sight_back0 = np.array([x - 1, y, z])
        sight_positions = [sight_down, sight_up, sight_left0, sight_right0, sight_back0, sight_front0] 
        sights = []
        for sight_position in sight_positions:
            x_s, y_s, z_s = sight_position
            if x_s in range(self.space_x) and y_s in range(self.space_y) and z_s in range(self.space_z):
                s_value = self.reward_space[x_s][y_s][z_s]
            else:
                s_value = self.reward_option['out']
            sights.append(s_value)
        return(sights)
    
    def scanner(self, position):
        '''
        Get the observations of the distribution of to-cut voxels in the working space
        '''
        x, y, z = position
        zone1_c = self.result_space_c[0:x, 0:y, z:self.space_z]
        zone2_c = self.result_space_c[0:x, y:self.space_y, z:self.space_z]
        zone3_c = self.result_space_c[x:self.space_x, y:self.space_y, z:self.space_z]
        zone4_c = self.result_space_c[x:self.space_x, 0:y, z:self.space_z]
        zone5_c = self.result_space_c[0:x, 0:y, 0:z]
        zone6_c = self.result_space_c[0:x, y:self.space_y, 0:z]
        zone7_c = self.result_space_c[x:self.space_x, y:self.space_y, 0:z]
        zone8_c = self.result_space_c[x:self.space_x, 0:y, 0:z]
        result_size_c = self.result_space_c[self.result_space_c == True].size
        scan1 = round((zone1_c[zone1_c == True].size / (result_size_c + 1)), 1)
        scan2 = round((zone2_c[zone2_c == True].size / (result_size_c + 1)), 1)
        scan3 = round((zone3_c[zone3_c == True].size / (result_size_c + 1)), 1)
        scan4 = round((zone4_c[zone4_c == True].size / (result_size_c + 1)), 1)
        scan5 = round((zone5_c[zone5_c == True].size / (result_size_c + 1)), 1)
        scan6 = round((zone6_c[zone6_c == True].size / (result_size_c + 1)), 1)
        scan7 = round((zone7_c[zone7_c == True].size / (result_size_c + 1)), 1)
        scan8 = round((zone8_c[zone8_c == True].size / (result_size_c + 1)), 1)
        scans = [scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8]
        return scans
    
    ################## Define Action(move, cut & reward) ######################
    
    def move(self, position, action):
        '''
        Movement mechanism of the agent
        '''
        x, y, z = position
        if action == self.action_option['up']:
            self.moves.append('up')
            z += 1
        elif action == self.action_option['down']:
            self.moves.append('down')
            z += -1
        elif action == self.action_option['left']:
            self.moves.append('left')
            y += -1
        elif action == self.action_option['right']:
            self.moves.append('right')
            y += 1
        elif action == self.action_option['back']:
            self.moves.append('back')
            x += -1
        elif action == self.action_option['front']:
            self.moves.append('front')
            x += 1
        else:
            pass
        position_ = np.array([x, y, z])
        self.tracks = np.concatenate((self.tracks, position_[np.newaxis, :]))
        return position_                
    
    def cut_and_reward(self, position):
        '''
        Cut and reward mechanism of the agent
        '''
        step_reward = 0
        out = False
        cut_points = self.cut_points(position)
        for cut_point in cut_points:
            x_c, y_c, z_c = cut_point
            if x_c in range(self.space_x) and y_c in range(self.space_y) and z_c in range(self.space_z):
                step_reward += self.reward_space[x_c][y_c][z_c]
                if self.result_space_c[x_c][y_c][z_c]:
                    self.result_space_c[x_c][y_c][z_c] = False
                    self.effective_num += 1
                    self.sight_space[x_c][y_c][z_c] = 0
                    self.reward_space[x_c][y_c][z_c] = 0
                elif self.result_space_p[x_c][y_c][z_c]:
                    self.result_space_p[x_c][y_c][z_c] = False
                    self.false_num += 1
                    self.sight_space[x_c][y_c][z_c] = 0
                    self.reward_space[x_c][y_c][z_c] = 0
                else:
                    self.sight_space[x_c][y_c][z_c] += self.sight_option['repeat']
                    self.reward_space[x_c][y_c][z_c] += self.reward_option['repeat']
            else:
                self.moves.append('This step runs out of the working space. \nProcess ends.')
                step_reward += self.reward_option['out']
                out = True
                break
        return step_reward, out
    
    ############################# Execute steps ###############################
    
    def step(self, action):
        '''
        action: [move]
        state: [s_down, s_left, s_right, s_back, s_front, 
                scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8]
        '''
        position = self.position
        
        out = False
        end = False
        
        self.position = self.move(position, action)
        step_reward, out = self.cut_and_reward(self.position)
        
        s_down_, s_up_, s_left0_, s_right0_, s_back0_, s_front0_ = self.sensor(self.position)
        scan1_, scan2_, scan3_, scan4_, scan5_, scan6_, scan7_, scan8_ = self.scanner(self.position)
        self.state = np.array([s_down_, s_up_, s_left0_, s_right0_, s_back0_, s_front0_,
                               scan1_, scan2_, scan3_, scan4_, scan5_, scan6_, scan7_, scan8_])     
        
        mission_complete = self.effective_num >= self.effective_standard
        mission_incomplete = self.false_num >= self.false_standard
        end = out or mission_complete or mission_incomplete
        
        return self.state, step_reward, self.effective_num, self.false_num, \
               mission_complete, mission_incomplete, out, end, self.tracks
            
    def reset(self):
        '''
        Reset variables:
            result_space_p     'np.ndarray'
            result_space_c     'np.ndarray'
            result_space       'np.ndarray'
            position           'np.ndarray'
            start              'np.ndarray'
            state              'np.ndarray'
            false_num          'int'
            effective_num      'int'
            moves              'list'
            tracks             'np.ndarray'
            result_space_ani   'np.ndarray'
            result_space_p_ani 'np.ndarray'
            result_space_c_ani 'np.ndarray'
        '''
        self.result_space = self.expansion(self.material)
        self.result_space_p = self.expansion(self.product)
        self.result_space_c = self.result_space_p != self.result_space
        
        self.reward_space = self.reward_option['reduce'] * ~ self.result_space + \
                            self.reward_option['effective'] * self.result_space_c + \
                            self.reward_option['false'] * self.result_space_p
        
        self.position_x = np.random.randint(0, self.space_x - 1)
        self.position_y = np.random.randint(0, self.space_y - 1)
        self.position_z = np.random.randint(self.top_z + 1, self.space_z - self.cutter_h)
        self.position = np.array([self.position_x, self.position_y, self.position_z])
        self.start = self.position
        
        self.action_last = -10 # no such action with spcific meaning, just avoid ambiguity
        self.cut_last = False
        
        s_down, s_up, s_left0, s_right0, s_back0, s_front0 = self.sensor(self.position)
        scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8 = self.scanner(self.position)
        self.state = np.array([s_down, s_up, s_left0, s_right0, s_back0, s_front0,
                               scan1, scan2, scan3, scan4, scan5, scan6, scan7, scan8])
        
        self.false_num = 0
        self.effective_num = 0
        
        self.moves = []
        self.tracks = self.start[np.newaxis, :]
        self.result_space_ani = self.expansion(self.material)
        self.result_space_p_ani = self.expansion(self.product)
        self.result_space_c_ani = self.result_space_p_ani != self.result_space_ani
        
        return self.state