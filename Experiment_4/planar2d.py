import numpy as np
from random import randint
from rllab.envs.base import Env
#from rllab.misc import special2 as special
#from rllab.spaces.discrete import Discrete
from planarspace import PlanarSpace
from rllab.spaces.box import Box
import os
import sys
from scipy.misc import imresize

parent_dir = os.path.dirname(os.getcwd())
experiment_3_dir  = parent_dir + "/" + "Experiment_3"
sys.path.append(parent_dir)

import training_tools as tt
import IPython

class env_2DOF_arm(Env):
    """
    Environment for a planar arm which takes actions via a delta theta
    and where its observations are a visual input of the planar surface
    The arm is reward whenever it reaches within an epsilon of the taret surface
    """


    def __init__(self,num_steps = 100,epsilon = 10,theta_i = np.array([0.,np.pi/2]),link_length = 50):
        """
        theta_i of shape [2] np array
        """
        self.prev_theta = theta_i
        self.link_length = link_length
        self.target = [tuple(np.random.normal([95,95],[5,5]))]
        self.sp = tt.shape_maker()
        self.init_theta = theta_i
        self.cur_theta = theta_i
        self.cur_image = self.render_image()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self._action_space = Box(np.array([-np.pi/2,-np.pi/2]),np.array([np.pi/2,np.pi/2]))
        self._observation_space = PlanarSpace()


    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : visualization of the 2DOF arm over the planar surface
        reward [Float] : Add a reward when the arm end effector is within the target
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self.cur_theta = self.prev_theta + action[0]
        self.cur_image = self.render_image()
        self.prev_theta = self.cur_theta
        #in addition compute the reward from the previous action
        #compare the end effector position to the target position and determine whether it is within epsilon of the target
        if abs(self.end_effector[0] - self.target[0][0]) < self.epsilon and abs(self.end_effector[1] - self.target[0][1]) < self.epsilon:
            reward = 1/(((self.end_effector[0] - self.target[0][0]) ** 2 + (self.end_effector[1] - self.target[0][1]) ** 2) ** 0.5)
        else:
            reward = 0.

        if abs(self.end_effector[0] - self.target[0][0]) < 3 and abs(self.end_effector[1] - self.target[0][1]) < 3:
            done = True
        else:
            done = False
        
        #let info record the joint angle state at the current timestep
        info = {"Joint State" : self.cur_theta}

        return self.cur_image,reward,done,info

    def render_image(self):
        theta_1 = self.cur_theta[0]
        theta_2 = self.cur_theta[1]
        #specify the anchor point
        start_x = 63
        start_y = 63
        start_point = (start_x,start_y)
        #get the link end positions
        x_link_1 = np.round(np.cos(theta_1)*self.link_length + start_x)
        y_link_1 = np.round(np.sin(theta_1)*self.link_length + start_y)
        x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*self.link_length)
        y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*self.link_length)
        link1_end_point = (int(x_link_1),int(y_link_1))
        link2_end_point = (int(x_link_2),int(y_link_2))
        self.end_effector = link2_end_point
        #print(np.shape(self.sp.draw_line(start_point,link1_end_point,0.1,0.1)))
        #print(np.shape(self.target))
        pos_list = self.sp.draw_line(start_point,link1_end_point,0.1,0.1) + self.sp.draw_line(link1_end_point,link2_end_point,0.1,0.1) + self.target
        #now get the extended point list in order to thicken the lines
        additional_points = self.sp.get_points_to_increase_line_thickness(pos_list)
        #now initialize a grid in order to save the correct images
        temp_grid = tt.grid(grid_size = (128,128))
        #draw the the points
        temp_grid.draw_figure(pos_list)
        #thicken the lines
        cur_image = temp_grid.draw_figure(additional_points)
        resize_im = imresize(cur_image,[64,64])
        return np.ceil(resize_im/255.0).flatten() #Making it binary

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        #this would require setting the observation to the 
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        print("env reset")
        #Reset the theta to the initial theta
        self.cur_theta = self.init_theta
        self.prev_theta = self.init_theta
        #render the grid using this theta and
        #also reset the theta
        self.target = [tuple(np.random.normal([95,95],[5,5]))]
        return self.render_image()


    # def _get_next_observation(self):
    #     return self._observation_space.flatten(self._next_obs)

    @property
    def observation_space(self):
        return self._observation_space
