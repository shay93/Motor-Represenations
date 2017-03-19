import numpy as np
from random import randint
from rllab.envs.base import Env
from planarspace import PlanarSpace
from rllab.spaces.box import Box
import os
import sys
from scipy.misc import imresize
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.getcwd())

#import training_tools as tt
import IPython

class Planar_arm_2DOF_lowdim(Env):
    """
    Environment for a planar arm which takes actions via a delta theta
    and where its observations are 
    """

    def __init__(self,num_steps = 100,
                epsilon = 18,
                theta_i = np.random.uniform(0.,2.,size = 2),
                link_length = 40):
        """
        Initialize the environment by specifying the starting state and the location
        of the target
        """
        self.link_length = link_length
        #randomly specify the location of the target over the 128 by 128 grid
        self.target = np.round(np.random.uniform(74,118,size=2))
        #initialize an object that will help you draw lines
        #self.sp = tt.shape_maker()
        #set both the current and previous theta to the given initial theta
        self.prev_theta = theta_i
        self.cur_theta = theta_i
        #based on the state of the arm render and target location render an image
        #self.cur_image = self.render_image()
        #specify an epsilong wrt to the target that determines when the target has
        #been reached
        self.epsilon = epsilon
        #num of steps determines the time horizon for the task
        self.num_steps = num_steps
        #the action space is the change in the delta in the joint angles
        #restrict the action space to small angles in order to ensure smooth
        #trajectories
        self._action_space = Box(-1.,1.,(2))
        #The observation is a concatenation of the joint states and target loc
        self._observation_space = Box(-1.,1.,(4))

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs : An array of observations consisting of target loc and actor state
        -------
        (observation, reward, done, info)
        observation : visualization of the 2DOF arm over the planar surface
        reward [Float] : Add a reward when the arm end effector is within the target
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        #add the action to the previous joint state to obtain the new state
        self.cur_theta = np.mod(np.sum([self.prev_theta,action[0]],axis = 0),2.)
        #Make the current state equal to the previous state
        self.prev_theta = self.cur_theta
        #using the current state find the end effector position 
        end_effector = self.get_end_effector_pos()
        #compute the L2 distance between the end effector location and target location
        distance = np.linalg.norm(np.subtract(end_effector,self.target))
        #compute the normalized distance by dividing by the length of the box
        distance_normalized = distance / 90.
        #compute the reward as hte inverse of the normalized distance and scale between 0 and 1
        reward = (1./(distance_normalized + 1.)-0.5)*2.
        #let info record the joint angle state at the current timestep
        info = {"Joint State" : self.cur_theta}
        #do not specify completion
        done = False
        #scale the target location to get an observation
        scaled_target_obs = (self.target - 95.)/22.
        #stack the target location and the joint angle state to obtain the obs
        obs = np.concatenate([self.shift_theta_range(self.cur_theta),scaled_target_obs])
    
        return np.copy(obs),np.copy(reward),done,info
    
    def get_end_effector_pos(self):
        """
        Use the link length and the current joint angle state
        to get the current end effector position
        """
        theta_1 = self.cur_theta[0]*np.pi
        theta_2 = self.cur_theta[1]*np.pi
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
        #IPython.embed()
        return np.copy(link2_end_point)
    
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        #Reset the joint angle state randomly
        self.cur_theta = np.random.uniform(0.,2.,size = 2)
        #set the previous and current state to be equal to one another
        self.prev_theta = self.cur_theta
        #reset the location of the target by sampling over a 128 by 128 grid
        self.target = np.round(np.random.uniform(73,117,size=2))
        scaled_target_obs = (self.target - 95.)/22.
        return np.copy(np.concatenate([self.shift_theta_range(self.cur_theta),scaled_target_obs]))

    def shift_theta_range(self,angle_array):
       """
       Shifts range of angle given between to 0 to 2pi to -pi and pi
       """
       array_length = np.shape(angle_array)[0]
       shifted_angles = np.zeros(array_length)
       for i,angle in enumerate(angle_array):
          if angle > 1.:
            shifted_angles[i] = angle - 2.
          else:
            shifted_angles[i] = angle
       return shifted_angles

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    @property
    def observation_space(self):
        return self._observation_space
