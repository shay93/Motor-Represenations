import numpy as np
from random import randint
from rllab.envs.base import Env
#from planarspace import PlanarSpace
from rllab.spaces.box import Box
import os
import sys
from scipy.misc import imresize
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
import training_tools as tt
import IPython

class Planar_arm_3DOF_lowdim(Env):
    """
    Environment for a planar arm which takes actions via a delta theta
    and where its observations are 
    """

    def __init__(self,num_steps = 1000,
                theta_i = np.array([0,np.pi,0]),#np.random.uniform(0.,2.,size = 3),
                link_length = 30):
        """
        Initialize the environment by specifying the starting state and the location
        of the target
        """
        self.link_length = link_length
        #randomly specify the location of the target over the 128 by 128 grid
        self.target = np.round(np.random.uniform(10,118,size=2))
        #initialize an object that will help you draw lines
        self.sp = tt.shape_maker()
        #set both the current theta to the given initial theta
        self.cur_theta = theta_i
        self.num_steps = num_steps
        #the action space is the change in the delta in the joint angles
        #restrict the action space to small angles in order to ensure smooth
        #trajectories
        self._action_space = Box(-1,1,(3))
        #The observation is a concatenation of the joint states and target loc
        self._observation_space = Box(-1.,1.,(5))

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
        #using the current state find the end effector position 
        end_effector = self.get_end_effector_pos()
        #let info record the image at the current timestep
        info = {"Observed Image" : self.render_image(),"Overlap" : self.check_overlap(end_effector)} 
        #compute the L2 distance between the end effector location and target location
        distance = np.linalg.norm(np.subtract(end_effector,self.target))
        #compute the normalized distance by dividing by the length of the box
        distance_normalized = distance / 90.
        #compute the reward as hte inverse of the normalized distance and scale between 0 and 1
        reward = (1./(distance_normalized + 1.)-0.5)*2.
        if self.check_overlap(end_effector):
           reward += 0.5
        #do not specify completion
        done = False
        #scale the target location to get an observation
        scaled_target_obs = (self.target - 95.)/22.
        #add the action to the previous joint state to obtain the new state
        self.cur_theta = np.mod(np.sum([self.cur_theta,action[0]/20],axis = 0),2.) 
        #stack the target location and the joint angle state to obtain the obs
        next_obs = np.concatenate([self.shift_theta_range(self.cur_theta),scaled_target_obs])

        return np.copy(next_obs).astype('float64'),np.copy(reward),done,info

    def get_end_effector_pos(self):
        """
        Use the link length and the current joint angle state
        to get the current end effector position
        """
        theta_1 = self.cur_theta[0]*np.pi
        theta_2 = self.cur_theta[1]*np.pi
        theta_3 = self.cur_theta[2]*np.pi
        #specify the anchor point
        start_x = 63
        start_y = 63
        start_point = (start_x,start_y)
        #get the link end positions
        x_link_1 = np.round(np.cos(theta_1)*self.link_length + start_x)
        y_link_1 = np.round(np.sin(theta_1)*self.link_length + start_y)
        x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*self.link_length)
        y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*self.link_length)
        x_link_3 = np.round(x_link_2 + np.cos(theta_1 + theta_2 + theta_3)*self.link_length)
        y_link_3 = np.round(y_link_2 + np.sin(theta_1 + theta_2 + theta_3)*self.link_length)
        link1_end_point = (int(x_link_1),int(y_link_1))
        link2_end_point = (int(x_link_2),int(y_link_2))
        link3_end_point = (int(x_link_3),int(y_link_3))
        #IPython.embed()
        return np.copy(link3_end_point)
    
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        
        #Reset the joint angle state randomly
        #self.cur_theta = np.random.uniform(0.,2.,size = 3)
        self.cur_theta = np.array([0,np.pi,0])
        #get the end effector position 
        end_effector = self.get_end_effector_pos()
        #let info record the image at the current timestep
        info = {"Observed Image" : self.render_image()}
        #reset the location of the target by sampling over a 128 by 128 grid
        self.target = np.round(np.random.uniform(20,108,size=2))
        scaled_target_obs = (self.target - 95.)/22.
        return np.copy(np.concatenate([self.shift_theta_range(self.cur_theta),scaled_target_obs])).astype("float64")

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
       
    def render_image(self):
        theta_1 = self.cur_theta[0]*np.pi
        theta_2 = self.cur_theta[1]*np.pi
        theta_3 = self.cur_theta[2]*np.pi
        #specify the anchor point
        start_x = 63
        start_y = 63
        start_point = (start_x,start_y)
        #get positions of the end of each link by applying forward kinematics
        x_link_1 = np.round(np.cos(theta_1)*self.link_length + start_x)
        y_link_1 = np.round(np.sin(theta_1)*self.link_length + start_y)
        x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*self.link_length)
        y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*self.link_length)
        x_link_3 = np.round(x_link_2 + np.cos(theta_1 + theta_2 + theta_3)*self.link_length)
        y_link_3 = np.round(y_link_2 + np.sin(theta_1 + theta_2 + theta_3)*self.link_length)
        #concatentate the x and y positions to get the tuple position
        link1_end_point = (int(x_link_1),int(y_link_1))
        link2_end_point = (int(x_link_2),int(y_link_2))
        link3_end_point = (int(x_link_3),int(y_link_3))
        #import IPython; IPython.embed()
        #get a list of positions that draw the intermediary points between the end effector positions
        pos_list = self.sp.draw_line(start_point,link1_end_point) + self.sp.draw_line(link1_end_point,link2_end_point) \
         + self.sp.draw_line(link2_end_point,link3_end_point)
        #now get the extended point list in order to thicken the lines
        arm_points = self.sp.get_points_to_increase_line_thickness(pos_list,width = 7)
        #now initialize a grid in order to save the correct images
        temp_grid = tt.grid(grid_size = (128,128))
        #draw the the points
        temp_grid.draw_figure(arm_points)
        #thicken the lines
        cur_image = temp_grid.draw_figure(self.sp.get_points_to_increase_line_thickness([self.target],width = 9),pixel_value = 125)
        resize_im = imresize(cur_image,[64,64])
        return (resize_im).astype('uint8').flatten()
    
    def check_overlap(self,end_effector):
        """
        Check if there is an overlap between the end effector position and the target box
        """
        if abs(end_effector[0] - self.target[0]) < 2 and abs(end_effector[1] - self.target[1]) < 2:
           return True
        else:
           return False

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    @property
    def observation_space(self):
        return self._observation_space
