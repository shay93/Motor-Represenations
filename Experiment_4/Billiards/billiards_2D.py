import numpy as np
from rllab.envs.base import Env
from rllab.spaces.box import Box
import os
import sys
from scipy.misc import imresize
import matplotlib.pyplot as plt
from planarspace import PlanarSpace

parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
import training_tools as tt
import IPython

class Billiards_2D(Env):
    """
    2D Billiards environment
    """


    def __init__(self,num_steps = 100,epsilon = 15, box_width = 5,step_size = 3):
        """
        theta_i of shape [2] np array
        """
        #specify the box width for the actor and the target
        self.box_width = box_width
        #initialize the centre of the target box on the 128 by 128 grid by sampling from a uniform distribution
        self.target = [tuple(np.round(np.random.uniform(10,54,size = 2)))]
        #initialize a helper function to draw the boxes on the grid
        self.sp = tt.shape_maker()
        #initialize the actors location as well by sampling 
        self.actor = self.get_actor_loc()
        #using the actor location and the target location render an observation image
        self.cur_obs_image = self.render_image()
        #if centre of the actor box is within epsilon of the target give a reward
        self.epsilon = epsilon
        #indicate the horizon
        self.num_steps = num_steps
        #indicate the number of pixels to jump over when a single action is taken
        self.step_size = step_size
        #specify the action and observation space
        self._action_space = Box(0,1,(4))
        self._observation_space = PlanarSpace()

    def get_actor_loc(self):
        """
        Sample from a uniform distribution to get actors location in grid
        Call on reset and initialization return a list with a single tuple as an element
        """
        #sample to get a tentative, initial actor location
        actor_loc = [tuple(np.round(np.random.uniform(10,54,size = 2)))]
        #check if the initial actor location overlaps with the target location if so get
        #get the actor location again
        if self.check_overlap(actor = actor_loc): #there is overlap
            return self.get_actor_loc()
        else:
            return actor_loc

    def check_overlap(self, actor):
        """
        Check if the actor and target boxes overlap
        Return a boolean indicating overlap or not
        """
        if abs(self.target[0][0] - actor[0][0]) < self.box_width or abs(self.target[0][1] - actor[0][1]) < self.box_width:
            return True
        else:
            return False

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment 1d array of size 4 [left,right,up,down]
        -------
        (observation, reward, done, info)
        reward [Float] : Add a reward based on inverse of distance
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        #take the max of the action to determine should move left,right,up,down
        
        amax = np.argmax(action)
        #update the target location center based on the output
        if amax == 0: #go left, decrement column by step
            print(action)
            self.actor = [(self.actor[0][0],self.actor[0][1] - self.step_size)]
        elif amax == 1: # go right, increment column by step
            self.actor = [(self.actor[0][0],self.actor[0][1] + self.step_size)]
        elif amax == 2: # go up, decrement row by step
            self.actor = [(self.actor[0][0] - self.step_size,self.actor[0][1])]
        elif amax == 3: # go down, increment row by step
            self.actor = [(self.actor[0][0] + self.step_size,self.actor[0][1])]
        
        #get the current observation image after the step has been taken
        self.cur_obs_image = self.render_image()

        #give a reward if the actor is within
        if abs(self.actor[0][0] - self.target[0][0]) < self.epsilon and abs(self.actor[0][1] - self.target[0][1]) < self.epsilon:
            reward = 1.
        else:
            reward = 0.

        #terminate the episode if the target is reached
        done = self.check_overlap(self.actor)
        
        return self.cur_obs_image,reward,done,{"Actor" : self.actor}


    def render_image(self):
        #initialize a grid to draw the environment
        temp_grid = tt.grid(grid_size = (64,64))
        #draw the actor box in white
        temp_grid.draw_figure(self.sp.get_points_to_increase_line_thickness(self.actor,width = self.box_width),pixel_value = 1)
        #draw the target box in grey
        cur_image = temp_grid.draw_figure(self.sp.get_points_to_increase_line_thickness(self.target,width = self.box_width),pixel_value = 0.5)
        #renormalize observation image
        #cur_image = cur_image / 255.
        return cur_image.flatten() 

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
        #Reset the target location by sampling 
        self.target = [tuple(np.round(np.random.uniform(10,54,size = 2)))]
        #Reset the actor location
        self.actor = self.get_actor_loc()
        return self.render_image()

    @property
    def observation_space(self):
        return self._observation_space