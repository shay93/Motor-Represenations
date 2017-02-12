
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

parent_dir = os.path.dirname(os.getcwd())
experiment_3_dir  = parent_dir + "/" + "Experiment_3"
sys.path.append(parent_dir)

import training_tools as tt

class rl_env_2DOF:
	
	def __init__(self,theta_i = np.array([0.,np.pi/2]),link_length = 50,target_loc = [(np.random.randint(67,125),np.random.randint(67,125))]):
		"""
		theta_i of shape [2] np array
		"""
		self.prev_theta = theta_i
		self.link_length = link_length
		self.target = target_loc
		self.sp = tt.shape_maker()
		self.cur_theta = theta_i
		self.cur_image = self.render_image()

	def step(self,delta_theta):
		"""
		delta_theta of shape [2] np.array
		"""
		self.cur_theta = self.prev_theta + delta_theta
		self.cur_image = self.render_image()
		self.prev_theta = self.cur_theta
		return self.cur_image

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
		pos_list = self.sp.draw_line(start_point,link1_end_point,0.1,0.1) + self.sp.draw_line(link1_end_point,link2_end_point,0.1,0.1) + self.target
		#now get the extended point list in order to thicken the lines
		additional_points = self.sp.get_points_to_increase_line_thickness(pos_list)
		#now initialize a grid in order to save the correct images
		temp_grid = tt.grid(grid_size = (128,128))
		#draw the the points
		temp_grid.draw_figure(pos_list)
		#thicken the lines
		cur_image = temp_grid.draw_figure(additional_points)
		return cur_image


	def get_world_state(self):
		return self.cur_theta,self.cur_image


	
###test

# rl_env = rl_env_2DOF(np.array([0.,np.pi/2]))


# start = time.time()
# for i in range(400):
# 	grid = rl_env.step(np.array([0,np.pi/600]))
# end = time.time()
# print(end - start)
# plt.imshow(grid, cmap = "Greys_r")
# plt.show()