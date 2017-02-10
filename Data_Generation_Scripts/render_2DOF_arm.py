from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

import training_tools as tt

#define the joint sequence directory
joint_seq_directory = parent_dir + "/" + "Eval_Shape_JointSeqs"
root_dir = parent_dir + "/" + "Eval_Planar_Arm_Rendering"

if not(os.path.exists(root_dir)):
	os.makedirs(root_dir)

#define the link length
link_length = 50
#define the grid size for the image
grid_size = (64,128)

#define a ist of shapes for which to load the joint sequence
shape_str_array = ["Rhombus","Hexagon"]

#initialize a shape maker object in order to draw lines
sp = tt.shape_maker()

#now load the npy array corresponding to each of these shapes
for shape_str in shape_str_array:
	file_path = joint_seq_directory + "/" + shape_str + ".npy"
	#now load the npy array
	with open(file_path,"rb") as f:
		joint_seq_list = pickle.load(f)
	#now loop through the joint seq list
	for j,joint_seq in enumerate(joint_seq_list[:10]):
		#specify the directory in which to store the images for this sequence
		seq_dir = root_dir + "/" + shape_str + str(j)
		#create this directory if it does not exist
		if not(os.path.exists(seq_dir)):
			os.makedirs(seq_dir)
		#get the seq length for the joint angle
		seq_length = np.shape(joint_seq)[1]
		#for each joint seq index out theta 1 and theta 2
		theta_1 = joint_seq[0,:]
		theta_2 = joint_seq[1,:]
		#now get the position in the 128 size grid that corresponds to the end effector position of each link
		start_x = 0
		start_y = 64
		x_link_1 = np.round(np.cos(theta_1)*link_length + start_x)
		y_link_1 = np.round(np.sin(theta_1)*link_length + start_y)
		x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*link_length)
		y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*link_length)
		#now loop through the link positions to generate an image at each timestep
		for i in xrange(seq_length):
			#get the end position of each arm and use that to construct a tuple to draw a line between each of the link end positions
			start_point = (start_x,start_y)
			link1_end_point = (int(x_link_1[i]),int(y_link_1[i]))
			link2_end_point = (int(x_link_2[i]),int(y_link_2[i]))
			pos_list = sp.draw_line(start_point,link1_end_point,0.1,0.1) + sp.draw_line(link1_end_point,link2_end_point,0.1,0.1)
			#now get the extended point list in order to thicken the lines
			additional_points = sp.get_points_to_increase_line_thickness(pos_list)
			#now initialize a grid in order to save the correct images
			temp_grid = tt.grid("timestep" + str(i), seq_dir + "/",grid_size)
			#draw the the points
			temp_grid.draw_figure(pos_list)
			#thicken the lines
			temp_grid.draw_figure(additional_points)
			#now save the image and hopefully we are done
			temp_grid.save_image()
		if (round((j / len(joint_seq_list))*100) % 5) == 0 : print "Percentage Completion for " + shape_str + " " + str(round((j / len(joint_seq_list))*100)) 
