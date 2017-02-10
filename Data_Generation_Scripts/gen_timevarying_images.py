from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import training_tools as tt
import shutil

link_length_2dof = 50

#save a set of time varying images to folders so that they may be loaded when appropriate
#this requires a good naming convention so in addition to specifying the shape number you must also specify the image number

def get_points_to_increase_line_thickness(pos_list):
	#initialize a list to store the information needed
	more_pts = [0] * (len(pos_list))
	for i,pos in enumerate(pos_list):
		x,y = pos
		#use these values of x and y to build a temp array with points needed to thicken the original line
		temp_list = [(x+1,y),(x-1,y),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1),(x,y+1),(x,y-1)]
		more_pts[i] = temp_list
	return more_pts

def remove_points_out_of_range(pos_list):
	return [pos for pos in pos_list if pos[0] < 64 and pos[1] < 64]


#Define some globals 
#shape_str_array = ["Triangle","Square","Rectangle"]
shape_str_array = ["Rhombus","Hexagon"]
parent_dir = os.path.dirname(os.getcwd())
shape_jointseq_root_directory = parent_dir + "/" + "Eval_Shape_JointSeqs/"
shape_root_directory = parent_dir + "/New_Shapes/"
samples_per_shape = 100
#create this directory if it doesnt exist
if os.path.exists(shape_root_directory):
	shutil.rmtree(shape_root_directory)
	os.makedirs(shape_root_directory)
else:
	os.makedirs(shape_root_directory)

#do the same for the shape joint sequence director
if not(os.path.exists(shape_jointseq_root_directory)):
	os.makedirs(shape_jointseq_root_directory)

#now use training tools to generate the images you want 
sp = tt.shape_maker(10,32)
arm_2dof = tt.two_link_arm(link_length_2dof)

for shape_name in shape_str_array:
	#initialize a joint seq list for a 2dof arm that may be used to store the data
	joint_seq_list = []
	for shape_number in range(samples_per_shape):
		#use this to get the points for a shape
		pos_list,_ = sp.get_points(shape_name)
		#now that you have the points you can initialize the grid that will 
		#now use the pos_list to generate the joint sequence for the 2DOF arm
		joint_seq_list.append(arm_2dof.inverse_kinematics(pos_list))
		additional_points = get_points_to_increase_line_thickness(pos_list)
		#pt_index tells you the index of the end effector position in the end effector sequence
		for pt_index in range(len(pos_list)):
			pos_list_truncated = pos_list[:pt_index+1]
			additional_points_truncated = additional_points[:pt_index+1]
			#flatten additional points 
			additional_points_flattened = [pos for sublist in additional_points_truncated for pos in sublist if pos[0] < 64 and pos[1] < 64]
			#create a directory for the shape name and number in order to save the images that are generated
			shape_dir = shape_root_directory + shape_name + str(shape_number) + "/"
			#create this directory if it doesnt exist
			if not(os.path.exists(shape_dir)):
				os.makedirs(shape_dir)
			#one can now initialize a grid to draw the image corresponding to these end effector positions 
			temp_grid = tt.grid(shape_name + str(shape_number) + '_' + str(pt_index),shape_dir)
			#draw the points using the grid object
			temp_grid.draw_figure(pos_list_truncated)
			temp_grid.draw_figure(additional_points_flattened)
			temp_grid.save_image()
		if (round((shape_number / samples_per_shape)*100) % 10) == 0 : print "Percentage Completion for " + shape_name + " " + str(round((shape_number / samples_per_shape)*100)) 

	#save this list at the end of the sequence 
	with open(shape_jointseq_root_directory + '/' + shape_name + ".npy","wb") as f:
		pickle.dump(joint_seq_list,f)





#so for every index of the pos_list I know which index in more more_pts corresponds to the set of points needed to thicken the point in question

