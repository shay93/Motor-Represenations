from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
import pickle
import os
import sys

parent_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if not parent_dir_path in sys.path:
	sys.path.insert(1,parent_dir_path)


import training_tools as tt

DOF = 3 
ROOT_DIR = parent_dir_path + "Experiment_3/" + "Joints_to_Image/"
link_length = 30

def generate_input_image_legacy():
	"""
	Generate the input image for the nn
	"""
	#initialize the input image
	input_image = np.zeros((64,64), dtype = float)
	#now generate a random number to determine the number of points that are illuminated in the input image
	num_points = np.random.randint(5)
	#now generate random joint states equal to the number of points
	random_states = [gen_rand_joint_state(DOF) for j in range(num_points)]
	#initialize a pos list
	pos_list = []
	for joint_state in random_states:
		pos_list += [forward_kinematics(joint_state)]

	additional_points = get_points_to_increase_line_thickness(pos_list)
	additional_points_flattened = [pos for sublist in additional_points for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	for pos in pos_list:
		input_image[pos[0],pos[1]] = 1.0

	for pos in additional_points_flattened:
		input_image[pos[0],pos[1]] = 1.0

	return input_image
		 

def gen_target_image_legacy(pos, input_image):
	"""
	Takes an input pos and generates a target image
	"""
	x = pos[0]
	y = pos[1]
	temp_list = [(x+1,y),(x-1,y),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1),(x,y+1),(x,y-1)]
	#use the position to fill in the specified index with 1
	pos_list = temp_list + [pos]
	return joint_state_array,target_image_array,input_image_array
	for pos in pos_list:
		if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0:
			input_image[pos[0],pos[1]] = 1.0
	return input_image


def generate_training_data_legacy(num,dof):
	joint_state_array = np.zeros((num,dof), dtype = float)
	target_image_array = np.ndarray(shape = (num,64,64))
	input_image_array = np.ndarray(shape = (num,64,64))
	#now loop through the num of examples and populates these arrays
	for i in range(num):
		joint_state_array[i,:] = gen_rand_joint_state(DOF)
		#get end effector position
		pos = forward_kinematics(np.expand_dims(joint_state_array[i,:], axis = 0))
		#get a randomly generated input image
		input_image = generate_input_image_legacy()
		input_image_array[i,...] = input_image
		#use the pos to get the target image and tack on to target_image_array
		target_image_array[i,:] = gen_target_image_legacy(pos,input_image)
	return joint_state_array,target_image_array,input_image


#######################################LINE TRAINING DATA############################################################


def get_points_to_increase_line_thickness(pos_list):
	#initialize a list to store the information needed
	more_pts = [0] * (len(pos_list))
	for i,pos in enumerate(pos_list):
		x,y = pos
		#use these values of x and y to build a temp array with points needed to thicken the original line
		temp_list = [(x+1,y),(x-1,y),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1),(x,y+1),(x,y-1)]
		more_pts[i] = temp_list
	return more_pts

def gen_random_start_point_and_end_point():
	rand_joint_state = gen_rand_joint_state(DOF)
	#use this to get the random end_point
	end_point = forward_kinematics(rand_joint_state)
	start_point = (np.random.randint(63),np.random.randint(63))
	if start_point == end_point or abs(start_point[0] - end_point[0]) < 2 or abs(start_point[1] - end_point[1]) < 2:
		return gen_random_start_point_and_end_point()
	else:
		return start_point,end_point,rand_joint_state

def gen_rand_joint_state(num_dof):
	"""
	Generates random joint state for a specified dof returns a row with each column corresponding to a theta
	"""
	joint_state =  (np.random.rand(1,num_dof) - 0.5)*(2*np.pi)
	pos = forward_kinematics(joint_state)
	if pos[0] > 63 or pos[1] > 63 or pos[0] < 0 or pos[1] < 0:
		return gen_rand_joint_state(num_dof)
	else:
		return joint_state

def forward_kinematics(joint_angle_state):
	"""
	use the joint information to map to a pixel position
	"""
	#initialize the xpos and ypos
	xpos = 0
	ypos = 0
	for i in range(1,DOF+1):
		xpos += round(link_length*np.cos(np.sum(joint_angle_state[0,:i])))
		ypos += round(link_length*np.sin(np.sum(joint_angle_state[0,:i])))
	return (int(xpos),int(ypos))

def generate_input_and_target_image():
	#initialize the input and target image
	input_image = np.zeros((64,64),dtype = float)
	target_image = np.zeros((64,64),dtype = float)
	#first generate a random start point and end point for a line
	start_point,end_point,joint_state = gen_random_start_point_and_end_point()
	#pass these to tt.draw_line to get a pos_list
	sp = tt.shape_maker()
	pos_list = sp.draw_line(start_point,end_point,0.01,0.01)
	#now generate addtional points to thicken image
	additional_points = get_points_to_increase_line_thickness(pos_list)
	#now truncate the last point from the lists so that this could be used as the input joint state map
	pos_list_truncated = pos_list[:-1]
	additional_points_truncated = additional_points[:-1]
	#flatten additional points and remove points which are out of bounds
	additional_points_truncated_flattened = [pos for sublist in additional_points_truncated for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	additional_points_flattened = [pos for sublist in additional_points for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	#concatenate the lists
	truncated_concatenated_pos_list = pos_list_truncated + additional_points_truncated_flattened
	concatenated_pos_list = pos_list + additional_points_flattened
	#now use these points to construct the input image
	for pos in truncated_concatenated_pos_list:
		input_image[pos[0],pos[1]] = 1.0

	for pos in concatenated_pos_list:
		target_image[pos[0],pos[1]] = 1.0


	#now also find the joint state required to produce this additional point
	return input_image,target_image,joint_state

def generate_training_data(num):
	joint_state_array = np.zeros((num,DOF), dtype = float)
	target_image_array = np.ndarray(shape = (num,64,64))
	input_image_array = np.ndarray(shape = (num,64,64))
	#now loop through the num of examples and populates these arrays
	for i in range(num):
		input_image,target_image,joint_state = generate_input_and_target_image()
		joint_state_array[i,...] = joint_state
		input_image_array[i,...] = input_image
		target_image_array[i,...] = target_image
	return joint_state_array,target_image_array,input_image_array

#call on training data 

joint_state_array,target_image_array,input_image_array = generate_training_data(20)
with open(ROOT_DIR + "joint_state_array_" + str(DOF) + "DOF.npy", "wb") as f:
	pickle.dump(joint_state_array,f)

with open(ROOT_DIR + "target_image_array_" + str(DOF) + "DOF.npy", "wb") as f:
	pickle.dump(target_image_array,f)

with open(ROOT_DIR + "input_image_array_" + str(DOF) + "DOF.npy","wb") as f:
	pickle.dump(input_image_array,f)
