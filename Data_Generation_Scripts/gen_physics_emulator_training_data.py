from __future__ import division
import numpy as np
import pickle
import sys
import os

save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Experiment_3/"
DOF = 3



def gen_rand_joint_state(num_dof,link_length):
	"""
	Generates random joint state for a specified dof returns a row with each column corresponding to a theta
	"""
	joint_state =  (np.random.rand(1,num_dof) - 0.5)*(2*np.pi)
	pos = forward_kinematics(joint_state,num_dof,link_length)
	if pos[0] > 62 or pos[1] > 62 or pos[0] < 2 or pos[1] < 2:
		return gen_rand_joint_state(num_dof,link_length)
	else:
		return joint_state

def forward_kinematics(joint_angle_state,num_dof,link_length):
	"""
	use the joint information to map to a pixel position
	"""
	#initialize the xpos and ypos
	xpos = 0
	ypos = 0
	for i in range(1,num_dof+1):
		xpos += (link_length*np.cos(np.sum(joint_angle_state[0,:i])))
		ypos += (link_length*np.sin(np.sum(joint_angle_state[0,:i])))
	return (int(xpos),int(ypos))



def increase_point_thickness(pt,grid_size):
	"""
	Returns a list of tuple with each element in a list corresponding to the (x,y) position of a point in the grid
	"""

	if grid_size % 2 == 0:
		return ValueError("Grid Size must be odd")
	
	grid_width = int((grid_size -1) / 2)
	pt_list = []
	for j in range(-grid_width,grid_width + 1):
		left_list = [(pt[0] - j,pt[1] - i) for i in range(1,grid_width + 1)]
		right_list = [(pt[0] - j,pt[1] + i) for i in range(1,grid_width + 1)]
		middle_list = left_list + [(pt[0] - j,pt[1])] + right_list
		pt_list.extend(middle_list)

	return pt_list


def constuct_image(pt_list):
	image_array = np.zeros((64,64))
	for pt in pt_list:
		image_array[pt[0],pt[1]] = 255.

	return image_array


def gen_data_sample(num_dof,link_length,thickness):
	#first generate joint angle state
	joint_angle_state = gen_rand_joint_state(num_dof,link_length)
	#then use joint angle state to get a pt
	pt = forward_kinematics(joint_angle_state,num_dof,link_length)
	#then get point list then generate the image corresponding to this joint angle position
	output_image = constuct_image(increase_point_thickness(pt,thickness))
	return joint_angle_state,output_image



def generate_data_batch(num_samples):
	image_batch_array = np.zeros((num_samples,64,64))
	joint_angle_array = np.zeros((num_samples,3))
	for i in xrange(num_samples):
		joint_angle_state,output_image = gen_data_sample(3,30,3)
		image_batch_array[i,...] = output_image
		joint_angle_array[i,...] = joint_angle_state

	return image_batch_array,joint_angle_array

image_batch_array,joint_angle_array = generate_data_batch(int(sys.argv[1]))

with open(save_dir + "joint_angle_array.npy","wb") as f:
	pickle.dump(joint_angle_array,f)

with open(save_dir + "image_batch_array.npy","wb") as f:
	pickle.dump(image_batch_array,f)
