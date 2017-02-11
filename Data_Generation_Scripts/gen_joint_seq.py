from __future__ import division
import numpy as np
import pickle
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import training_tools as tt
import pickle

root_dir = parent_dir + "/" + "Random_2DOF_Trajectories"
seq_length = 5
link_length = 30 
num_dof = 3
#initialize a 3dof arm with these properties
arm = tt.three_link_arm(link_length)
#initialize a shape maker object to thicken the line
sp = tt.shape_maker()
num_sequences = 20000

def gen_rand_joint_state(num_dof,link_length):
	"""
	Generates random joint state for a specified dof returns a row with each column corresponding to a theta
	"""
	joint_state =  (np.random.rand(1,num_dof) - 0.5)*(4*np.pi)
	pos = arm.forward_kinematics(np.transpose(joint_state))[0]
	if pos[0] > 62 or pos[1] > 62 or pos[0] < 2 or pos[1] < 2:
		return gen_rand_joint_state(num_dof,link_length)
	else:
		return joint_state


#generate a random sequence of joint angles using a delta

def gen_joint_angle_sequence(num_dof,link_length,seq_length):
	#generate the first joint angle
	joint_state = gen_rand_joint_state(3,30)
	#initialize a joint sequence list
	joint_seq_list = [joint_state]
	#we should then compute the x and y value of the end effector position that corresponds to this point
	x,y = arm.forward_kinematics(np.transpose(joint_state))[0]
	#initialize an x and y list
	x_list = [int(x)]
	y_list = [int(y)]
	#now add a delta to the current joint angle state in order to obtain a new joint angle for which to compute the end effector state
	while  len(joint_seq_list) < seq_length:
		joint_state = joint_state + (np.random.rand(1,num_dof) - 0.5)*0.01
 		x,y = arm.forward_kinematics(np.transpose(joint_state))[0]
 		if (not(int(x) in x_list) or not(int(y) in y_list)) and int(x) < 62 and int(x) > 2 and int(y) < 62 and int(y) > 2:
 			joint_seq_list.append(joint_state)
 			x_list.append(int(x))
 			y_list.append(int(y))

	return joint_seq_list,zip(x_list,y_list)


def render_trajectory(pos_list,dir_name):
	#add the root dir to the dir_name to get the dir_path
	dir_path = root_dir + "/" + dir_name
	#create the directory if it exists
	if not(os.path.exists(dir_path)):
		os.makedirs(dir_path)

	#loop through the position list and index out the right portion of the list in order to generate time varying trajectories
	for i in xrange(len(pos_list)):
		pos_sub_list = pos_list[:i+1]
		#once the directory has been created get the additional points needed to thicken the line
		additional_points = sp.get_points_to_increase_line_thickness(pos_sub_list)
		#initialize a grid to plot the trajectory
		temp_grid = tt.grid( "timestep" + str(i),dir_path)
		temp_grid.draw_figure(additional_points + pos_sub_list)
		temp_grid.save_image()


#initialize a numpy array to store the joint angle sequences
joint_angle_sequence_array = np.ndarray(shape = [num_sequences,num_dof,seq_length])
#now loop through the number of sequences that you want to generate
for j in xrange(num_sequences):
	#first get the random joint angle sequence
	joint_seq_list,pos_list = gen_joint_angle_sequence(num_dof,link_length,seq_length)
	#now use the joint_seq_list to render a trajectory
	dir_name = "Trajectory" + str(j) + "/"
	render_trajectory(pos_list,dir_name)
	#furthermore append the joint seq_list to the joint_angle_sequence_array
	joint_angle_sequence_array[j,:,:] = np.transpose(np.squeeze(np.array(joint_seq_list)))
	if (round((j / num_sequences)*100) % 5) == 0 : print "Percentage Completion" +  " " + str(round((j / num_sequences)*100)) 

#save the joint_angle_sequence in the root dir
with open(root_dir + "/" + "random_trajectory.npy","wb") as f:
	pickle.dump(joint_angle_sequence_array,f)
