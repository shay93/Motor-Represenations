from __future__ import division
import numpy as np
import tensorflow as tf
import pickle


#figure out a way to load data from the saved control array in a format that is acceptable
#for the variable seq2seq model, i.e. generate an x_list array a y_list array and a termination timestep array
EPOCHS = 5
BATCH_SIZE = 12
ENCODER_MAX_LENGTH = 250
INPUT_FEATURES = 2
number_of_shapes = 3
num_shapes_per_Epoch = 3000
num_each_shape_per_batch = BATCH_SIZE // number_of_shapes



def extract_data_lstm():
	"""
	Given the number of EPOCHS the number of shapes being used and the batch size
	this function should be able to extract the state of the arms used to draw shapes
	in a format that is acceptable for variable_seq2seq.py
	inputs: -
	returns: Three lists with length equal to the number of batches required to traverse the EPOCHS.
			 The lists are the termination tstep list, the x_list and the y_list  
	"""
	
	#first things first load the saved state array
	rectangle_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Rectangle_50.npy', 'rb'))
	square_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Square_50.npy', 'rb'))
	triangle_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Triangle_50.npy', 'rb'))

	rectangle_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Rectangle_80.npy', 'rb'))
	square_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Square_80.npy', 'rb'))
	triangle_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Triangle_80.npy', 'rb'))

	#each of these states are lists with a thousand elements, the aim is to now create batches out of these
	#lets first initialize the lists that we are dealing with the length of the list should be equal to 
	num_batches_in_Epoch =  num_shapes_per_Epoch // BATCH_SIZE
	#now we know the length of x_list and y_list
	x_list = [0] * (num_batches_in_Epoch * EPOCHS)
	y_list = [0] * (num_batches_in_Epoch * EPOCHS)
	termination_tstep_list = [0] * (num_batches_in_Epoch * EPOCHS)

	#inorder to make looping easier define a shape array
	shape_state_array_first_arm = [rectangle_state_first_arm,square_state_first_arm,triangle_state_first_arm]
	shape_state_array_second_arm = [rectangle_state_second_arm,square_state_second_arm,triangle_state_second_arm]
	
	for batch_num in range(num_batches_in_Epoch * EPOCHS):
		#initialize an empty array of zeros as x_list
		x_temp = np.zeros([BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH])
		y_temp = np.zeros([BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		
		batch_index = batch_num % num_batches_in_Epoch
		termination_tstep = 0
		
		for j,shape_state in enumerate(shape_state_array_first_arm):
			for i in range(num_each_shape_per_batch):
				shape_index = batch_index*4 + i
				#get the number of time steps in each input
				_,timesteps = np.shape(shape_state[shape_index])
				if timesteps > termination_tstep:
					termination_tstep = timesteps
				x_temp[j*num_each_shape_per_batch + i,0:INPUT_FEATURES,:timesteps] = shape_state[shape_index]

		for j,shape_state in enumerate(shape_state_array_second_arm):
			for i in range(num_each_shape_per_batch):
				shape_index = batch_index*4 + i
				#get the number of time steps in each input
				_,timesteps = np.shape(shape_state[shape_index])
				y_temp[j*num_each_shape_per_batch + i,0:OUTPUT_FEATURES,:timesteps] = shape_state[shape_index]
		
		termination_tstep_list[batch_num] = [termination_tstep]
		x_list[batch_num] = x_temp
		y_list[batch_num] = y_temp

	return x_list,y_list,termination_tstep_list

