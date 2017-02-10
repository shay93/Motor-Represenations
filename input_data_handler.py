from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

class shape_sequence_data_loader:
	
	def __init__(self, num_sequences, shape_str_array = ['Rectangle','Square','Triangle'], shape_dir = os.path.dirname(os.path.abspath(__file__)) + "/Shapes/" ):
		self.shape_str_array = shape_str_array
		self.num_sequences =  num_sequences
		self.shape_dir = shape_dir
		self.total_tsteps_list = []
	
	def find_seq_max_length(self):
		self.total_tsteps_list = []
		for seq_num in xrange(self.num_sequences):
			#figure out which shape control needs to be loaded
			shape_name_index = seq_num % len(self.shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = seq_num // len(self.shape_str_array)
			file_list = [file_name for file_name in os.listdir(self.shape_dir + self.shape_str_array[shape_name_index] + str(shape_index)) if "png" in file_name]
			self.total_tsteps_list.append(len(file_list))
		return self.total_tsteps_list	

	def extract_observed_images(self,step_size):
		#get the max sequence length
		max_seq_length = max(self.total_tsteps_list)
		x_1 = np.ndarray(shape = (self.num_sequences,64,64,(max_seq_length - 1) // step_size + 1), dtype = np.float32)
		x_2 = np.ndarray(shape = (self.num_sequences,64,64,(max_seq_length - 1) // step_size + 1), dtype = np.float32)
		for shape_sequence_index in xrange(self.num_sequences):
			#figure out which shape control needs to be loaded
			shape_name_index = shape_sequence_index % len(self.shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = shape_sequence_index // len(self.shape_str_array)
			total_tsteps = 	self.total_tsteps_list[shape_sequence_index]
			for timestep in xrange(0,max_seq_length,step_size):
				if timestep < total_tsteps:
					#load the next observed image timestep if the max time step has not been reached yet
					x_2[shape_sequence_index,:,:,timestep // step_size] = plt.imread(self.shape_dir + self.shape_str_array[shape_name_index] + str(shape_index) + "/" + self.shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + '.png')
				else:
					#if the max time step has been reached i.e. the complete shape drawing has been observed then continue loading the last image for the remaining timesteps
					x_2[shape_sequence_index,:,:,timestep // step_size] = plt.imread(self.shape_dir + self.shape_str_array[shape_name_index] + str(shape_index) + "/" + self.shape_str_array[shape_name_index] + str(shape_index) + "_" + str(total_tsteps - 1) + '.png')
			
			#now get x_1_temp
			x_1[shape_sequence_index,...] = np.concatenate((np.zeros((64,64,1)),x_2[shape_sequence_index,:,:,: (max_seq_length - 1) // step_size]),axis = 2)
		self.total_tsteps_list = [(max_tstep - 1) // step_size for max_tstep in self.total_tsteps_list]
		return x_1,x_2

	def get_binary_loss(self):
		"""
		use the tstep list to get a numpy array of 1s and 0s to zero out the loss
		"""
		binary_loss = np.zeros((len(self.total_tsteps_list),max(self.total_tsteps_list)),dtype = np.float32)
		for i,max_tstep in enumerate(self.total_tsteps_list):
			binary_loss[i,:max_tstep] = np.ones(max_tstep,dtype = np.float32)	
		return binary_loss


class generic_image_sequence_loader:

	def __init__(self,num_sequences, shape_str_array = ['Rectangle','Square','Triangle'], load_dir = os.path.dirname(os.path.abspath(__file__)) + "/Planar_Arm_Rendering/"):
		self.shape_str_array = shape_str_array
		self.num_sequences = num_sequences
		self.load_dir = load_dir
		self.total_tsteps_list = []

	def find_seq_max_length(self):
 		self.total_tsteps_list = []
		for seq_num in xrange(self.num_sequences):
			#figure out which shape control needs to be loaded
			shape_name_index = seq_num % len(self.shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = seq_num // len(self.shape_str_array)
			file_list = [file_name for file_name in os.listdir(self.load_dir + self.shape_str_array[shape_name_index] + str(shape_index)) if "png" in file_name]
			self.total_tsteps_list.append(len(file_list))
		return self.total_tsteps_list	


	def extract_seq_images(self,step_size):
		#get the max sequence length
		max_seq_length = max(self.total_tsteps_list)
		x = np.ndarray(shape = (self.num_sequences,64,128,(max_seq_length - 1) // step_size + 1), dtype = np.float32)
		for sequence_index in xrange(self.num_sequences):
			#figure out which shape 
			shape_name_index = sequence_index % len(self.shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = sequence_index // len(self.shape_str_array)
			total_tsteps = 	self.total_tsteps_list[sequence_index]
			for timestep in xrange(1,max_seq_length + 1,step_size):
				if timestep < total_tsteps:
					#load the next observed image timestep if the max time step has not been reached yet
					x[sequence_index,:,:,(timestep - 1) // step_size] = plt.imread(self.load_dir + self.shape_str_array[shape_name_index] + str(shape_index) + "/" + "timestep" + str(timestep) + '.png')
				else:
					#if the max time step has been reached i.e. the complete shape drawing has been observed then continue loading the last image for the remaining timesteps
					x[sequence_index,:,:,(timestep - 1) // step_size] = plt.imread(self.load_dir + self.shape_str_array[shape_name_index] + str(shape_index) + "/" + "timestep" + str(total_tsteps - 1) + '.png')
			
			#now get x_1_temp
		self.total_tsteps_list = [(max_tstep - 1) // step_size for max_tstep in self.total_tsteps_list]
		return x
