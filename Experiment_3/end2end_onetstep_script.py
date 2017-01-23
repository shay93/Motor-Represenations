from __future__ import division
from model_classes import onetstep_observed_to_output
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import results_handling as rh

eval_set_size = 400
Epochs = 10000
batch_size = 1000
eval_batch_size =  20
#also specify the number of samples
num_shape_sequences = 500
num_samples = 20000
root_dir = "end2end_onetstep/"
learning_rate = 1e-3
shape_str_array = ['Rectangle', 'Square', 'Triangle']

#specify all the relevant directories
log_dir = root_dir + "tmp/summary/"
save_dir = root_dir + "model/"
output_dir = root_dir + "Output_Images/"
saved_variable_directory = "joint2image/" + "model/" + "model.ckpt"
shape_dir = os.path.dirname(os.getcwd()) + "/Shapes/"

#check if the directories exist and create them if necessary
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)


def find_seq_max_length(num_of_samples):
	#initialize a list to record the total number of tsteps for each time varying image
	total_tsteps_list = []
	for image_num in xrange(num_of_samples):
		#figure out which shape control needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		total_tsteps_list.append(len(os.listdir(shape_dir + shape_str_array[shape_name_index] + str(shape_index))))
	return max(total_tsteps_list),total_tsteps_list	


def extract_observed_images(shape_sequence_num,total_tsteps_list,max_seq_length):

	x_1 = np.ndarray(shape = (shape_sequence_num,64,64,max_seq_length), dtype = np.float32)
	x_2 = np.ndarray(shape = (shape_sequence_num,64,64,max_seq_length), dtype = np.float32)
	for shape_sequence_index in xrange(shape_sequence_num):
		#figure out which shape control needs to be loaded
		shape_name_index = shape_sequence_index % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = shape_sequence_index // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[shape_sequence_index]
		for timestep in xrange(max_seq_length):
			if timestep < total_tsteps:
				#load the next observed image timestep if the max time step has not been reached yet
				x_2[shape_sequence_index,:,:,timestep] = plt.imread(shape_dir + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + '.png')
			else:
				#if the max time step has been reached i.e. the complete shape drawing has been observed then continue loading the last image for the remaining timesteps
				x_2[shape_sequence_index,:,:,timestep] = plt.imread(shape_dir + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(total_tsteps - 1) + '.png')	
		
		#now get x_1_temp
		x_1[shape_sequence_index,...] = np.concatenate((np.zeros((64,64,1)),x_2[shape_sequence_index,:,:,:max_seq_length - 1]),axis = 2)
	return x_1,x_2

def load_images(num_samples,num_shape_sequences):
	"""
	Should return x_1 and x_2 of shape [num_samples,64,64,1]
	"""
	#initialize an array for x_1 and x_2
	x_1 = np.ndarray(shape = [num_samples,64,64,1])
	x_2 = np.ndarray(shape = [num_samples,64,64,1])
	#this gives you the maximum number of images in a sequence that comprise a single shape being drawn
	max_seq_length,total_tsteps_list = find_seq_max_length(num_shape_sequences)
	#once the max sequence length has been obtained one may extract the images into an array comprising of images at t and images at t+1
	x_1_array,x_2_array = extract_observed_images(num_shape_sequences,total_tsteps_list,max_seq_length)
	#now generate a list of randome indexes for the shape position
	random_shape_sequence_index = np.random.randint(0,num_shape_sequences,size = [num_samples])
	#for each of these random shape sequence indexes pick an observed image that has not been completed
	random_image_index = [np.random.randint(0,total_tsteps_list[shape_sequence_index] - 10) for shape_sequence_index in random_shape_sequence_index]
	#zip together these random indices so that they are easier to work with
	random_image_pos_tuple = zip(random_shape_sequence_index,random_image_index)
	#use the random positions to load random observed images into the earlier initialized arrays
	for i,pos_tuple in enumerate(random_image_pos_tuple):
		x_1[i,:,:,0] = x_1_array[pos_tuple[0],:,:,pos_tuple[1]]
		x_2[i,:,:,0] = x_2_array[pos_tuple[0],:,:,pos_tuple[1]]

	return x_1,x_2

x_1,x_2 = load_images(num_samples,num_shape_sequences)
print np.shape(x_1)
#now separate the arrays into the training and eval sets
x_1_train = x_1[eval_set_size:,...]
x_2_train = x_2[eval_set_size:,...]
#now specify the eval set
x_1_eval = x_1[:eval_set_size,...]
x_2_eval = x_2[:eval_set_size,...]
#instantiate physics emulator graph
model_graph = onetstep_observed_to_output(learning_rate)

#build the graph
op_dict,sess = model_graph.build_graph()

train_size = num_samples - eval_set_size

#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["x_2"]] = x_2_train
placeholder_train_dict[op_dict["x_1"]] = x_1_train
model_graph.init_graph_vars(sess,op_dict["init_op"])
#load the saved variables for the model graph
model_graph.load_graph_vars(sess,op_dict["saver"],saved_variable_directory)

#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
#model_graph.save_graph_vars(sess,op_dict["saver"],save_dir)
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x_2"]] = x_2_eval
placeholder_eval_dict[op_dict["x_1"]] = x_1_eval
print np.shape(x_1_eval)
predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["x_2"])


def calculate_IOU(predictions,target,directory):
	threshold_list = np.arange(0,0.9,step = 0.025)
	IoU_list = []
	for i,threshold in enumerate(threshold_list):
		good_mapping_count = 0
		bad_mapping_count = 0
		for i in range(eval_set_size):
			arr_pred = np.nonzero(np.round(predictions[i,...]))
			pos_list_pred = zip(arr_pred[0],arr_pred[1])
			arr_input = np.nonzero(target[i,...])
			pos_list_input = zip(arr_input[0],arr_input[1])
			intersection = set(pos_list_pred) & set(pos_list_input)
			union = set(pos_list_input + pos_list_pred)
			if (len(intersection) / len(union)) > threshold:
				good_mapping_count += 1
			else:
				bad_mapping_count += 1

		IoU_list.append(good_mapping_count / eval_set_size)


	with open(directory + "percentage_correct.npy","wb") as f:
		pickle.dump(IoU_list,f)


def save_images(predictions,target,directory):
	#initialize an empty array to store the flattenen images so that they may be passed to the tile raster function
	image_array = np.zeros([2,64*64,eval_set_size])
	#now loop through all these images and construct an array that may be used to store the images
	for i in range(eval_set_size):
		image_array[0,:,i] = target[i,:,:,0].flatten()
		image_array[1,:,i] = predictions[i,:,:,0].flatten()
	#now that the image array consists of the targets and the prediction split it into a list of images and use the raster function to get the tiled images and png to saver the image appropriately
	image_array_list = np.split(image_array,eval_set_size,2)
	for i,image in enumerate(image_array_list):
		image = np.squeeze(image)
		#now pass this to the raster function to obtain the tiled image that may be saved using the png module
		tiled_image = rh.tile_raster_images(image, (64,64), (1,2))
		#now save the tiled image using png
		png.from_array(tiled_image.tolist(),'L').save(directory + "output_image" + str(i) + ".png")


calculate_IOU(predictions,x_2_eval,root_dir)

save_images(predictions,x_2_eval,output_dir)
