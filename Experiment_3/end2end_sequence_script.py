from __future__ import division
from model_classes import observed_to_output_seq2seq
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import png
import sys
import string
sys.path.append(os.path.dirname(os.getcwd()))

import results_handling as rh
import training_tools as tt
import tensorflow as tf
eval_set_size = 40
Epochs = 1
batch_size = 50
eval_batch_size =  20
root_dir = "end2end_sequence/"
log_dir = root_dir + "tmp/summary/"
save_dir = root_dir + "model/"
saved_variable_directory = "joint2image/" + "model/" + "model.ckpt"
NUM_SHAPE_SEQUENCES = 2000
DOF = 3
LINK_LENGTH = 30
shape_str_array = ['Rectangle','Square','Triangle']


if not os.path.exists(log_dir):
	os.makedirs(log_dir)

output_dir = root_dir + "Output_Images/"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

shape_dir = string.join(os.getcwd().split("/")[:-1], "/") + "/Shapes/"

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

def get_binary_loss(total_tsteps_list,max_seq_length):
	"""
	use the tstep list to get a numpy array of 1s and 0s to zero out the loss
	"""
	binary_loss = np.zeros((len(total_tsteps_list),max_seq_length),dtype = np.float32)
	for i,max_tstep in enumerate(total_tsteps_list):
		binary_loss[i,:max_tstep- 4] = np.ones(max_tstep - 4,dtype = np.float32)	
	return binary_loss

SEQ_MAX_LENGTH,total_tsteps_list = find_seq_max_length(NUM_SHAPE_SEQUENCES)


print "Sequence Max Length is ",SEQ_MAX_LENGTH
x_1_array,x_2_array = extract_observed_images(NUM_SHAPE_SEQUENCES,total_tsteps_list,SEQ_MAX_LENGTH)
binary_loss_array = get_binary_loss(total_tsteps_list,SEQ_MAX_LENGTH)
#get the previous time step by appending 
#split this data into a training and validation set
x_2_image_array_train = x_2_array[eval_set_size:,...]
x_1_image_array_train = x_1_array[eval_set_size:,...]
binary_loss_array_train = binary_loss_array[eval_set_size:,...]
#now specify the eval set
x_2_image_array_eval = x_2_array[:eval_set_size,...]
x_1_image_array_eval = x_1_array[:eval_set_size,...]
binary_loss_array_eval = binary_loss_array[:eval_set_size,...]

#instantiate physics emulator graph
model_graph = observed_to_output_seq2seq(1e-4,SEQ_MAX_LENGTH)

#build the graph
op_dict,sess = model_graph.build_graph()
print len(tf.all_variables())
#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["x_2_sequence"]] = x_2_image_array_train
placeholder_train_dict[op_dict["x_1_sequence"]] = x_1_image_array_train
placeholder_train_dict[op_dict["binary_loss_tensor"]] = binary_loss_array_train


train_size = 20000 - eval_set_size

model_graph.init_graph_vars(sess,op_dict["init_op"])
#load the saved variables for the model graph
model_graph.load_graph_vars(sess,op_dict["saver"],saved_variable_directory)

#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)

#model_graph.save_graph_vars(sess,op_dict["saver"],save_dir)
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x_2_sequence"]] = x_2_image_array_eval
placeholder_eval_dict[op_dict["x_1_sequence"]] = x_1_image_array_eval
placeholder_eval_dict[op_dict["binary_loss_tensor"]] = binary_loss_array_eval


predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["x_2_sequence"])
#also get the joint angle sequence
joint_angle_sequence = sess.run(op_dict["joint_angle_sequence"], feed_dict = placeholder_eval_dict)


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


def save_output_images(predictions,joint_angle_sequence_batch,target):
	"""
	Save the output shapes to the output root directory
	"""
	prediction_size = np.shape(predictions)[0]
	#multiply predictions by scalar 255 so that they can be sved as grey map images
	predictions = predictions * 255


	#initialize the three link arm that will compute the output images from the joint angles inferred
	three_link_arm = tt.three_link_arm(LINK_LENGTH)
	for output_image_num in xrange(prediction_size):
		#Get the shape name and index number in order to save correctly
		shape_name_index = output_image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = output_image_num // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[output_image_num]
		if total_tsteps > SEQ_MAX_LENGTH:
			total_tsteps = SEQ_MAX_LENGTH
		shape_name = shape_str_array[shape_name_index]
		shape_output_dir = output_dir + shape_str_array[shape_name_index] + str(shape_index) + "/"
		#index out a joint angle sequence from the batch
		joint_angle_sequence = joint_angle_sequence_batch[output_image_num,:,:]
		#create this directory if it doesnt exist
		if not(os.path.exists(shape_output_dir)):
			os.makedirs(shape_output_dir)
		
		for tstep in xrange(total_tsteps):
				#index out the right joing angle state
				joint_angle_subseq = joint_angle_sequence[:,:tstep + 1]
				#use the three link arm to get the position list from this
				effec_pos = three_link_arm.forward_kinematics(joint_angle_subseq)
				#initialize a grid to store the image
				image_grid = tt.grid("joint_pred_" + shape_name + str(shape_index) + '_' + str(tstep),shape_output_dir)
				#write the effec pos to the grid
				image_grid.draw_figure(effec_pos)
				#now get the points from the effec pos to make the figure lines thicker
				#initialize a shape maker to make this possible
				sp = tt.shape_maker()
				more_pts = sp.get_points_to_increase_line_thickness(effec_pos)
				#write these points to the figure as well
				image_grid_array = image_grid.draw_figure(more_pts)
				#flatten the two images and 
				flattened_image_array = np.zeros([3,64*64])
				flattened_image_array[0,:] = target[output_image_num,:,:,tstep].flatten()
				flattened_image_array[1,:] = predictions[output_image_num,:,:,tstep].flatten()
				flattened_image_array[2,:] = image_grid_array.flatten()
				tiled_image = rh.tile_raster_images(flattened_image_array, (64,64), (1,3))
				#now save the tiled image using png
				png.from_array(tiled_image.tolist(),'L').save(shape_output_dir + shape_name + str(shape_index) + '_' + str(tstep) + '.png')




calculate_IOU(predictions,x_2_image_array_eval,root_dir)

save_output_images(predictions,joint_angle_sequence,x_2_image_array_eval)
