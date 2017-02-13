from __future__ import division
from model_classes import onetstep_rendered_arm_to_delta_output
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import png
import results_handling as rh
import training_tools as tt
import input_data_handler as dh

eval_set_size = 200
Epochs = 2
batch_size = 1000
eval_batch_size =  20
#also specify the number of samples
num_shape_sequences = 20
step_size = 2
root_dir = "arm2delta/"
learning_rate = 1e-3
#specify all the relevant directories
log_dir = root_dir + "tmp/"
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


def load_data(num_shape_sequences,step_size):
	#initialize a shape sequence data handler object
	shape_dh = dh.shape_sequence_data_loader(num_shape_sequences)
	_ = shape_dh.find_seq_max_length()
	x_1,x_2 = shape_dh.extract_observed_images(step_size)
	delta_sequence_array = x_2 - x_1
	#perform the same for the rendered arm images
	seq_loader = dh.generic_image_sequence_loader(num_shape_sequences)
	_ = seq_loader.find_seq_max_length()
	rendered_arm_array = seq_loader.extract_seq_images(step_size)
	return delta_sequence_array,shape_dh.total_tsteps_list,rendered_arm_array,seq_loader.total_tsteps_list

delta_seq_array,total_tsteps_list_shape,rendered_arm_array,total_tsteps_list_arm = load_data(num_shape_sequences,step_size)
#now flatten the delta sequence along the batch dimension
delta_images = np.concatenate([np.transpose(delta_seq_array[i,:,:,:total_tsteps_list_shape[i]],[2,0,1]) for i in xrange(num_shape_sequences)])
#flatten the rendered arm images as well
arm_images = np.concatenate([np.transpose(rendered_arm_array[i,:,:,:total_tsteps_list_arm[i]],[2,0,1]) for i in xrange(num_shape_sequences)])

x = np.expand_dims(arm_images,-1)*255.
delta_images = np.expand_dims(delta_images,-1)
print np.max(x)
print np.shape(x)
#now separate the arrays into the training and eval sets
x_train = x[eval_set_size:,...]
delta_train = delta_images[eval_set_size:,...]
#now specify the eval set
x_eval = x[:eval_set_size,...]
delta_eval = delta_images[:eval_set_size,...]
#instantiate physics emulator graph
model_graph = onetstep_rendered_arm_to_delta_output(learning_rate)

#build the graph
op_dict,sess = model_graph.build_graph()

#train_size = num_samples - eval_set_size

#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["x"]] = x_train
placeholder_train_dict[op_dict["delta"]] = delta_train
model_graph.init_graph_vars(sess,op_dict["init_op"])
#load the saved variables for the model graph
model_graph.load_graph_vars(sess,op_dict["physics_saver"],saved_variable_directory)

#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
model_graph.save_graph_vars(sess,op_dict["infer_saver"],save_dir + "model.ckpt")
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["delta"]] = delta_eval
print np.shape(x_eval)
predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["delta"])
#also get the joint angles that are predicted using the sessions object and the placeholder_dict
joint_angle_predictions = sess.run(op_dict["joint_angle_state"],feed_dict = placeholder_eval_dict)

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


def save_images(predictions,target,joint_angle_predictions,directory):
	#initialize a three link arm to check whether joint angles have been inferred or not
	three_link_arm = tt.three_link_arm(30)
	#initialize an empty array to store the flattenen images so that they may be passed to the tile raster function
	image_array = np.zeros([3,64*64,eval_set_size])
	#now loop through all these images and construct an array that may be used to store the images
	for i in range(eval_set_size):
		image_array[0,:,i] = target[i,:,:,0].flatten()
		image_array[1,:,i] = predictions[i,:,:,0].flatten()
		#use the three link arm to get the position list from this
		effec_pos = three_link_arm.forward_kinematics(np.expand_dims(joint_angle_predictions[i,...],-1))
		#initialize a grid to store the image
		joint_angle_image_grid = tt.grid("None","None")
		#write the effec pos to the grid
		joint_angle_image_grid.draw_figure(effec_pos)
		#now get the points from the effec pos to make the figure lines thicker
		#initialize a shape maker to make this possible
		sp = tt.shape_maker()
		more_pts = sp.get_points_to_increase_line_thickness(effec_pos)
		#write these points to the figure as well
		joint_angle_image = joint_angle_image_grid.draw_figure(more_pts)
		image_array[2,:,i] = joint_angle_image.flatten()
	
	#now that the image array consists of the targets and the prediction split it into a list of images and use the raster function to get the tiled images and png to saver the image appropriately
	image_array_list = np.split(image_array,eval_set_size,2)
	for i,image in enumerate(image_array_list):
		image = np.squeeze(image)
		#now pass this to the raster function to obtain the tiled image that may be saved using the png module
		tiled_image = rh.tile_raster_images(image, (64,64), (1,3))
		#now save the tiled image using png
		png.from_array(tiled_image.tolist(),'L').save(directory + "output_image" + str(i) + ".png")


calculate_IOU(predictions,x_eval,root_dir)

save_images(predictions,x_eval, joint_angle_predictions,output_dir)