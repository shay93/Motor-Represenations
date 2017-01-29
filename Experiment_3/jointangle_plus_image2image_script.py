from __future__ import division
from model_classes import physics_emulator
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
import png

sys.path.append(os.path.dirname(os.getcwd()))

import results_handling as rh


eval_set_size = 200
Epochs = 10
batch_size = 1000
eval_batch_size =  20
root_dir = "joint_plus_image2image/"
log_dir = root_dir + "tmp/summary/"
save_dir = root_dir + "model/" 

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

output_dir = root_dir + "Output_Images/"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#load the data first
def load_data(num):
	with open("joint_angle_array.npy","rb") as f:
		joint_state_array = pickle.load(f)[:num,...]

	with open("image_batch_array.npy","rb") as f:
		delta_image_array = pickle.load(f)[:num,...]

	return joint_state_array,delta_image_array



joint_state_array,delta_image_array = load_data(20000)

#form get the delta image
delta_image_array = np.expand_dims(delta_image_array,-1)
#now separate the arrays into the training and eval sets
joint_state_array_train = joint_state_array[eval_set_size:,...]
delta_image_array_train = delta_image_array[eval_set_size:,...]
#now specify the eval se
joint_state_array_eval = joint_state_array[:eval_set_size,...]
delta_image_array_eval = delta_image_array[:eval_set_size,...]
#instantiate physics emulator graph
pe = physics_emulator(1e-3)

#build the graph
op_dict,sess = pe.build_graph()
#initialize the variables
pe.init_graph_vars(sess,op_dict["init_op"])

#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["y_"]] = delta_image_array_train
placeholder_train_dict[op_dict["x_image"]] = delta_image_array_train
placeholder_train_dict[op_dict["joint_angle_state"]] = joint_state_array_train

#pass the placeholder dict to the train graph function
pe.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
pe.save_graph_vars(sess,op_dict["saver"],save_dir + "model.ckpt")
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["y_"]] = delta_image_array_eval
placeholder_eval_dict[op_dict["x_image"]] = delta_image_array_eval
placeholder_eval_dict[op_dict["joint_angle_state"]] = joint_state_array_eval


predictions,test_loss_array = pe.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["y_"])


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



calculate_IOU(predictions,delta_image_array_eval,root_dir)

save_images(predictions,delta_image_array_eval,output_dir)
