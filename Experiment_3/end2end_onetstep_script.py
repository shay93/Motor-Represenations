from __future__ import division
from model_classes import onetstep_observed_to_output
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

eval_set_size = 200
Epochs = 10
batch_size = 1000
eval_batch_size =  20
root_dir = "end2end_onetstep_script/"
log_dir = root_dir + "tmp/summary/"
save_dir = root_dir + "model/"
saved_variable_directory = "joint2image/" + "model/" + "model.ckpt"

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
delta_image_array_train = delta_image_array[eval_set_size:,...]
#now specify the eval set
delta_image_array_eval = delta_image_array[:eval_set_size,...]
#instantiate physics emulator graph
model_graph = onetstep_observed_to_output(1e-3)

#build the graph
op_dict,sess = model_graph.build_graph()

train_size = 20000 - eval_set_size

#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["x_2"]] = delta_image_array_train
placeholder_train_dict[op_dict["x_1"]] = np.zeros([train_size,64,64,1])
model_graph.init_graph_vars(sess,op_dict["init_op"])
#load the saved variables for the model graph
model_graph.load_graph_vars(sess,op_dict["saver"],saved_variable_directory)

#pass the placeholder dict to the train graph function
sess = model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
#model_graph.save_graph_vars(sess,op_dict["saver"],save_dir)
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x_2"]] = delta_image_array_eval
placeholder_eval_dict[op_dict["x_1"]] = np.zeros([eval_set_size,64,64,1])

predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["y_"])


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
	for i in range(eval_set_size):
		plt.imsave(directory + "output_image" + str(i) + ".png", predictions[i,:,:,0], cmap = "Greys_r")
		plt.imsave(directory + "target_image" + str(i) + ".png", target[i,:,:,0], cmap = "Greys_r")


calculate_IOU(predictions,delta_image_array_eval,root_dir)

save_images(predictions,delta_image_array_eval,output_dir)
