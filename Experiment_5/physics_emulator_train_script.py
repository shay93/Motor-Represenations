from __future__ import division 
from models_classes import physics_emulator_fixed_joint_seq
import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import pickle
import png
import results_handling as rh

root_dir = "physics_emulator/"
num_sequences = 20000
eval_set_size = 100
load_dir = parent_dir + "/" + "Random_3DOF_Trajectories"
log_dir = root_dir + "tmp/"
save_dir = root_dir + "model/"
output_dir = root_dir + "Output_Images/"
###model parameters
learning_rate = 1e-3
Epochs = 2
batch_size = 500


#check if the directories exist and create them if necessary
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)


#first thing is to load the data
def load_image_data(num_sequences,seq_length):
	"""
	Load data in format [Batch,3*seq_length]
	"""
	#initialize an empty array to hold the sequences
	image_array = np.ndarray(shape = [num_sequences,64,64,seq_length])
	#load data by looping over relevant directory and by picking out the final image in each case
	for i in xrange(num_sequences):
		dir_name = load_dir + "/" + "Trajectory" + str(i)
		#now load the final image from this driectory
		for j in xrange(seq_length):
			image_array[i,:,:,j] = plt.imread(dir_name + "/" + "timestep" + str(j) + ".png")

	#return this image array
	return image_array


#similarly load the joint sequence data
def load_joint_sequence(num_sequences):

	with open("random_trajectory.npy","rb") as f:
		joint_seq = pickle.load(f)

	#now only index out the num of sequences corresponding to the image data
	joint_seq = joint_seq[:num_sequences,...]
	#furthermore reshape into a 2d array
	joint_seq = np.reshape(joint_seq,shape = [num_sequences,-1])
	return joint_seq


y = load_image_data(num_sequences)
x = load_joint_sequence(num_sequences)
#separate out the training test set data
x_train = x[eval_set_size:,...]
y_train = y[eval_set_size:,...]
#separate out the eval set
x_eval = x[:eval_set_size,...]
y_eval = y[:eval_set_size,...]

#now instantiate the model
model_graph = physics_emulator_fixed_joint_seq(learning_rate)

#build the graph
op_dict,sess = model_graph,build_graph()

placeholder_train_dict = {}
placeholder_train_dict[op_dict["x"]] = x_train
placeholder_train_dict[op_dict["y_"]] = y_train
model_graph.init_graph_vars(sess,op_dict["init_op"])
#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
model_graph.save_graph_vars(sess,op_dict["saver"],save_dir + "model.ckpt")
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["y"]] = y_eval
print np.shape(x_eval)
predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"],op_dict["delta"])

def save_images(predictions,target,directory):
	#initialize an empty array to store the flattenen images so that they may be passed to the tile raster function
	rh_image_array = np.zeros([2,64*64,eval_set_size])
	#now loop through all these images and construct an array that may be used to store the images
	for i in range(eval_set_size):
		rh_image_array[0,:,i] = target[i,:,:,0].flatten()
		rh_image_array[1,:,i] = predictions[i,:,:,0].flatten()	
	#now that the image array consists of the targets and the prediction split it into a list of images and use the raster function to get the tiled images and png to saver the image appropriately
	image_array_list = np.split(rh_image_array,eval_set_size,2)
	for i,image in enumerate(image_array_list):
		image = np.squeeze(image)
		#now pass this to the raster function to obtain the tiled image that may be saved using the png module
		tiled_image = rh.tile_raster_images(image, (64,64), (1,2))
		#now save the tiled image using png
		png.from_array(tiled_image.tolist(),'L').save(directory + "output_image" + str(i) + ".png")

save_images(predictions,y_eval,output_dir)
rh.calculate_IOU(predictions,y_eval,"IoU.npy",root_dir)