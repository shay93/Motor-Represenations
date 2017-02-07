from __future__ import division
from model_classes import onetstep_rendered_arm_to_delta_output
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import png
import results_handling as rh
import training_tools as tt
import input_data_handler as dh

num_shape_sequences = 50
step_size = 3
root_dir = "arm2delta/"
#specify all the relevant directories
output_dir = root_dir + "Eval_Output_Images/"
physics_saved_directory = "joint2image/" + "model/" + "model.ckpt"
infer_save_dir = root_dir + "model/" + "model.ckpt"

#create the output directory
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

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
delta_image_list = [np.expand_dims(np.transpose(delta_seq_array[i,:,:,:total_tsteps_list_shape[i]],[2,0,1]) * 255.,-1) for i in xrange(num_shape_sequences)]
#flatten the rendered arm images as well
x_list = [np.expand_dims(np.transpose(rendered_arm_array[i,:,:,:total_tsteps_list_arm[i]],[2,0,1]) * 255., -1) for i in xrange(num_shape_sequences)]

#now initialize the graph by loading the model initializing the variables and then loading the correct values
model_graph = onetstep_rendered_arm_to_delta_output()
#build the graph
op_dict,sess = model_graph.build_graph()
#now initialize and load
model_graph.init_graph_vars(sess,op_dict["init_op"])
model_graph.load_graph_vars(sess,op_dict["physics_saver"],physics_saved_directory)
model_graph.load_graph_vars(sess,op_dict["infer_saver"],infer_save_dir)
#now loop through inputs and evaluate graph predictions and joint angles

def save_images(predictions,target,joint_angle_predictions,directory):
	"""
	Saves images in correct directory also return the last predicted and observed image in each sequence
	"""
	#initialize a three link arm to check whether joint angles have been inferred or not
	three_link_arm = tt.three_link_arm(30)
	#get the sequence length for the current sequence being considered
	seq_length = np.shape(predictions)[0]
	#intialize an array that will hold the observed images obtained after summing all the delta images preceding the current step
	observed_images = np.zeros([seq_length,64,64,1])
	#initialize another array to hold the delta images constructed via the known kinematics of joint angles
	joint_angle_constructed_delta_images = np.zeros([seq_length,64,64,1])
	#similarly define an array to hold output images obtained after summing up these deltas
	joint_angle_constructed_observed_images = np.zeros([seq_length,64,64,1])
	#initialize an empty array to store the flattenen images so that they may be passed to the tile raster function
	flattened_image_array = np.zeros([3,64*64,seq_length])
	#initialize a shape maker object to translate the end effector positions into images that may be saved
	sp = tt.shape_maker()

	#now loop through the predictions to construct the output images at each timestep
	for i in xrange(seq_length):
		observed_images[i ,...] = np.sum(predictions[:i+1,...],axis = 0)
		#for each joint angle construct the delta image 
		#use the three link arm to get the position list from this
		effec_pos = three_link_arm.forward_kinematics(np.expand_dims(joint_angle_predictions[i,...],-1))
		#initialize a grid to store the image
		joint_angle_image_grid = tt.grid("None","None")
		#write the effec pos to the grid
		joint_angle_image_grid.draw_figure(effec_pos)
		#now get the points from the effec pos to make the figure lines thicker
		more_pts = sp.get_points_to_increase_line_thickness(effec_pos)
		#write these points to the figure as well and assign to the array storing the constructed delta images
		joint_angle_constructed_delta_images[i,:,:,0] = joint_angle_image_grid.draw_figure(more_pts)
		#now sum up all the construced images to obtain the observed images at each timestep
		joint_angle_constructed_observed_images[i,...] = np.sum(joint_angle_constructed_delta_images[:i+1,...], axis = 0)
		#do the same for the target delta image
		target_observed_images = np.sum(target[:i+1,...], axis = 0)



	#now take the observed images and add to the flattened array
	for i in xrange(seq_length):
		flattened_image_array[0,:,i] = target_observed_images[i,:,:,0].flatten()
		flattened_image_array[1,:,i] = observed_images[i,:,:,0].flatten()
		flattened_image_array[2,:,i] = joint_angle_constructed_observed_images[i,:,:,0].flatten()
	
	#now that the image array consists of the targets and the prediction split it into a list of images and use the raster function to get the tiled images and png to saver the image appropriately
	image_array_list = np.split(flattened_image_array,seq_length,2)
	for i,image in enumerate(image_array_list):
		image = np.squeeze(image)
		#now pass this to the raster function to obtain the tiled image that may be saved using the png module
		tiled_image = rh.tile_raster_images(image, (64,64), (1,3))
		#now save the tiled image using png
		png.from_array(tiled_image.tolist(),'L').save(directory + "output_image" + str(i) + ".png")

	return observed_images[-1,:,:,0],joint_angle_constructed_observed_images[-1,:,:,0],target_observed_images[-1,:,:,0]

#initialize three arrays to hold the last image in each target and observed image sequence
joint_last_image_array = np.ndarray([len(x_list),64,64])
phys_last_image_array = np.ndarray([len(x_list),64,64])
target_last_image_array = np.ndarray([len(x_list),64,64])

for i,x in enumerate(x_list):
	predictions,test_loss_array = model_graph.evaluate_graph(sess,np.shape(x)[0],{op_dict["x"]: x, op_dict["delta"] : delta_image_list[i]},op_dict["y"],op_dict["delta"], op_dict["loss"])
	#also get the joint angles that are predicted using the sessions object and the placeholder_dict
	joint_angle_predictions = sess.run(op_dict["joint_angle_state"],feed_dict = {op_dict["x"] : x})
	#for a given set of predictions and targets calculated the IoU's, using both the joint angle predictions and the outputs from the physics network calculate the IoU between the last target and observed imag
	if not os.path.exists(output_dir + "Shape_Seq_" + str(i) + "/"):
		os.makedirs(output_dir + "Shape_Seq_" + str(i) + "/")

	#now save the predictions and append the last images to the respective arrays
	phys_last_image_array[i,...],joint_last_image_array[i,...],target_last_image_array[i,...] = save_images(predictions,delta_image_list[i],joint_angle_predictions,output_dir + "Shape_Seq_" + str(i) + "/")
	

def calculate_IOU(predictions,target,file_name,directory):
	"""
	Returns a 2d array of dimension [2,0.99/0.025], first dimension corresponds to the thresholds and second dimension corresponds to 
	"""
	threshold_list = np.arange(0,0.99,step = 0.025)
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

	IoU_array = np.concatenate([np.expand_dims(threshold_list,0),np.expand_dims(IoU_list,0)])

	with open(directory + file_name,"wb") as f:
		pickle.dump(IoU_array,f)


#finally calculate the IoU for each set of images
calculate_IOU(phys_last_image_array,target_last_image_array,"phys_IoU.npy",root_dir)
calculate_IOU(joint_last_image_array,target_last_image_array,"joint_IoU.npy",root_dir)




