
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import training_tools as tt


### write a function that can save the files and then load them as required specifically you want to save folders of images 
#generate time array

BATCH_SIZE = 250
HIDDEN_UNITS = 100
EVAL_BATCH_SIZE = 250


IMAGE_SIZE = 64

VALIDATION_SIZE = 250
EPOCHS = 100
ROOT_DIR = "Seq2Seq_Outputs/"

EVAL_FREQUENCY = 20

num_dof = 3

link_length_xdof = 30
link_length_2dof = 50
path_2dof = "Training_Data_First_Arm/"
arm_2dof = tt.two_link_arm(link_length_2dof)
armx = tt.two_link_arm(link_length_xdof)
shape_str_array = ['Rectangle', 'Square', 'Triangle']

NORMALIZATION_FACTOR = 1.

#MODEL GLOBALS
CONV_KERNELS_1 =
CONV_KERNELS_2 = 
FC_2_UNITS
OUTPUT_FEATURES = 2
LEARNING_RATE = 1e-3
Layers = 3


########################################## HANDLING AND EXTRACTING DATA #######################################################################



def load_data():
	data_dict = {}
	for shape_name in shape_str_array:
		with open("Shape_JointSeqs/" + shape_name +'.npy',"rb") as f:
			key = "arm1_" + shape_name
			data_dict[key] = pickle.load(f)

	return data_dict


def find_seq_max_length(data_dict):
	"""
	loop through the lists in the data list in order to figure out the maximum length of each sequence
	"""
	max_length = 1
	length_list = []
	for sequence_list in data_dict.values():
		for sequence in sequence_list:
			length_list.append(np.shape(sequence)[1])
			if np.shape(sequence)[1] > max_length:
				max_length = np.shape(sequence)[1]

	return length_list,max_length


def extract_input_data(num,length_list,max_seq_length):

	time_varying_images = np.zeros([num,IMAGE_SIZE,IMAGE_SIZE,max_seq_length])
	
	for image_num in xrange(num):
			#figure out which shape control needs to be loaded
			shape_name_index = image_control_num % len(shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = image_control_num // len(shape_str_array)

		for timestep in xrange(max_seq_length):
			if timestep < length_list[image_num]:
				#this information may now be combined to load the right control space value
				time_varying_images[image_num,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep))
			else:
				time_varying_images[image_num,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(length_list[image_num]))	
	
	return time_varying_images,length_list

jointseq_dict = load_data()
length_list,SEQ_MAX_LENGTH = find_seq_max_length(jointseq_dict)
time_varying_images,length_array = extract_data(3000,length_list,SEQ_MAX_LENGTH)


######################BUILD THE TENSORFLOW MODEL############################################


x = tf.placeholder(tf.float32, shape = [None,IMAGE_SIZE,IMAGE_SIZE,SEQ_MAX_LENGTH])
#split this into individual images with one channel
image_seq_list = tf.split(3,SEQ_MAX_LENGTH,x)
#now initialize the variable for two conv layers interspersed by pooling layers and finally a fc layer before the list is passed on to the lstm
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,CONV_KERNELS_1],stddev = 0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_1]))
	
		
#define parameters for the second convolutional layer
W_conv2 = tf.Variable(tf.truncated_normal([5,5,CONV_KERNELS_1,CONV_KERNELS_2],stddev = 0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_2]))

		
#define parameters for full connected layer
W_fc1 = tf.Variable(tf.truncated_normal(shape = [IMAGE_SIZE	* IMAGE_SIZE * CONV_KERNELS_2 // 16,FC_2_UNITS],stddev = 0.1)) 
b_fc1 = tf.Variable(tf.constant(0.,shape = [FC_2_UNITS])) 

#initialize a list to hold the lstm inputs
lstm_input_list = []

for image_seq in image_seq_list:
	conv1 = tf.nn.conv2d(image_seq,W_conv1,strides = [1,1,1,1],padding = 'SAME')
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
	pool1 = tf.nn.max_pool(h_conv1, ksize =[1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
	conv2 = tf.nn.conv2d(pool1,W_conv2,strides = [1,1,1,1],padding = 'SAME')
	h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))
	pool2 = tf.nn.max_pool(h_conv2, ksize = [1,3,3,1],strides = [1,2,2,1], padding = 'SAME')
	#Reshape the output from pooling layers to pass to fully connected layers
	h_conv2_reshape = tf.reshape(pool2, shape = [-1,IMAGE_SIZE * IMAGE_SIZE * CONV_KERNELS_2 // 16])
	lstm_input_list.append(tf.nn.relu(tf.matmul(h_conv2_reshape, W_fc1) + b_fc1))



#initialize an lstm cell
lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS,state_is_tuple = True)] * Layers
#pass this list into a multirnn_cell
lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list, state_is_tuple = True)
#split the x input into a list as specified above
x_list = tf.split(0,SEQ_MAX_LENGTH,x_reshape)
#now pass this into a function that generates the rnn graph
outputs,states = tf.nn.rnn(lstm_cell,x_list,dtype = tf.float32)
#now we want to take each element of this output list and pass it into fc layer 
W_fc = tf.Variable(tf.truncated_normal(shape = [HIDDEN_UNITS,OUTPUT_FEATURES],stddev = 0.1))
b_fc = tf.Variable(tf.constant(0., shape = [OUTPUT_FEATURES]))
#loop through the outputs
#but first initialize y_list
#while samples in the batch remain and while timesteps in the sample remain compute the output and append zeros to the output
y_list = []
for output_timestep in outputs:
	y_timestep = tf.nn.elu(tf.matmul(output_timestep,W_fc) + b_fc)
	y_list.append(y_timestep)

#now combine the elements in y_list to get a tensor of shape [SEQ_MAX_LENGTH,None,OUTPUT_FEATURES]
y_tensor = tf.pack(y_list)
y = tf.mul(tf.transpose(y_tensor, [1,0,2]),bin_tensor)
y_split = tf.split(2,OUTPUT_FEATURES,y)
theta_1,theta_2= y_split



def encode_input_image(x_image):
	"""
	Takes an input placeholder for an image
	"""
	#define a place holder for the outputs
	x_image = tf.expand_dims(x_image, -1)
	W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,CONV_KERNELS_1],stddev = 0.1))
	b_conv1 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_1]))
	conv1 = tf.nn.conv2d(x_image,W_conv1,strides = [1,2,2,1],padding = 'SAME')
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
		
	
	#define parameters for the second convolutional layer
	W_conv2 = tf.Variable(tf.truncated_normal([3,3,CONV_KERNELS_1,CONV_KERNELS_2],stddev = 0.1))
	b_conv2 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_2]))
	conv2 = tf.nn.conv2d(h_conv1,W_conv2,strides = [1,2,2,1],padding = 'SAME')
	h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))

	#define a third convolutional layer
	W_conv3 = tf.Variable(tf.truncated_normal([3,3,CONV_KERNELS_2,CONV_KERNELS_3],stddev = 0.1))
	b_conv3 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_3]))W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1
	conv3 = tf.nn.conv2d(h_conv2,W_conv3,strides = [1,2,2,1],padding = 'SAME')
	h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3,b_conv3))

	#Add another convolutional layer
	W_conv4 = tf.Variable(tf.truncated_normal([3,3,CONV_KERNELS_3,CONV_KERNELS_4], stddev = 0.1))
	b_conv4 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_4]))
	conv4 = tf.nn.conv2d(h_conv3,W_conv4,strides = [1,2,2,1],padding = 'SAME')
	h_conv4 = tf.nn.relu(tf.nn.bias_add(conv4,b_conv4))

	#Add an additonal conv layer
	W_conv5 = tf.Variable(tf.truncated_normal([2,2,CONV_KERNELS_4,CONV_KERNELS_5], stddev = 0.1))
	b_conv5 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_5]))
	conv5 = tf.nn.conv2d(h_conv4,W_conv5,strides = [1,2,2,1],padding = 'SAME')
	h_conv5 = tf.nn.relu(tf.nn.bias_add(conv5,b_conv5))
	print h_conv5	

	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*CONV_KERNELS_5])

	
	#define parameters for full connected layer
	W_fc1 = tf.Variable(tf.truncated_normal(shape = [4*CONV_KERNELS_5,FC_UNITS_IMAGE],stddev = 0.1)) 
	b_fc1 = tf.Variable(tf.constant(0.,shape = [FC_UNITS_IMAGE])) 
	h_fc1 = tf.nn.relu(tf.matmul(h_conv5_reshape, W_fc1) + b_fc1)

	image_encode_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
	image_weights = [W_conv1,W_conv2,W_conv3]
	return h_fc1,image_encode_variable_list,image_weights

def encode_joints(x_joints, image_encoder_variable_list):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	#define a fully connected layer
	W_fc1 = tf.Variable(tf.truncated_normal(shape = [DOF,FC_UNITS_JOINTS_1], stddev = 0.1))
	b_fc1 = tf.Variable(tf.constant(0.,shape = [FC_UNITS_JOINTS_1]))
	h_fc1 = tf.nn.relu(tf.matmul(x_joints,W_fc1) + b_fc1)
	#now pass through second fully connected layer
	W_fc2 = tf.Variable(tf.truncated_normal(shape = [FC_UNITS_JOINTS_1, FC_UNITS_JOINTS_FINAL], stddev = 0.1))
	b_fc2 = tf.Variable(tf.constant(0.,shape = [FC_UNITS_JOINTS_FINAL]))
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

	joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]
	joint_weights = [W_fc1]
	
	return h_fc2,joint_encoder_variable_list,joint_weights


def decode_outputs(hidden_vector,decoder_parameter_list):
	"""
	Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
	and generate an image of size [None,64,64,1], do this via 
	"""	
	#Assume FC_UNITS_JOINTS + FC_UNITS_IMAGE is 256
	#then reshape tensor from 2d to 4d to be compatible with deconvoh_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))lution
	batch_size = tf.shape(hidden_vector)[0]
	hidden_image = tf.reshape(hidden_vector, shape = [batch_size,4,4,64])
	
	W_deconv1 = tf.Variable(tf.truncated_normal([2,2,DECONV_OUTPUT_CHANNELS_1,64], stddev = 0.1))
	b_deconv1 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_1]))
	deconv1 = tf.nn.conv2d_transpose(hidden_image,W_deconv1,[batch_size,4,4,DECONV_OUTPUT_CHANNELS_1],[1,1,1,1])
	h_deconv1 = tf.nn.relu(tf.nn.bias_add(deconv1,b_deconv1))

	W_deconv2 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_2,DECONV_OUTPUT_CHANNELS_1], stddev = 0.1))
	b_deconv2 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_2]))
	deconv2 = tf.nn.conv2d_transpose(h_deconv1,W_deconv2,[batch_size,8,8,DECONV_OUTPUT_CHANNELS_2],[1,2,2,1])
	h_deconv2 = tf.nn.relu(tf.nn.bias_add(deconv2,b_deconv2))

	W_deconv3 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_3,DECONV_OUTPUT_CHANNELS_2], stddev = 0.1))
	b_deconv3 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_3]))
	deconv3 = tf.nn.conv2d_transpose(h_deconv2,W_deconv3,[batch_size,16,16,DECONV_OUTPUT_CHANNELS_3],[1,2,2,1])
	h_deconv3 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(deconv3,b_deconv3)),0.5)

	W_deconv4 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_4,DECONV_OUTPUT_CHANNELS_3], stddev = 0.1))
	b_deconv4 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_4]))
	deconv4 = tf.nn.conv2d_transpose(h_deconv3,W_deconv4,[batch_size,32,32,DECONV_OUTPUT_CHANNELS_4],[1,2,2,1])
	h_deconv4 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(deconv4,b_deconv4)),0.5)

	W_deconv5 = tf.Variable(tf.truncated_normal([3,3,1,DECONV_OUTPUT_CHANNELS_4], stddev = 0.1))
	b_deconv5 = tf.Variable(tf.constant(0.1, shape = [1]))
	deconv5 = tf.nn.conv2d_transpose(h_deconv4,W_deconv5,[batch_size,64,64,1],[1,2,2,1])
	h_deconv5 = tf.nn.dropout(tf.nn.bias_add(deconv5,b_deconv5),0.5)

	decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]
	decoder_weights = [W_deconv3,W_deconv4,W_deconv5]

	return tf.squeeze(h_deconv5),decoder_variable_list,decoder_weights


