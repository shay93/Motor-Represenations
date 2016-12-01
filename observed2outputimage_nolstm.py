

from __future__ import division
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import training_tools as tt
import os

#define a max sequence length
DOF = 3
#model globals
NUM_SHAPE_SEQUENCES = 50
EVAL_SHAPE_SEQUENCES = 6
IMAGE_SIZE = 64
BATCH_SIZE = 1000
learning_rate = 1e-4
EVAL_BATCH_SIZE = 100
EPOCHS = 3000
ROOT_DIR = "observed_to_reconstructed_shapes/"
SUMMARY_DIR = "tmp/summary_logs"
model_dir = "Joints_to_Image/tmp/model.cpkt"
OUTPUT_DIR = ROOT_DIR + "Output_Images/"
EVAL_FREQUENCY = 2000
shape_str_array = ['Rectangle', 'Square', 'Triangle']


#####THIS MODEL SHOULD TAKE IN TWO INPUT IMAGES x_1 and x_2 and should infer the joint angle that maps x_1 to x_2####################
#create the Root dir if it does not exist
if not(os.path.exists(ROOT_DIR)):
	os.makedirs(ROOT_DIR)
#create the summary directory if it does not exist
if not(os.path.exists(ROOT_DIR + SUMMARY_DIR)):
	os.makedirs(ROOT_DIR + SUMMARY_DIR)

##################################################LOAD THE DATA TO TRAIN THE MODEL##########################################

def find_seq_max_length(num_of_samples):
	#initialize a list to record the total number of tsteps for each time varying image
	total_tsteps_list = []
	for image_num in xrange(num_of_samples):
		#figure out which shape control needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		total_tsteps_list.append(len(os.listdir("Shapes/" + shape_str_array[shape_name_index] + str(shape_index))))
	return max(total_tsteps_list),total_tsteps_list	


def extract_observed_images(num,total_tsteps_list):

	time_varying_images = np.zeros([num,IMAGE_SIZE,IMAGE_SIZE])
	i = 0
	for image_num in xrange(num):
		#figure out which shape control needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[image_num]
		for timestep in xrange(total_tsteps):
			#load the next observed image timestep if the max time step has not been reached yet
			time_varying_images[i,:,:] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + '.png')	
			i+=1	
	return time_varying_images


def load_data(num):
	with open("Joints_to_Image/" + "joint_state_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		joint_state_array = pickle.load(f)[:num,...]

	with open("Joints_to_Image/" + "target_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		target_image_array = pickle.load(f)[:num,...]

	with open("Joints_to_Image/" + "input_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		input_image_array = pickle.load(f)[:num,...]


	return joint_state_array,target_image_array,input_image_array


#first find the number of tsteps for all the image sequences that are going to be loaded
_,total_tstep_list = find_seq_max_length(NUM_SHAPE_SEQUENCES)
#now use the tstep list to extract the images into an array
time_varying_images = extract_observed_images(NUM_SHAPE_SEQUENCES,total_tsteps_list)
#determine how many images there are in the set
num_images = np.shape(time_varying_images)[0]
#now we have to separate these sets of images into pairs with one member of the pair a timestep before the other
x_2_image_array = time_varying_images
x_1_image_array = np.append(time_varying_images[:-1, ...], np.zeros([1,IMAGE_SIZE,IMAGE_SIZE]), axis = 0)
#Determine the number of images in the validation set versus those in the training set
EVAL_SIZE = np.sum(total_tsteps_list[:EVAL_SHAPE_SEQUENCES])
#the train size is given by the sum of the length of the sequences that are not in the training set
TRAIN_SIZE = np.sum(total_tsteps_list[EVAL_SHAPE_SEQUENCES:])
#get the previous time step by appending 
#split this data into a training and validation set
x_2_image_array_train = x_2_image_array[EVAL_SIZE:,...]
x_1_image_array_train = x_1_image_array[EVAL_SIZE:,...]
#now specify the eval set
x_2_image_array_eval = x_2_image_array[:EVAL_SIZE,...]
x_1_image_array_eval = x_1_image_array[:EVAL_SIZE,...]

###########################################DEFINE PARAMETERS FOR MODEL#########################################
observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 20}
joint_encoder_parameters = {"fc_1" : 200 , "fc_2" : 56}
output_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 200}
output_image_decoder_parameters = {"deconv_output_channels_1" : 32, "deconv_output_channels_2" : 16, "deconv_output_channels_3" : 8, "deconv_output_channels_4" : 4,"deconv_output_channels_5" : 1}

####first define an input placeholder, note that the input images will also serve as the output labels so that it is possible to compute

###The first input should be the image at x_t
x_1 = tf.placeholder(tf.float32,shape = [None,64,64,1],name = "x_t1")
#now define a placeholder for the second image
x_2 = tf.placeholder(tf.float32,shape = [None,64,64,1], name = "x_t2")
#now concatenate these two images along the channel dimension 
x_concatenated = tf.concat(3,[x_1,x_2], name = "Concatenated_Images")


def conv(x,weight_shape, scope, stddev = 0.1,trainable = True, reuse_variables = False):
	"""
	x should be the 4d tensor which is being convolved
	weight shape should be a list of the form [Kernel Width, Kernel Width, input channels, output channels]
	scope should be string specifying the scope of the variables in question
	"""

	with tf.variable_scope(scope) as scope:
		if reuse_variables:
			scope.reuse_variables()
		#initialize the weights for the convolutional layer
		W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_conv")
		#initiaize the biases
		b = tf.Variable(tf.constant(0.1,shape = [weight_shape[-1]]), trainable = trainable, name = "b_conv")
		#calculate the output from the convolution 
		conv = tf.nn.conv2d(x,W,strides = [1,2,2,1],padding = "SAME")
		#compute the activations
		h = tf.nn.relu(tf.nn.bias_add(conv,b))

	return h,W,b


def fc_layer(x,weight_shape,scope, stddev = 0.1,trainable = True, reuse_variables = False):
	"""
	Compute the activations of the fc layer
	"""
	with tf.variable_scope(scope) as scope:
		if reuse_variables:
			scope.reuse_variables()
	
		#initialize the weights for the convolutional layer
		W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_fc")
		#initiaize the biases
		b = tf.Variable(tf.constant(0.,shape = [weight_shape[-1]]), trainable = trainable, name = "b_fc")
		#calculate biases
		h = tf.nn.relu(tf.matmul(x,W) + b)

	return h,W,b 

def deconv(x,weight_shape,output_shape,scope,strides = [1,2,2,1], stddev = 0.1,trainable = True, reuse_variables = False,non_linearity = True):
	"""
	generalizable deconv function
	"""
	with tf.variable_scope(scope) as scope:
		if reuse_variables:
			scope.reuse_variables()
		#initialize the weights for the convolutional layer
		W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_deconv")
		#initiaize the biases
		b = tf.Variable(tf.constant(0.1,shape = [weight_shape[-2]]), trainable = trainable, name = "b_deconv")
		#calculate the output from the deconvolution
		deconv = tf.nn.conv2d_transpose(x,W,output_shape,strides = strides)
		#calculate the activations
		if non_linearity:
			h = tf.nn.relu(tf.nn.bias_add(deconv,b))
		else:
			h = tf.nn.bias_add(deconv,b)

	return h,W,b


def encode_previous_output_image(previous_output_image):
	"""
	Takes an input placeholder for an image
	"""

	#expand the dimensionality of the input image
	#x_image = tf.expand_dims(previous_output_image, -1)
	#find the activations of the first conv layer
	h_conv1,W_conv1,b_conv1 = conv(previous_output_image,[3,3,1,observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_output",trainable = False)
	#find the activations of the second conv layer
	h_conv2,W_conv2,b_conv2 = conv(h_conv1,[3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_output",trainable = False)
	#find the activations of the third conv layer
	h_conv3,W_conv3,b_conv3 = conv(h_conv2,[3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_output",trainable = False)
	#find the activations of the second conv layer
	h_conv4,W_conv4,b_conv4 = conv(h_conv3,[3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_output",trainable = False)
	#find the activations of the second conv layer
	h_conv5,W_conv5,b_conv5 = conv(h_conv4,[3,3,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_output",trainable = False)
	#flatten the activations in the final conv layer in order to obtain an output image
	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])
	#pass flattened activations to a fully connected layer
	h_fc1,W_fc1,b_fc1 = fc_layer(h_conv5_reshape,[4*observed_image_encoder_parameters["conv5_kernels"],1024 - 56],"fc_layer_encode_output",trainable = False)
	output_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]

	return h_fc1,output_image_encoder_variable_list 

def encode_joints(x_joints):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	h_fc1,W_fc1,b_fc1 = fc_layer(x_joints,[DOF,joint_encoder_parameters["fc_1"]],"fc_joint_encoder_1",trainable = False)
	#pass the activations to a second fc layer
	h_fc2,W_fc2,b_fc2 = fc_layer(h_fc1,[joint_encoder_parameters["fc_1"], joint_encoder_parameters["fc_2"]],"fc_joint_encoder_2",trainable = False)
	joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]

	return h_fc2,joint_encoder_variable_list


def decode_outputs(hidden_vector):
	"""
	Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
	and generate an image of size [None,64,64,1], do this via 
	"""	
	#find the batch size of the input data in order to use later
	batch_size = tf.shape(hidden_vector)[0]
	#reshape the hidden activation vector into a 4d image that can be deconvolved to form an image
	hidden_image = tf.reshape(hidden_vector, shape = [batch_size,4,4,64])
	#calculate activations for the first deconv layer
	h_deconv1,W_deconv1,b_deconv1 = deconv(hidden_image,[2,2,output_image_decoder_parameters['deconv_output_channels_1'],64],[batch_size,4,4,output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1], trainable = False)
	#calculate activations for second deconv layer
	h_deconv2,W_deconv2,b_deconv2 = deconv(h_deconv1,[3,3,output_image_decoder_parameters['deconv_output_channels_2'],output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2", trainable = False)
	#calculate activations for third deconv layer
	h_deconv3,W_deconv3,b_deconv3 = deconv(h_deconv2,[3,3,output_image_decoder_parameters['deconv_output_channels_3'],output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3", trainable = False)
	#calculate activations for fourth deconv layer
	h_deconv4,W_deconv4,b_deconv4 = deconv(h_deconv3,[3,3,output_image_decoder_parameters['deconv_output_channels_4'],output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4", trainable = False)
	#calculate activations for fifth deconv layer
	h_deconv5,W_deconv5,b_deconv5 = deconv(h_deconv4,[3,3,output_image_decoder_parameters['deconv_output_channels_5'],output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5",non_linearity = False, trainable = False)
	decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

	return tf.squeeze(h_deconv5),decoder_variable_list



def input_image_to_joint_angle(x):
	"""
	Take in the two channel image with the first channel corresponding to the observed image at the first timestep and the second channel corresponding to the image at the second timestep
	"""
	h_conv1,W_conv1,b_conv1 = conv(x,[3,3,2,observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_input")
	#find the activations of the second conv layer
	h_conv2,W_conv2,b_conv2 = conv(h_conv1,[3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_input")
	#find the activations of the third conv layer
	h_conv3,W_conv3,b_conv3 = conv(h_conv2,[3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_input")
	#find the activations of the second conv layer
	h_conv4,W_conv4,b_conv4 = conv(h_conv3,[3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_input")
	#find the activations of the second conv layer
	h_conv5,W_conv5,b_conv5 = conv(h_conv4,[3,3,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_input")
	#flatten the activations in the final conv layer in order to obtain an output image
	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])
	#pass flattened activations to a fully connected layer
	h_fc1,W_fc1,b_fc1 = fc_layer(h_conv5_reshape,[4*observed_image_encoder_parameters["conv5_kernels"],DOF],"fc_layer_encode_input_image")
	input_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
	return h_fc1,input_image_encoder_variable_list




def jointangle2image(joint_angle,previous_image):
	"""
	Calls on the respective decoder and encoders in order to map a joint angle state to an output image joint_angle and previous image are both tensors
	"""
	encoded_joint_angle,joint_encoder_variable_list = encode_joints(joint_angle)
	previous_image_encoded,image_encode_variable_list = encode_previous_output_image(previous_image)
	#now concatenate to obtain encoded vector
	encoded_vector = tf.concat(1,[encoded_joint_angle,previous_image_encoded])
	#pass to a decoder in order to get the output
	y_before_sigmoid,decoder_variable_list = decode_outputs(encoded_vector)
	return y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list


#compute the joint angle state from the input images
joint_angle_state,input_image_encoder_variable_list = input_image_to_joint_angle(x_concatenated)
#now feed this joint angle to the jointangle2image mapping this is necesary in order to compute the loss in pixel space
y_before_sigmoid,joint_encoder_variable_list,observed_image_encoder_variable_list,decoder_variable_list = jointangle2image(joint_angle_state,x_1)
#take the sigmoid of the output to ensure that the output lies between 0 and 1
y = tf.nn.sigmoid(y_before_sigmoid)
#compute the sigmoid cross entropy between the output activation tensor and the next observed image, hence penalizing outputs deviate from the image at the next step
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.expand_dims(y_before_sigmoid,-1),x_2))
#define an operation for the optimizer
opt = tf.train.AdamOptimizer(learning_rate)
#define an operation to initialize variables
#init_op = tf.initialize_all_variables()
#Add ops to restore all the variables in the second network which is 
saver = tf.train.Saver(observed_image_encoder_variable_list + joint_encoder_variable_list + decoder_variable_list)
#compute the gradients for all variables
grads_and_vars = opt.compute_gradients(loss,input_image_encoder_variable_list + joint_encoder_variable_list + observed_image_encoder_variable_list + decoder_variable_list)
#define a training operation which applies the gradient updates inorder to tune the parameters of the graph
train_op = opt.apply_gradients(grads_and_vars)
init_op = tf.initialize_all_variables()
#apply histogram summary nodes for the gradients of all the variables
gradient_summary_nodes = [tf.histogram_summary(str(gv[1].name) + "_gradient",gv[0]) for gv in grads_and_vars]
#apply histogram summary nodes to the values of all variables
var_summary_nodes = [tf.histogram_summary(str(gv[1].name),gv[1]) for gv in grads_and_vars]
#Also record the loss
tf.scalar_summary("loss",loss)
#save images
tf.image_summary("Reconstructed Output", tf.expand_dims(y,-1))
tf.image_summary("Target Image", x_2)
#now merge all the summary nodes
merged = tf.merge_all_summaries()


#####NOW THAT ALL OPERATIONS HAVE BEEN DEFINED WRITE A FUNCTION THAT INITIALIZES GRAPH AND TRAINS IT ON THE TRAINING DATA SET##############################################
def train_graph():
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		average_test_loss = []

		with tf.Session(config = config) as sess:
			#initialize the variables
			log_dir = ROOT_DIR + SUMMARY_DIR
			train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

			sess.run(init_op)
			saver.restore(sess, model_dir)
  			print("Model restored.")
			#initialize a training loss array
			training_loss_array = [0] * (int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE)
			for step in xrange(int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (TRAIN_SIZE)
				x_1_image_batch = x_1_image_array_train[offset:(offset + BATCH_SIZE),...]
				x_2_image_batch = x_2_image_array_train[offset:(offset + BATCH_SIZE),...]
				
				#construct a feed dictionary in order to run the model
				feed_dict = {x_1 : np.expand_dims(x_1_image_batch,axis=-1), x_2 : np.expand_dims(x_2_image_batch,axis = -1)}

				#run the graph
				_, l,merged_summary= sess.run(
					[train_op,loss,merged],
					feed_dict=feed_dict)
				
				training_loss_array[step] = l

				if step % 5 == 0:
					train_writer.add_summary(merged_summary,step)
					print step,l			

			predictions,test_loss_array = eval_in_batches(sess)
		return predictions,training_loss_array,test_loss_array


def eval_in_batches(sess):
	"""Get combined loss for dataset by running in batches"""
	size = EVAL_SIZE

	if size < EVAL_BATCH_SIZE:
		raise ValueError("batch size for evals larger than dataset: %d" % size)

	predictions = np.ndarray(shape = (size,IMAGE_SIZE,IMAGE_SIZE), dtype = np.float32)
	test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
	i = 0
	for begin in xrange(0,size,EVAL_BATCH_SIZE):
		end = begin + EVAL_BATCH_SIZE
		
		if end <= size:
			predictions[begin:end, ...],l = sess.run([y,loss],feed_dict={x_1 : np.expand_dims(x_1_image_array_eval[begin:end, ...], axis = -1), x_2 : np.expand_dims(x_2_image_array_eval[begin:end, ...],axis = -1)})
		else:
			batch_prediction,l = sess.run([y,loss],feed_dict={x_1 : np.expand_dims(x_1_image_array_eval[-EVAL_BATCH_SIZE:, ...], axis = -1), x_2 : np.expand_dims(x_2_image_array_eval[-EVAL_BATCH_SIZE:, ...], axis = -1)})
			predictions[begin:, ...] = batch_prediction[-(size - begin):,...]

		test_loss_array[i] = l
		i += 1
	return predictions,test_loss_array

def plot_predictions(predictions,total_tsteps_list):
	"""
	Takes in the predictions and the total number of tsteps and computes the 
	"""
	num_predicted_images = np.shape(predictions)[0]

	#take the predictions
	for image_num in xrange(num_predicted_images):
		#figure out which shape control needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[image_num]
		#create the summary directory if it does not exist
		if not(os.path.exists(OUTPUT_DIR + shape_str_array[shape_name_index] + str(shape_index) + "/")):
			os.makedirs(OUTPUT_DIR + shape_str_array[shape_name_index] + str(shape_index) + "/"ROOT_DIR + SUMMARY_DIR)
		for timestep in xrange(total_tsteps):
			#load the next observed image timestep if the max time step has not been reached yet
			plt.imsave(OUTPUT_DIR + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + "rec" + '.png', cmap = 'Greys_r')



predictions,training_loss_array,test_loss_array = train_graph()
plot_predictions(predictions)