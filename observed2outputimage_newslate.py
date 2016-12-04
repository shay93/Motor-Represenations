

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
NUM_SHAPE_SEQUENCES = 1000
EVAL_SHAPE_SEQUENCES = 6
TRAIN_SIZE = NUM_SHAPE_SEQUENCES - EVAL_SHAPE_SEQUENCES
EVAL_SIZE = EVAL_SHAPE_SEQUENCES
IMAGE_SIZE = 64
BATCH_SIZE = 500
learning_rate = 1e-3
EVAL_BATCH_SIZE = 6
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


def extract_observed_images(shape_sequence_num,total_tsteps_list,max_seq_length):

	x_1 = np.ndarray(shape = (shape_sequence_num,IMAGE_SIZE,IMAGE_SIZE,max_seq_length), dtype = np.float32)
	x_2 = np.ndarray(shape = (shape_sequence_num,IMAGE_SIZE,IMAGE_SIZE,max_seq_length), dtype = np.float32)
	for shape_sequence_index in xrange(shape_sequence_num):
		#figure out which shape control needs to be loaded
		shape_name_index = shape_sequence_index % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = shape_sequence_index // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[shape_sequence_index]
		for timestep in xrange(max_seq_length):
			if timestep < total_tsteps:
				#load the next observed image timestep if the max time step has not been reached yet
				x_2[shape_sequence_index,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + '.png')
			else:
				#if the max time step has been reached i.e. the complete shape drawing has been observed then continue loading the last image for the remaining timesteps
				x_2[shape_sequence_index,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(total_tsteps - 1) + '.png')	
		
		#now get x_1_temp
		x_1[shape_sequence_index,...] = np.concatenate((np.zeros((IMAGE_SIZE,IMAGE_SIZE,1)),x_2[shape_sequence_index,:,:,:SEQ_MAX_LENGTH - 1]),axis = 2)
	return x_1,x_2


def get_binary_loss(total_tsteps_list):
	"""
	use the tstep list to get a numpy array of 1s and 0s to zero out the loss as needed
	"""
	binary_loss = np.ones((len(total_tsteps_list),SEQ_MAX_LENGTH),dtype = np.float32)
	#for i,max_tstep in enumerate(total_tsteps_list):
		#binary_loss[i,:max_tstep- 4] = np.ones(max_tstep - 4,dtype = np.float32)
	return binary_loss

SEQ_MAX_LENGTH,total_tsteps_list = find_seq_max_length(NUM_SHAPE_SEQUENCES)
print "Sequence Max Length is ",SEQ_MAX_LENGTH
SEQ_MAX_LENGTH = 2
x_1_array,x_2_array = extract_observed_images(NUM_SHAPE_SEQUENCES,total_tsteps_list,SEQ_MAX_LENGTH)
binary_loss_array = get_binary_loss(total_tsteps_list)
#get the previous time step by appending 
#split this data into a training and validation set
x_2_image_array_train = x_2_array[EVAL_SHAPE_SEQUENCES:,...]
x_1_image_array_train = x_1_array[EVAL_SHAPE_SEQUENCES:,...]
binary_loss_array_train = binary_loss_array[EVAL_SHAPE_SEQUENCES:,...]
#now specify the eval set
x_2_image_array_eval = x_2_array[:EVAL_SHAPE_SEQUENCES,...]
x_1_image_array_eval = x_1_array[:EVAL_SHAPE_SEQUENCES,...]
binary_loss_array_eval = binary_loss_array[:EVAL_SHAPE_SEQUENCES,...]


###########################################DEFINE PARAMETERS FOR MODEL#########################################
observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 20}
joint_encoder_parameters = {"fc_1" : 200 , "fc_2" : 56}
output_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 200}
output_image_decoder_parameters = {"deconv_output_channels_1" : 32, "deconv_output_channels_2" : 16, "deconv_output_channels_3" : 8, "deconv_output_channels_4" : 4,"deconv_output_channels_5" : 1}

####first define an input placeholder, note that the input images will also serve as the output labels so that it is possible to compute

###The first input should be the image at x_t
binary_loss_tensor = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH])
x_1_sequence = tf.placeholder(tf.float32,shape = [None,64,64,SEQ_MAX_LENGTH],name = "x_t1_sequence_tensor")
#now define a placeholder for the second image
x_2_sequence = tf.placeholder(tf.float32,shape = [None,64,64,SEQ_MAX_LENGTH], name = "x_t2_sequence_tensor")
#split these two sequences into lists so that they can be fed to the model sequentially
x_1_list = tf.split(3,SEQ_MAX_LENGTH,x_1_sequence,name = "x_t1_list")
x_2_list = tf.split(3,SEQ_MAX_LENGTH,x_2_sequence,name = "x_t2_list")
#intialize a list to store the concatenated tensors
x_concatenated_list = []
#now loop through the list and concatenate each member of the sequence along the 3rd dimension
for tstep in xrange(SEQ_MAX_LENGTH):
	#now concatenate these two images along the channel dimension 
	x_concatenated_list.append(tf.concat(3,[x_1_list[tstep],x_2_list[tstep]], name = "Concatenated_Images" + "_" + str(tstep)))


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


def encode_previous_output_image(previous_output_image, reuse_variables = False):
	"""
	Takes an input placeholder for an image
	"""

	#expand the dimensionality of the input image
	#x_image = tf.expand_dims(previous_output_image, -1)
	#find the activations of the first conv layer
	h_conv1,W_conv1,b_conv1 = conv(previous_output_image,[3,3,1,observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_output",trainable = False, reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv2,W_conv2,b_conv2 = conv(h_conv1,[3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_output",trainable = False, reuse_variables = reuse_variables)
	#find the activations of the third conv layer
	h_conv3,W_conv3,b_conv3 = conv(h_conv2,[3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_output",trainable = False, reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv4,W_conv4,b_conv4 = conv(h_conv3,[3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_output",trainable = False, reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv5,W_conv5,b_conv5 = conv(h_conv4,[3,3,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_output",trainable = False, reuse_variables = reuse_variables)
	#flatten the activations in the final conv layer in order to obtain an output image
	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])
	#pass flattened activations to a fully connected layer
	h_fc1,W_fc1,b_fc1 = fc_layer(h_conv5_reshape,[4*observed_image_encoder_parameters["conv5_kernels"],1024 - 56],"fc_layer_encode_output",trainable = False, reuse_variables = reuse_variables)
	output_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]

	return h_fc1,output_image_encoder_variable_list 

def encode_joints(x_joints, reuse_variables = False):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	h_fc1,W_fc1,b_fc1 = fc_layer(x_joints,[DOF,joint_encoder_parameters["fc_1"]],"fc_joint_encoder_1",trainable = False, reuse_variables = reuse_variables)
	#pass the activations to a second fc layer
	h_fc2,W_fc2,b_fc2 = fc_layer(h_fc1,[joint_encoder_parameters["fc_1"], joint_encoder_parameters["fc_2"]],"fc_joint_encoder_2",trainable = False, reuse_variables = reuse_variables)
	joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]

	return h_fc2,joint_encoder_variable_list


def decode_outputs(hidden_vector, reuse_variables = False):
	"""
	Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
	and generate an image of size [None,64,64,1], do this via 
	"""	
	#find the batch size of the input data in order to use later
	batch_size = tf.shape(hidden_vector)[0]
	#reshape the hidden activation vector into a 4d image that can be deconvolved to form an image
	hidden_image = tf.reshape(hidden_vector, shape = [batch_size,4,4,64])
	#calculate activations for the first deconv layer
	h_deconv1,W_deconv1,b_deconv1 = deconv(hidden_image,[2,2,output_image_decoder_parameters['deconv_output_channels_1'],64],[batch_size,4,4,output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1], trainable = False, reuse_variables = reuse_variables)
	#calculate activations for second deconv layer
	h_deconv2,W_deconv2,b_deconv2 = deconv(h_deconv1,[3,3,output_image_decoder_parameters['deconv_output_channels_2'],output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2", trainable = False, reuse_variables = reuse_variables)
	#calculate activations for third deconv layer
	h_deconv3,W_deconv3,b_deconv3 = deconv(h_deconv2,[3,3,output_image_decoder_parameters['deconv_output_channels_3'],output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3", trainable = False, reuse_variables = reuse_variables)
	#calculate activations for fourth deconv layer
	h_deconv4,W_deconv4,b_deconv4 = deconv(h_deconv3,[3,3,output_image_decoder_parameters['deconv_output_channels_4'],output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4", trainable = False, reuse_variables = reuse_variables)
	#calculate activations for fifth deconv layer
	h_deconv5,W_deconv5,b_deconv5 = deconv(h_deconv4,[3,3,output_image_decoder_parameters['deconv_output_channels_5'],output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5",non_linearity = False, trainable = False, reuse_variables = reuse_variables)
	decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

	return h_deconv5,decoder_variable_list



def input_image_to_joint_angle(x, reuse_variables = False):
	"""
	Take in the two channel image with the first channel corresponding to the observed image at the first timestep and the second channel corresponding to the image at the second timestep
	"""
	h_conv1,W_conv1,b_conv1 = conv(x,[3,3,2,observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_input", reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv2,W_conv2,b_conv2 = conv(h_conv1,[3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_input", reuse_variables = reuse_variables)
	#find the activations of the third conv layer
	h_conv3,W_conv3,b_conv3 = conv(h_conv2,[3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_input", reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv4,W_conv4,b_conv4 = conv(h_conv3,[3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_input", reuse_variables = reuse_variables)
	#find the activations of the second conv layer
	h_conv5,W_conv5,b_conv5 = conv(h_conv4,[3,3,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_input", reuse_variables = reuse_variables)
	#flatten the activations in the final conv layer in order to obtain an output image
	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])
	#pass flattened activations to a fully connected layer
	h_fc1,W_fc1,b_fc1 = fc_layer(h_conv5_reshape,[4*observed_image_encoder_parameters["conv5_kernels"],DOF],"fc_layer_encode_input_image",reuse_variables = reuse_variables)
	input_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
	return h_fc1,input_image_encoder_variable_list




def jointangle2image(joint_angle,previous_image,reuse_variables = False):
	"""
	Calls on the respective decoder and encoders in order to map a joint angle state to an output image joint_angle and previous image are both tensors
	"""
	encoded_joint_angle,joint_encoder_variable_list = encode_joints(joint_angle, reuse_variables = reuse_variables)
	previous_image_encoded,image_encode_variable_list = encode_previous_output_image(previous_image, reuse_variables = reuse_variables)
	#now concatenate to obtain encoded vector
	encoded_vector = tf.concat(1,[encoded_joint_angle,previous_image_encoded])
	#pass to a decoder in order to get the output
	y_before_sigmoid,decoder_variable_list = decode_outputs(encoded_vector, reuse_variables = reuse_variables)
	return y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list

#get the batch size
batch_size_tensor = tf.shape(x_concatenated_list[0])[0]
#initialize a list to hold the previous output image
previous_output_image_list = [tf.zeros([batch_size_tensor,IMAGE_SIZE,IMAGE_SIZE,1])]
#initialize a list to store the joint angle tensor at each step
joint_angle_state_list = []
#initialize another list to store the output image at the current tstep
current_output_image_list = []
#initialize a list to record the loss per tstep
loss_per_tstep_list = []
####NOW COMPUTE THE OUTPUT THE FIRST TIMESTEP
#compute the joint angle state from the input images
joint_angle_state,input_image_encoder_variable_list = input_image_to_joint_angle(x_concatenated_list[0])
#now feed this joint angle to the jointangle2image mapping this is necesary in order to compute the loss in pixel space
y_before_sigmoid,joint_encoder_variable_list,observed_image_encoder_variable_list,decoder_variable_list = jointangle2image(joint_angle_state,previous_output_image_list[0])
print joint_angle_state
#append the joint angle state for that tstep to the joint_angle_state_list
cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_before_sigmoid,x_2_list[0])
loss_per_tstep_list.append(cross_entropy_loss)
joint_angle_state_list.append(joint_angle_state)
current_output_image_list.append(y_before_sigmoid)
previous_output_image_list.append(tf.nn.sigmoid(y_before_sigmoid))
#now perform the same but for a sequence images
for tstep in xrange(1,SEQ_MAX_LENGTH):
	#infer the joint angle
	joint_angle_state,_ = input_image_to_joint_angle(x_concatenated_list[tstep], reuse_variables = True)
	#append the joint angle op to the list
	joint_angle_state_list.append(joint_angle_state)
	#now pass this inferred joint angle and the previous output image to the second network that produces the output image
	y_before_sigmoid,_,_,_= jointangle2image(joint_angle_state,previous_output_image_list[tstep])
	#append this output image to the output image list
	previous_output_image_list.append(tf.nn.sigmoid(y_before_sigmoid))
	current_output_image_list.append(y_before_sigmoid)
	cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_before_sigmoid,x_2_list[tstep])
	loss_per_tstep_list.append(cross_entropy_loss)

#pack together the loss into a tensor of size [None,MAX SEQ LENGTH]
print "previous_image_list",previous_image_list
loss_tensor = tf.pack(loss_per_tstep_list,axis = -1)
print "loss_tensor",loss_tensor
#now multiply this by the binary loss tensor to zero out those timesteps in the sequence which you do not want to account for
loss = tf.reduce_mean(loss_per_tstep_list)
#now pack together the output image list
output_image_tensor_before_sigmoid = tf.concat(3,current_output_image_list)
#compute the sigmoid cross entropy between the output activation tensor and the next observed image, hence penalizing outputs deviate from the image at the next step
y = tf.nn.sigmoid(output_image_tensor_before_sigmoid)
#define an operation for the optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#define an operation to initialize variables
#init_op = tf.initialize_all_variables()
#Add ops to restore all the variables in the second network which is 
saver = tf.train.Saver(observed_image_encoder_variable_list + joint_encoder_variable_list + decoder_variable_list)
#compute the gradients for all variables
#grads_and_vars = opt.compute_gradients(loss,input_image_encoder_variable_list + joint_encoder_variable_list + observed_image_encoder_variable_list + decoder_variable_list)
#define a training operation which applies the gradient updates inorder to tune the parameters of the graph
#train_op = opt.apply_gradients(grads_and_vars)
init_op = tf.initialize_all_variables()
#apply histogram summary nodes for the gradients of all the variables
#gradient_summary_nodes = [tf.histogram_summary(str(gv[1].name) + "_gradient",gv[0]) for gv in grads_and_vars]
#apply histogram summary nodes to the values of all variables
#var_summary_nodes = [tf.histogram_summary(str(gv[1].name),gv[1]) for gv in grads_and_vars]
#Also record the loss
tf.scalar_summary("loss",loss)
#save images
#tf.image_summary("Reconstructed Output", tf.expand_dims(y,-1))
#tf.image_summary("Target Image", x_2)
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
				binary_loss_array_batch = binary_loss_array_train[offset:(offset + BATCH_SIZE),...]
				#construct a feed dictionary in order to run the model
				feed_dict = {x_1_sequence : x_1_image_batch, x_2_sequence : x_2_image_batch, binary_loss_tensor : binary_loss_array_batch}

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
			predictions[begin:end, ...],l = sess.run([y,loss],feed_dict={x_1_sequence : x_1_image_array_eval[begin:end, ...], x_2_sequence : x_2_image_array_eval[begin:end, ...], binary_loss_tensor: binary_loss_array_eval[begin:end, ...]})
		else:
			batch_prediction,l = sess.run([y,loss],feed_dict={x_1_sequence : x_1_image_array_eval[-EVAL_BATCH_SIZE:, ...], x_2_sequence: x_2_image_array_eval[-EVAL_BATCH_SIZE:, ...],binary_loss_tensor: binary_loss_array_eval[-EVAL_BATCH_SIZE:,...]})
			predictions[begin:, ...] = batch_prediction[-(size - begin):,...]

		test_loss_array[i] = l
		i += 1
	return predictions,test_loss_array

def save_output_images(predictions):
	"""
	Save the output shapes to the output root directory
	"""
	prediction_size = np.shape(predictions)[0]
	#multiply predictions by scalar 255 so that they can be sved as grey map images
	predictions = predictions * 255
	#first construct the output root directory if it does not exist
	if not(os.path.exists(ROOT_DIR)):
		os.makedirs(ROOT_DIR)
	for output_image_num in xrange(prediction_size):
		#Get the shape name and index number in order to save correctly
		shape_name_index = output_image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = output_image_num // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[output_image_num]
		shape_name = shape_str_array[shape_name_index]
		shape_dir = OUTPUT_DIR + shape_str_array[shape_name_index] + str(shape_index) + "/"
		#create this directory if it doesnt exist
		if not(os.path.exists(shape_dir)):
			os.makedirs(shape_dir)
		for tstep in xrange(total_tsteps):
				plt.imsave(shape_dir + shape_name + str(shape_index) + '_' + str(tstep),predictions[output_image_num,:,:,tstep],cmap = "Greys_r")



predictions,training_loss_array,test_loss_array = train_graph()
save_output_images(predictions)
