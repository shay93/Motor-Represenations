
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import training_tools as tt
import os


BATCH_SIZE = 50
EVAL_BATCH_SIZE = 50
IMAGE_SIZE = 64
VALIDATION_SIZE = 250
EPOCHS = 2
ROOT_DIR = "Baseline_Seq2Seq_Outputs/"
EVAL_FREQUENCY = 20
NUM_SAMPLES = 200
EVAL_SIZE = 50
TRAIN_SIZE = NUM_SAMPLES - EVAL_SIZE
DISPLAY_SIZE = 20
SUMMARY_DIR = "/tmp/summary_logs"

shape_str_array = ['Rectangle', 'Square', 'Triangle']

NORMALIZATION_FACTOR = 1.

#MODEL GLOBALS
output_arm_dof = 3
OUTPUT_FEATURES = 3


LEARNING_RATE = 1e-3

###########################################DEFINE PARAMETERS FOR MODEL#########################################
observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 200}
joint_angle_decoder_parameters = {"lstm_hidden_units": 200,"output_features" : output_arm_dof,"layers" : 3}
joint_encoder_parameters = {"fc_1" : 200 , "fc_2" : 56}
output_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 200}
output_image_decoder_parameters = {"deconv_output_channels_1" : 32, "deconv_output_channels_2" : 16, "deconv_output_channels_3" : 8, "deconv_output_channels_4" : 4}

########################################## EXTRACT DATA #######################################################################

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


def extract_observed_images(num,total_tsteps_list,max_seq_length):

	time_varying_images = np.zeros([num,IMAGE_SIZE,IMAGE_SIZE,max_seq_length])
	
	for image_num in xrange(num):
		#figure out which shape control needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		total_tsteps = 	total_tsteps_list[image_num]
		for timestep in xrange(max_seq_length):
			if timestep < total_tsteps:
				#load the next observed image timestep if the max time step has not been reached yet
				time_varying_images[image_num,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(timestep) + '.png')
			else:
				#if the max time step has been reached i.e. the complete shape drawing has been observed then continue loading the last image for the remaining timesteps
				time_varying_images[image_num,:,:,timestep] = plt.imread("Shapes/" + shape_str_array[shape_name_index] + str(shape_index) + "/" + shape_str_array[shape_name_index] + str(shape_index) + "_" + str(total_tsteps - 1) + '.png')	
	
	return time_varying_images


def load_saved_variables():
	"""
	Loads the saved weights and biases used in the joint angle to image mapping
	"""
	with open("Joints_to_Image/learned_variable_list.npy","rb") as f:
		learned_variable_list = pickle.load(f)
	print len(learned_variable_list)

	output_image_encoder_variable_list = learned_variable_list[:12]
	joint_encoder_variable_list = learned_variable_list[12:16]
	decoder_variable_list = learned_variable_list[-10:]
	return output_image_encoder_variable_list,joint_encoder_variable_list,decoder_variable_list


def get_binary_loss(total_tsteps_list):
	"""
	use the tstep list to get a numpy array of 1s and 0s to zero out the loss as needed
	"""
	binary_loss = np.zeros((len(total_tsteps_list),SEQ_MAX_LENGTH))
	for i,max_tstep in enumerate(total_tsteps_list):
		binary_loss[i,:max_tstep] = np.ones(max_tstep,dtype = np.float32)
	return binary_loss

output_image_encoder_variable_list,joint_encoder_variable_list,decoder_variable_list = load_saved_variables()
SEQ_MAX_LENGTH,total_tsteps_list = find_seq_max_length(NUM_SAMPLES)

time_varying_images = extract_observed_images(NUM_SAMPLES,total_tsteps_list,SEQ_MAX_LENGTH)
binary_loss_array = get_binary_loss(total_tsteps_list)

################################ HANDLE DATA BY SEPARATING INTO TRAINING AND TESTING SETS ###############################################
#split this data into a training and validation set
time_varying_images_train = time_varying_images[EVAL_SIZE:,...]
binary_loss_array_train = binary_loss_array[EVAL_SIZE:,...]
#now specify the eval set
time_varying_images_eval = time_varying_images[:EVAL_SIZE,...]
binary_loss_array_eval = binary_loss_array[:EVAL_SIZE,...]

######################BUILD THE TENSORFLOW MODEL############################################

x = tf.placeholder(tf.float32, shape = [None,IMAGE_SIZE,IMAGE_SIZE,SEQ_MAX_LENGTH])
y_ = tf.placeholder(tf.float32, shape = [None,IMAGE_SIZE,IMAGE_SIZE,SEQ_MAX_LENGTH])
binary_loss_tensor = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH])
#split this into individual images with one channel
observed_image_sequence = tf.split(3,SEQ_MAX_LENGTH,x)
target_image_list = tf.split(3,SEQ_MAX_LENGTH,y_)

def observed_image_encoder(observed_image):
	"""
	Encodes the observed image to a vector that may be passed to an lstm in order to obtain the joint angles for the 
	"""
	W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,observed_image_encoder_parameters["conv1_kernels"]],stddev = 0.1))
	b_conv1 = tf.Variable(tf.constant(0.1,shape = [observed_image_encoder_parameters["conv1_kernels"]]))
	conv1 = tf.nn.conv2d(observed_image,W_conv1,strides = [1,2,2,1],padding = 'SAME')
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))	

	#define parameters for the second convolutional layer
	W_conv2 = tf.Variable(tf.truncated_normal([3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],stddev = 0.1))
	b_conv2 = tf.Variable(tf.constant(0.1,shape = [observed_image_encoder_parameters["conv2_kernels"]]))
	conv2 = tf.nn.conv2d(h_conv1,W_conv2,strides = [1,2,2,1],padding = 'SAME')
	h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))

	#define a third convolutional layer
	W_conv3 = tf.Variable(tf.truncated_normal([3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],stddev = 0.1))
	b_conv3 = tf.Variable(tf.constant(0.1,shape = [observed_image_encoder_parameters["conv3_kernels"]]))
	conv3 = tf.nn.conv2d(h_conv2,W_conv3,strides = [1,2,2,1],padding = 'SAME')
	h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3,b_conv3))

	#Add another convolutional layer
	W_conv4 = tf.Variable(tf.truncated_normal([3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]], stddev = 0.1))
	b_conv4 = tf.Variable(tf.constant(0.1,shape = [observed_image_encoder_parameters["conv4_kernels"]]))
	conv4 = tf.nn.conv2d(h_conv3,W_conv4,strides = [1,2,2,1],padding = 'SAME')
	h_conv4 = tf.nn.relu(tf.nn.bias_add(conv4,b_conv4))

	#Add an additonal conv layer
	W_conv5 = tf.Variable(tf.truncated_normal([2,2,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]], stddev = 0.1))
	b_conv5 = tf.Variable(tf.constant(0.1,shape = [observed_image_encoder_parameters["conv5_kernels"]]))
	conv5 = tf.nn.conv2d(h_conv4,W_conv5,strides = [1,2,2,1],padding = 'SAME')
	h_conv5 = tf.nn.relu(tf.nn.bias_add(conv5,b_conv5))	

	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])


	#define parameters for full connected layer
	W_fc1 = tf.Variable(tf.truncated_normal(shape = [4*observed_image_encoder_parameters["conv5_kernels"],observed_image_encoder_parameters["fc_1"]],stddev = 0.1)) 
	b_fc1 = tf.Variable(tf.constant(0.,shape = [observed_image_encoder_parameters["fc_1"]])) 
	h_fc1 = tf.nn.relu(tf.matmul(h_conv5_reshape, W_fc1) + b_fc1)

	#create a list of all the variables in order to compute their gradients
	observed_image_encoder_variables = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
	return h_fc1, observed_image_encoder_variables

def joint_angle_decoder(encoded_observed_image_list):
	"""
	Takes in a list of the encoded observed image and returns a sequence of joint angle states
	"""
	lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(joint_angle_decoder_parameters["lstm_hidden_units"],state_is_tuple = True)] * joint_angle_decoder_parameters["layers"]
	#pass this list to construct a multirnn_cell
	lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list, state_is_tuple = True)
	#use the in built rnn function to get the output sequence from the lstm timesteps
	outputs,states = tf.nn.rnn(lstm_cell,encoded_observed_image_list,dtype = tf.float32)
	#initialize the weights and biases that will be used to generate the output joint angle sequence
	W_fc_jointangle = tf.Variable(tf.truncated_normal(shape = [joint_angle_decoder_parameters["lstm_hidden_units"],joint_angle_decoder_parameters["output_features"]],stddev = 0.1))
	b_fc_jointangle = tf.Variable(tf.constant(0., shape = [joint_angle_decoder_parameters["output_features"]]))
	#while samples in the batch remain and while timesteps in the sample remain compute the output and append zeros to the output
	jointangle_list = []
	for output_timestep in outputs:
		jointangle_timestep = tf.nn.elu(tf.matmul(output_timestep,W_fc_jointangle) + b_fc_jointangle)
		jointangle_list.append(jointangle_timestep)

	return jointangle_list



def encode_previous_output_image(previous_output_image,output_image_encoder_variable_list):
	"""
	Takes an input placeholder for an image
	"""
	W_conv1_np,W_conv2_np,W_conv3_np,W_conv4_np,W_conv5_np,b_conv1_np,b_conv2_np,b_conv3_np,b_conv4_np,b_conv5_np,W_fc1_np,b_fc1_np = output_image_encoder_variable_list
	#define a place holder for the outputs
	x_image = tf.expand_dims(previous_output_image, -1)
	W_conv1 = tf.constant(W_conv1_np)
	b_conv1 = tf.constant(b_conv1_np)
	conv1 = tf.nn.conv2d(x_image,W_conv1,strides = [1,2,2,1],padding = 'SAME')
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
		
	
	#define parameters for the second convolutional layer
	W_conv2 = tf.constant(W_conv2_np)
	b_conv2 = tf.constant(b_conv2_np)
	conv2 = tf.nn.conv2d(h_conv1,W_conv2,strides = [1,2,2,1],padding = 'SAME')
	h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))

	#define a third convolutional layer
	W_conv3 = tf.constant(W_conv3_np)
	b_conv3 = tf.constant(b_conv3_np)
	conv3 = tf.nn.conv2d(h_conv2,W_conv3,strides = [1,2,2,1],padding = 'SAME')
	h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3,b_conv3))

	#Add another convolutional layer
	W_conv4 = tf.constant(W_conv4_np)
	b_conv4 = tf.constant(b_conv4_np)
	conv4 = tf.nn.conv2d(h_conv3,W_conv4,strides = [1,2,2,1],padding = 'SAME')
	h_conv4 = tf.nn.relu(tf.nn.bias_add(conv4,b_conv4))

	#Add an additonal conv layer
	W_conv5 = tf.constant(W_conv5_np)
	b_conv5 = tf.constant(b_conv5_np)
	conv5 = tf.nn.conv2d(h_conv4,W_conv5,strides = [1,2,2,1],padding = 'SAME')
	h_conv5 = tf.nn.relu(tf.nn.bias_add(conv5,b_conv5))

	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*4])

	
	#define parameters for full connected layer
	W_fc1 = tf.constant(W_fc1_np) 
	b_fc1 = tf.constant(b_fc1_np) 
	h_fc1 = tf.nn.relu(tf.matmul(h_conv5_reshape, W_fc1) + b_fc1)

	return h_fc1

def encode_joints(x_joints, joint_encoder_variable_list):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	W_fc1_np,b_fc1_np,W_fc2_np,b_fc2_np = joint_encoder_variable_list
	#define a fully connected layer
	W_fc1 = tf.constant(W_fc1_np)
	b_fc1 = tf.constant(b_fc1_np)
	h_fc1 = tf.nn.relu(tf.matmul(x_joints,W_fc1) + b_fc1)
	#now pass through second fully connected layer
	W_fc2 = tf.constant(W_fc2_np)
	b_fc2 = tf.constant(b_fc2_np)
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
	
	return h_fc2


def decode_outputs(hidden_vector,decoder_variable_list):
	"""
	Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
	and generate an image of size [None,64,64,1], do this via 
	"""	
	#Assume FC_UNITS_JOINTS + FC_UNITS_IMAGE is 256
	#then reshape tensor from 2d to 4d to be compatible with deconvoh_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))lution
	W_deconv1_np,W_deconv2_np,W_deconv3_np,W_deconv4_np,W_deconv5_np,b_deconv1_np,b_deconv2_np,b_deconv3_np,b_deconv4_np,b_deconv5_np = decoder_variable_list
	batch_size = tf.shape(hidden_vector)[0]
	hidden_image = tf.reshape(hidden_vector, shape = [batch_size,4,4,64])
	
	W_deconv1 = tf.constant(W_deconv1_np)
	b_deconv1 = tf.constant(b_deconv1_np)
	deconv1 = tf.nn.conv2d_transpose(hidden_image,W_deconv1,[batch_size,4,4,output_image_decoder_parameters['deconv_output_channels_1']],[1,1,1,1])
	h_deconv1 = tf.nn.relu(tf.nn.bias_add(deconv1,b_deconv1))

	W_deconv2 = tf.constant(W_deconv2_np)
	b_deconv2 = tf.constant(b_deconv2_np)
	deconv2 = tf.nn.conv2d_transpose(h_deconv1,W_deconv2,[batch_size,8,8,output_image_decoder_parameters['deconv_output_channels_2']],[1,2,2,1])
	h_deconv2 = tf.nn.relu(tf.nn.bias_add(deconv2,b_deconv2))

	W_deconv3 = tf.constant(W_deconv3_np)
	b_deconv3 = tf.constant(b_deconv3_np)
	deconv3 = tf.nn.conv2d_transpose(h_deconv2,W_deconv3,[batch_size,16,16,output_image_decoder_parameters['deconv_output_channels_3']],[1,2,2,1])
	h_deconv3 = tf.nn.relu(tf.nn.bias_add(deconv3,b_deconv3))

	W_deconv4 = tf.constant(W_deconv4_np)
	b_deconv4 = tf.constant(b_deconv4_np)
	deconv4 = tf.nn.conv2d_transpose(h_deconv3,W_deconv4,[batch_size,32,32,output_image_decoder_parameters['deconv_output_channels_4']],[1,2,2,1])
	h_deconv4 = tf.nn.relu(tf.nn.bias_add(deconv4,b_deconv4))

	W_deconv5 = tf.constant(W_deconv5_np)
	b_deconv5 = tf.constant(b_deconv5_np)
	deconv5 = tf.nn.conv2d_transpose(h_deconv4,W_deconv5,[batch_size,64,64,1],[1,2,2,1])
	h_deconv5 = tf.nn.bias_add(deconv5,b_deconv5)

	return tf.squeeze(h_deconv5)


def jointangle2image(joint_angle,previous_image):
	"""
	Calls on the respective decoder and encoders in order to map a joint angle state to an output image joint_angle and previous image are both tensors
	"""
	encoded_joint_angle = encode_joints(joint_angle,joint_encoder_variable_list)
	previous_image_encoded = encode_previous_output_image(previous_image,output_image_encoder_variable_list)
	#now concatenate to obtain encoded vector
	encoded_vector = tf.concat(1,[encoded_joint_angle,previous_image_encoded])
	#pass to a decoder in order to get the output
	y_before_sigmoid = decode_outputs(encoded_vector,decoder_variable_list)
	return y_before_sigmoid


#iterate through the observed image sequence and decode them in order to get the encoded_observed_image_sequence
encoded_observed_image_list = []
for observed_image in observed_image_sequence:
	encoded_observed_image, observed_image_encoder_variables = observed_image_encoder(observed_image)
	encoded_observed_image_list.append(encoded_observed_image)

#now decode the encoded observed image into a joint sequence list that may then be fed into the jointangleseq2output image mapping
joint_angle_list = joint_angle_decoder(encoded_observed_image_list) 
#initialize an output image array to record the output image tensor at each timestep
output_image_list = []
#initialize a loss accumulator
image_loss_list = []
#initialize the previous image to be an empty black image
print joint_angle_list[0]

previous_image = tf.zeros([tf.shape(joint_angle_list[0])[0],64,64])
print previous_image
#now loop through the joint angle list at each timestep and pass this to the jointangle2image map to get the output image at each timestep
for i,joint_angle_state in enumerate(joint_angle_list):
	#pass the joint_angle_state to the mapping
	output_image_before_sigmoid = jointangle2image(joint_angle_state,previous_image)
	image_loss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_image_before_sigmoid,target_image_list[i]),[1,2])) #reduce each image in batch to a single value
	output_image_list.append(tf.nn.sigmoid(output_image_before_sigmoid))
	#now set the previous image to be the current output image
	previous_image = tf.nn.sigmoid(output_image_before_sigmoid)

loss_per_image = tf.transpose(tf.pack(image_loss_list))
loss = tf.reduce_sum(tf.reduce_mean(tf.mul(loss_per_image,binary_loss_tensor),[1]))
#use this loss to compute the
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
# opt = tf.train.AdamOptimizer(learning_rate)
# variable_names = ["W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","b_conv1","b_conv2","b_conv3","b_conv4", "b_conv5","W_image_fc1","b_image_fc1","W_joint_fc1","b_joint_fc1","W_joint_fc2","b_joint_fc2","W_deconv1","W_deconv2","W_deconv3","W_deconv4","W_deconv5","b_deconv1","b_deconv2","b_deconv3","b_deconv4","b_deconv5"]
# grads_and_vars = opt.compute_gradients(loss, image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
# summary_nodes = [tf.histogram_summary(variable_names[i],gv[0]) for i,gv in enumerate(grads_and_vars)]
# merged = tf.merge_all_summaries()

######################################################################TRAIN AND EVALUATE MODEL############################################################
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

			sess.run(tf.initialize_all_variables())
			#initialize a training loss array
			training_loss_array = [0] * (int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE)
			for step in xrange(int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (TRAIN_SIZE)
				time_varying_image_batch = time_varying_images_train[offset:(offset + BATCH_SIZE),...]
				binary_loss_batch = binary_loss_array_train[offset:(offset + BATCH_SIZE),...]
				#construct a feed dictionary in order to run the model
				feed_dict = {x : time_varying_image_batch, y_ : time_varying_image_batch, binary_loss_tensor : binary_loss_batch}

				#run the graph
				_, l= sess.run(
					[train_op,loss],
					feed_dict=feed_dict)
				
				training_loss_array[step] = l

				if step % 20 == 0:
					#train_writer.add_summary(merged_summary,step)
					print step,l
				
				# if step % EVAL_FREQUENCY == 0:
				# 	predictions,test_loss_array = eval_in_batches(sess)
				# 	print "Test Loss is " + str(np.mean(test_loss_array))
				# 	average_test_loss.append(np.mean(test_loss_array))
				# 	#also svae the predictions to get
				# 	checkpoint_num = step // EVAL_FREQUENCY
				# 	#use the checkpoint_num to specify the correct directory to save an image
				# 	checkpoint_dir = ROOT_DIR + "Checkpoint" + str(checkpoint_num) + "/"
					
				# 	if not os.path.exists(checkpoint_dir):
				# 		os.makedirs(checkpoint_dir)
					
				# 	for i in range(EVAL_SIZE):
				# 		plt.imsave(checkpoint_dir + "output_image" + str(i) + ".png", predictions[i,...], cmap = "Greys_r")
				# 		plt.imsave(checkpoint_dir + "target_image" + str(i) + ".png", target_image_array_eval[i,...], cmap = "Greys_r")
				# 		plt.imsave(checkpoint_dir + "input_image" + str(i) + ".png", input_image_array_eval[i,...], cmap = "Greys_r")

			predictions,test_loss_array = eval_in_batches(sess)
			#now get the learned variable values and dump to a list
			#variable_list = sess.run(image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
		return predictions,training_loss_array,test_loss_array


def eval_in_batches(sess):
		"""Get combined loss for dataset by running in batches"""
		size = EVAL_SIZE

		if size < EVAL_BATCH_SIZE:
			raise ValueError("batch size for evals larger than dataset: %d" % size)

		predictions = np.ndarray(shape = (size,IMAGE_SIZE,IMAGE_SIZE,SEQ_MAX_LENGTH), dtype = np.float32)
		test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
		i = 0
		for begin in xrange(0,size,EVAL_BATCH_SIZE):
			end = begin + EVAL_BATCH_SIZE
			
			if end <= size:
				predictions[begin:end, ...],l = sess.run([y,loss],feed_dict={x : time_varying_images_eval[begin:end, ...], y_ : time_varying_images_eval[begin:end, ...], binary_loss_tensor : binary_loss_array_eval[begin:end, ...]})
			else:
				batch_prediction,l = sess.run([y,loss],feed_dict={x : time_varying_images_eval[-EVAL_BATCH_SIZE:, ...], y_ : time_varying_images_eval[-EVAL_BATCH_SIZE:, ...], binary_loss_tensor : binary_loss_array_eval[begin:end, ...]})
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
		total_tsteps = 	total_tsteps_list[image_num]
		shape_dir = ROOT_DIR + shape_name + str(shape_number) + "/"
		#create this directory if it doesnt exist
		if not(os.path.exists(shape_dir)):
			os.makedirs(shape_dir)
		for tstep in xrange(total_tsteps):
				plt.imsave(shape_dir + shape_name + str(shape_number) + '_' + str(tstep),predictions[output_image_num,:,:,tstep],cmap = "Greys_r")

def calculate_intersection_over_union(predictions):
	#first construct an array of pixels for the
	threshold_list = np.arange(0,0.9,step = 0.025)
	IoU_list = []
	for i,threshold in enumerate(threshold_list):
		good_mapping_count = 0
		bad_mapping_count = 0
		for i in range(EVAL_SIZE):
			arr_pred = np.nonzero(np.round(predictions[i,...]))
			pos_list_pred = zip(arr_pred[0],arr_pred[1])
			arr_input = np.nonzero(input_image_array_eval[i,...])
			pos_list_input = zip(arr_input[0],arr_input[1])
			intersection = set(pos_list_pred) & set(pos_list_input)
			union = set(pos_list_input + pos_list_pred)
			if (len(intersection) / len(union)) > threshold:
				good_mapping_count += 1
			else:
				bad_mapping_count += 1

		IoU_list.append(good_mapping_count / TRAIN_SIZE)


predictions,training_loss_array,test_loss_array = train_graph()
save_output_images(predictions[:DISPLAY_SIZE,...])