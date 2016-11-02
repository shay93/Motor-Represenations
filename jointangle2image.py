from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
import pickle
import training_tools as tt

#define a max sequence length
DOF = 3
GAUSS_STD = 0.5
link_length = 50
#model globals
CONV_KERNELS_1 = 64
CONV_KERNELS_2 = 32
CONV_KERNELS_3 = 16
DECONV_OUTPUT_CHANNELS_1 = 32
DECONV_OUTPUT_CHANNELS_2 = 16
DECONV_OUTPUT_CHANNELS_3 = 8
DECONV_OUTPUT_CHANNELS_4 = 4
DECONV_KERNELS_1 = 8

FC_UNITS = 100
FC_UNITS_IMAGE = 200
FC_UNITS_JOINTS = 56
#model globals
NUM_SAMPLES = 5000
IMAGE_SIZE = 64
BATCH_SIZE = 200
learning_rate = 1e-3
display_num = 10
EVAL_BATCH_SIZE = 200
EPOCHS = 500
TRAIN_SIZE = 400
ROOT_DIR = "Joints_to_Image/"
EVAL_FREQUENCY = 200
DISPLAY = False


##############################LEGACY############################

def generate_input_image_legacy():
	"""
	Generate the input image for the nn
	"""
	#initialize the input image
	input_image = np.zeros((64,64), dtype = float)
	#now generate a random number to determine the number of points that are illuminated in the input image
	num_points = np.random.randint(5)
	#now generate random joint states equal to the number of points
	random_states = [gen_rand_joint_state(DOF) for j in range(num_points)]
	#initialize a pos list
	pos_list = []
	for joint_state in random_states:
		pos_list += [forward_kinematics(joint_state)]

	additional_points = get_points_to_increase_line_thickness(pos_list)
	additional_points_flattened = [pos for sublist in additional_points for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	for pos in pos_list:
		input_image[pos[0],pos[1]] = 1.0

	for pos in additional_points_flattened:
		input_image[pos[0],pos[1]] = 1.0

	return input_image
		 

def gen_target_image_legacy(pos, input_image):
	"""
	Takes an input pos and generates a target image
	"""
	x = pos[0]
	y = pos[1]
	temp_list = [(x+1,y),(x-1,y),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1),(x,y+1),(x,y-1)]
	#use the position to fill in the specified index with 1
	pos_list = temp_list + [pos]
	for pos in pos_list:
		if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0:
			input_image[pos[0],pos[1]] = 1.0
	return input_image


def generate_training_data_legacy(num,dof):
	joint_state_array = np.zeros((num,dof), dtype = float)
	target_image_array = np.ndarray(shape = (num,64,64))
	input_image_array = np.ndarray(shape = (num,64,64))
	#now loop through the num of examples and populates these arrays
	for i in range(num):
		joint_state_array[i,:] = gen_rand_joint_state(DOF)
		#get end effector position
		pos = forward_kinematics(np.expand_dims(joint_state_array[i,:], axis = 0))
		#get a randomly generated input image
		input_image = generate_input_image_legacy()
		input_image_array[i,...] = input_image
		#use the pos to get the target image and tack on to target_image_array
		target_image_array[i,:] = gen_target_image_legacy(pos,input_image)
	return joint_state_array,target_image_array,input_image_array

###############################################LEGACY END###############################################

def get_points_to_increase_line_thickness(pos_list):
	#initialize a list to store the information needed
	more_pts = [0] * (len(pos_list))
	for i,pos in enumerate(pos_list):
		x,y = pos
		#use these values of x and y to build a temp array with points needed to thicken the original line
		temp_list = [(x+1,y),(x-1,y),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x-1,y-1),(x,y+1),(x,y-1)]
		more_pts[i] = temp_list
	return more_pts

def gen_random_start_point_and_end_point():
	rand_joint_state = gen_rand_joint_state(DOF)
	#use this to get the random end_point
	end_point = forward_kinematics(rand_joint_state)
	start_point = (np.random.randint(63),np.random.randint(63))
	if start_point == end_point or abs(start_point[0] - end_point[0]) < 2 or abs(start_point[1] - end_point[1]) < 2:
		return gen_random_start_point_and_end_point()
	else:
		return start_point,end_point,rand_joint_state

def gen_rand_joint_state(num_dof):
	"""
	Generates random joint state for a specified dof returns a row with each column corresponding to a theta
	"""
	joint_state =  (np.random.rand(1,num_dof) - 0.5)*(2*np.pi)
	pos = forward_kinematics(joint_state)
	if pos[0] > 63 or pos[1] > 63 or pos[0] < 0 or pos[1] < 0:
		return gen_rand_joint_state(num_dof)
	else:
		return joint_state

def forward_kinematics(joint_angle_state):
	"""
	use the joint information to map to a pixel position
	"""
	#initialize the xpos and ypos
	xpos = 0
	ypos = 0
	for i in range(1,DOF+1):
		xpos += round(link_length*np.cos(np.sum(joint_angle_state[0,:i])))
		ypos += round(link_length*np.sin(np.sum(joint_angle_state[0,:i])))
	return (int(xpos),int(ypos))

def generate_input_and_target_image():
	#initialize the input and target image
	input_image = np.zeros((64,64),dtype = float)
	target_image = np.zeros((64,64),dtype = float)
	#first generate a random start point and end point for a line
	start_point,end_point,joint_state = gen_random_start_point_and_end_point()
	#pass these to tt.draw_line to get a pos_list
	sp = tt.shape_maker()
	pos_list = sp.draw_line(start_point,end_point,0.01,0.01)
	#now generate addtional points to thicken image
	additional_points = get_points_to_increase_line_thickness(pos_list)
	#now truncate the last point from the lists so that this could be used as the input joint state map
	pos_list_truncated = pos_list[:-1]
	additional_points_truncated = additional_points[:-1]
	#flatten additional points and remove points which are out of bounds
	additional_points_truncated_flattened = [pos for sublist in additional_points_truncated for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	additional_points_flattened = [pos for sublist in additional_points for pos in sublist if pos[0] < 64 and pos[1] < 64 and pos[0] > 0 and pos[1] > 0]
	#concatenate the lists
	truncated_concatenated_pos_list = pos_list_truncated + additional_points_truncated_flattened
	concatenated_pos_list = pos_list + additional_points_flattened
	#now use these points to construct the input image
	for pos in truncated_concatenated_pos_list:
		input_image[pos[0],pos[1]] = 1.0

	for pos in concatenated_pos_list:
		target_image[pos[0],pos[1]] = 1.0

	#now also find the joint state required to produce this additional point
	return input_image,target_image,joint_state

def generate_training_data(num):
	joint_state_array = np.zeros((num,DOF), dtype = float)
	target_image_array = np.ndarray(shape = (num,64,64))
	input_image_array = np.ndarray(shape = (num,64,64))
	#now loop through the num of examples and populates these arrays
	for i in range(num):
		input_image,target_image,joint_state = generate_input_and_target_image()
		joint_state_array[i,...] = joint_state
		input_image_array[i,...] = input_image
		target_image_array[i,...] = target_image
	return joint_state_array,target_image_array,input_image_array

joint_state_array,target_image_array,input_image_array = generate_training_data(NUM_SAMPLES)
#split this data into a training and validation set
joint_state_array_train = joint_state_array[TRAIN_SIZE:,...]
target_image_array_train = target_image_array[TRAIN_SIZE:,...]
input_image_array_train = input_image_array[TRAIN_SIZE:,...]
#now specify the eval set
joint_state_array_eval = joint_state_array[:TRAIN_SIZE,...]
target_image_array_eval = target_image_array[:TRAIN_SIZE,...]
input_image_array_eval = input_image_array[:TRAIN_SIZE,...]

def encode_input_image(x_image):
	"""
	Takes an input placeholder for an image
	"""
	#define a place holder for the outputs
	x_image = tf.expand_dims(x_image, -1)
	W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,CONV_KERNELS_1],stddev = 0.1))
	b_conv1 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_1]))
	conv1 = tf.nn.conv2d(x_image,W_conv1,strides = [1,2,2,1],padding = 'SAME')
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))
		
	
	#define parameters for the second convolutional layer
	W_conv2 = tf.Variable(tf.truncated_normal([3,3,CONV_KERNELS_1,CONV_KERNELS_2],stddev = 0.1))
	b_conv2 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_2]))
	conv2 = tf.nn.conv2d(h_conv1,W_conv2,strides = [1,2,2,1],padding = 'SAME')
	h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))

	#define a third convolutional layer
	W_conv3 = tf.Variable(tf.truncated_normal([2,2,CONV_KERNELS_2,CONV_KERNELS_3],stddev = 0.1))
	b_conv3 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_3]))
	conv3 = tf.nn.conv2d(h_conv2,W_conv3,strides = [1,2,2,1],padding = 'SAME')
	h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3,b_conv3))

	
	#Reshape the output from pooling layers to pass to fully connected layers
	h_conv3_reshape = tf.reshape(h_conv3, shape = [-1,64*CONV_KERNELS_3])

	
	#define parameters for full connected layer
	W_fc1 = tf.Variable(tf.truncated_normal(shape = [64*CONV_KERNELS_3,FC_UNITS_IMAGE],stddev = 0.1)) 
	b_fc1 = tf.Variable(tf.constant(0.,shape = [FC_UNITS_IMAGE])) 
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_reshape, W_fc1) + b_fc1)

	image_encode_variable_list = [W_conv1,W_conv2,W_conv3,b_conv1,b_conv2,b_conv3,W_fc1,b_fc1]
	return h_fc1,image_encode_variable_list

def encode_joints(x_joints):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	#define a fully connected layer
	W_fc1 = tf.Variable(tf.truncated_normal(shape = [DOF,FC_UNITS], stddev = 0.1))
	b_fc1 = tf.Variable(tf.constant(0.,shape = [FC_UNITS]))
	h_fc1 = tf.nn.relu(tf.matmul(x_joints,W_fc1) + b_fc1)
	#now pass through second fully connected layer
	W_fc2 = tf.Variable(tf.truncated_normal(shape = [FC_UNITS, FC_UNITS_JOINTS], stddev = 0.1))
	b_fc2 = tf.Variable(tf.constant(0.,shape = [FC_UNITS_JOINTS]))
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

	joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]
	return h_fc2,joint_encoder_variable_list


def decode_outputs(hidden_vector):
	"""
	Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
	and generate an image of size [None,64,64,1], do this via 
	"""	
	#Assume FC_UNITS_JOINTS + FC_UNITS_IMAGE is 256
	#then reshape tensor from 2d to 4d to be compatible with deconvoh_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1))lution
	batch_size = tf.shape(hidden_vector)[0]
	hidden_image = tf.reshape(hidden_vector, shape = [batch_size,2,2,64])
	
	W_deconv1 = tf.Variable(tf.truncated_normal([2,2,DECONV_OUTPUT_CHANNELS_1,64], stddev = 0.1))
	b_deconv1 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_1]))
	deconv1 = tf.nn.conv2d_transpose(hidden_image,W_deconv1,[batch_size,4,4,DECONV_OUTPUT_CHANNELS_1],[1,2,2,1])
	h_deconv1 = tf.nn.relu(tf.nn.bias_add(deconv1,b_deconv1))

	W_deconv2 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_2,DECONV_OUTPUT_CHANNELS_1], stddev = 0.1))
	b_deconv2 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_2]))
	deconv2 = tf.nn.conv2d_transpose(h_deconv1,W_deconv2,[batch_size,8,8,DECONV_OUTPUT_CHANNELS_2],[1,2,2,1])
	h_deconv2 = tf.nn.relu(tf.nn.bias_add(deconv2,b_deconv2))

	W_deconv3 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_3,DECONV_OUTPUT_CHANNELS_2], stddev = 0.1))
	b_deconv3 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_3]))
	deconv3 = tf.nn.conv2d_transpose(h_deconv2,W_deconv3,[batch_size,16,16,DECONV_OUTPUT_CHANNELS_3],[1,2,2,1])
	h_deconv3 = tf.nn.relu(tf.nn.bias_add(deconv3,b_deconv3))

	W_deconv4 = tf.Variable(tf.truncated_normal([3,3,DECONV_OUTPUT_CHANNELS_4,DECONV_OUTPUT_CHANNELS_3], stddev = 0.1))
	b_deconv4 = tf.Variable(tf.constant(0.1, shape = [DECONV_OUTPUT_CHANNELS_4]))
	deconv4 = tf.nn.conv2d_transpose(h_deconv3,W_deconv4,[batch_size,32,32,DECONV_OUTPUT_CHANNELS_4],[1,2,2,1])
	h_deconv4 = tf.nn.relu(tf.nn.bias_add(deconv4,b_deconv4))

	W_deconv5 = tf.Variable(tf.truncated_normal([3,3,1,DECONV_OUTPUT_CHANNELS_4], stddev = 0.1))
	b_deconv5 = tf.Variable(tf.constant(0.1, shape = [1]))
	deconv5 = tf.nn.conv2d_transpose(h_deconv4,W_deconv5,[batch_size,64,64,1],[1,2,2,1])
	h_deconv5 = tf.nn.sigmoid(tf.nn.bias_add(deconv5,b_deconv5))

	decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

	return tf.squeeze(h_deconv5),decoder_variable_list



x_image = tf.placeholder(tf.float32,shape = [None,64,64])
x_joint = tf.placeholder(tf.float32,shape = [None,DOF])
y_ = tf.placeholder(tf.float32,shape = [None,64,64])

#get the encoded values for the joint angles as well as the images
encoded_image,image_encode_variable_list = encode_input_image(x_image)
encoded_joints, joint_encoder_variable_list = encode_joints(x_joint)
#now concatenate the two encoded vectors to get a single vector that may be decoded to an output image
h_encoded = tf.concat(1,[encoded_image,encoded_joints])
#decode to get image
y,decoder_variable_list = decode_outputs(h_encoded)


#now define a loss between y and the target image
#try cross entropy loss
loss = -tf.reduce_mean(tf.mul(y_,tf.log(y+1e-10)) + tf.mul(1.-y_,tf.log(1.-y + 1e-10)))#tf.reduce_mean(tf.square(y - y_))
opt = tf.train.AdamOptimizer(learning_rate)
variable_names = ["W_conv1","W_conv2","W_conv3","b_conv1","b_conv2","b_conv3","W_image_fc1","b_image_fc1","W_joint_fc1","b_joint_fc1","W_joint_fc2","b_joint_fc2","W_deconv1","W_deconv2","W_deconv3","W_deconv4","W_deconv5","b_deconv1","b_deconv2","b_deconv3","b_deconv4","b_deconv5"]
grads_and_vars = opt.compute_gradients(loss, image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
#summary_nodes = [tf.histogram_summary(variable_names[i],gv[0]) for i,gv in enumerate(grads_and_vars)]
#merged = tf.merge_all_summaries()
train_op = opt.apply_gradients(grads_and_vars)


def train_graph():
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config = config) as sess:
			#initialize the variables
			log_dir = ROOT_DIR + "/tmp/summary_logs"
			train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

			sess.run(tf.initialize_all_variables())
			#initialize a training loss array
			training_loss_array = [0] * (int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE)
			for step in xrange(int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
				joint_batch = joint_state_array_train[offset:(offset + BATCH_SIZE),...]
				input_image_batch = target_image_array_train[offset:(offset + BATCH_SIZE),...]
				target_image_batch = input_image_array_train[offset:(offset + BATCH_SIZE),...]
				feed_dict = { x_joint: joint_batch, x_image : input_image_batch, y_ : target_image_batch}

				#run the graph
				_, l= sess.run(
					[train_op,loss],
					feed_dict=feed_dict)
				training_loss_array[step] = l

				if step % 20 == 0:
					print step,l
				if step % EVAL_FREQUENCY == 0:
					predictions,test_loss_array = eval_in_batches(sess)
					#train_writer.add_summary(merged_summary,step)
					print "Test Loss is " + str(np.mean(test_loss_array))
			predictions,test_loss_array = eval_in_batches(sess)
		return predictions,training_loss_array,test_loss_array

def eval_in_batches(sess):
		"""Get combined loss for dataset by running in batches"""
		size = TRAIN_SIZE

		if size < EVAL_BATCH_SIZE:
			raise ValueError("batch size for evals larger than dataset: %d" % size)

		predictions = np.ndarray(shape = (size,IMAGE_SIZE,IMAGE_SIZE), dtype = np.float32)
		test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
		i = 0
		for begin in xrange(0,size,EVAL_BATCH_SIZE):
			end = begin + EVAL_BATCH_SIZE
			
			if end <= size:
				predictions[begin:end, ...],l = sess.run([y,loss],feed_dict={x_joint: joint_state_array_eval[begin:end, ...], x_image : input_image_array_eval[begin:end, ...], y_ : target_image_array_eval[begin:end,...] })
			else:
				batch_prediction,l = sess.run([y,loss],feed_dict={x_joint: joint_state_array_eval[-EVAL_BATCH_SIZE:, ...], x_image : input_image_array_eval[-EVAL_BATCH_SIZE:, ...], y_ : target_image_array_eval[-EVAL_BATCH_SIZE:,...] })
				predictions[begin:, ...] = batch_prediction[-(size - begin):,...]

			test_loss_array[i] = l
			i += 1
		return predictions,test_loss_array

predictions,training_loss_array,test_loss_array = train_graph()

#now plot the subplots
# Four axes, returned as a 2-d array
if DISPLAY:
	
	f, axarr = plt.subplots(2, display_num)

	for i in range(display_num):
		output_sample = predictions[i,...]
		input_sample = target_image_array_eval[i,...]
		axarr[0,i].imshow(output_sample * 255,cmap = "Greys_r")
		axarr[1,i].imshow(input_sample * 255,cmap = "Greys_r")

	plt.show()

#now save images in a directory
#loop through predictions and save
for i in range(TRAIN_SIZE):
	plt.imsave(ROOT_DIR + "Output_Images/" + "output_image" + str(i) + ".png", predictions[i,...], cmap = "Greys_r")
	plt.imsave(ROOT_DIR + "Output_Images/" + "target_image" + str(i) + ".png", target_image_array_eval[i,...], cmap = "Greys_r")
	plt.imsave(ROOT_DIR + "Output_Images/" + "input_image" + str(i) + ".png", input_image_array_eval[i,...], cmap = "Greys_r")

#save testing loss array with pickle
with open(ROOT_DIR + "training_loss.npy","wb") as f:
	pickle.dump(training_loss_array,f)

print "Test Loss is " + str(np.mean(test_loss_array))
