from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
import pickle
import training_tools as tt
import os

#define a max sequence length
DOF = 3
GAUSS_STD = 2.5
link_length = 30
#model globals
CONV_KERNELS_1 = 64
CONV_KERNELS_2 = 32
CONV_KERNELS_3 = 16
CONV_KERNELS_4 = 8
CONV_KERNELS_5 = 4
DECONV_OUTPUT_CHANNELS_1 = 32
DECONV_OUTPUT_CHANNELS_2 = 16
DECONV_OUTPUT_CHANNELS_3 = 8
DECONV_OUTPUT_CHANNELS_4 = 4

##Parameters for Joint encoder
FC_UNITS_JOINTS_1 = 100
FC_UNITS_JOINTS_FINAL = 56

#Parameters for Image encoder
FC_UNITS_IMAGE = 1024 - 56
#model globals
NUM_SAMPLES = 10000
IMAGE_SIZE = 64
BATCH_SIZE = 200
learning_rate = 1e-2
display_num = 10
EVAL_BATCH_SIZE = 200
EPOCHS = 500
TRAIN_SIZE = 400
ROOT_DIR = "Joints_to_Image/"
EVAL_FREQUENCY = 60
DISPLAY = False
KEEP_PROB = 1.0
LAMBDA = 4.5e-3

##########################HELPER FUNCTION#########################
def regularizer(tensor):
	# with tf.op_scope([tensor], scope, 'L2Regularizer'):
 # 		l2_weight = tf.convert_to_tensor(weight,
 #                                   dtype=tensor.dtype.base_dtype,
 #                                   name='weight')
  	return tf.nn.l2_loss(tensor)
##############################LEGACY############################
def load_data(num):
	with open(ROOT_DIR + "joint_state_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		joint_state_array = pickle.load(f)[:num,...]

	with open(ROOT_DIR + "target_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		target_image_array = pickle.load(f)[:num,...]

	with open(ROOT_DIR + "input_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		input_image_array = pickle.load(f)[:num,...]


	return joint_state_array,target_image_array,input_image_array



joint_state_array,target_image_array,input_image_array = load_data(NUM_SAMPLES)
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
	b_conv3 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_3]))
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

def encode_joints(x_joints):
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


def decode_outputs(hidden_vector):
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



x_image = tf.placeholder(tf.float32,shape = [None,64,64])
x_joint = tf.placeholder(tf.float32,shape = [None,DOF])
y_ = tf.placeholder(tf.float32,shape = [None,64,64])

#get the encoded values for the joint angles as well as the images
encoded_image,image_encode_variable_list,image_weights = encode_input_image(x_image)
encoded_joints, joint_encoder_variable_list,joint_weights = encode_joints(x_joint)
#now concatenate the two encoded vectors to get a single vector that may be decoded to an output image
h_encoded = tf.concat(1,[encoded_image,encoded_joints])
print h_encoded
#h_encoded_dropped = tf.nn.dropout(h_encoded,KEEP_PROB)
#decode to get image
y_before_sigmoid,decoder_variable_list,decoder_weights = decode_outputs(h_encoded)
#apply sigmoid to get y
y = tf.nn.sigmoid(y_before_sigmoid)
#get the regularaization term
weight_norm_sum = 0
for weight in image_weights + joint_weights + decoder_weights:
	weight_norm_sum += regularizer(weight)
#now define a loss between y and the target image
#try cross entropy loss
#-tf.reduce_mean(tf.mul(y_,tf.log(y+1e-10)) + tf.mul(1.-y_,tf.log(1.-y + 1e-10)))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_before_sigmoid,y_)) + LAMBDA*weight_norm_sum
opt = tf.train.AdamOptimizer(learning_rate)
variable_names = ["W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","b_conv1","b_conv2","b_conv3","b_conv4", "b_conv5","W_image_fc1","b_image_fc1","W_joint_fc1","b_joint_fc1","W_joint_fc2","b_joint_fc2","W_deconv1","W_deconv2","W_deconv3","W_deconv4","W_deconv5","b_deconv1","b_deconv2","b_deconv3","b_deconv4","b_deconv5"]
grads_and_vars = opt.compute_gradients(loss, image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
summary_nodes = [tf.histogram_summary(variable_names[i],gv[0]) for i,gv in enumerate(grads_and_vars)]
merged = tf.merge_all_summaries()
train_op = opt.apply_gradients(grads_and_vars)


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
				_, l, merged_summary= sess.run(
					[train_op,loss,merged],
					feed_dict=feed_dict)
				training_loss_array[step] = l

				if step % 20 == 0:
					train_writer.add_summary(merged_summary,step)
					print step,l
				if step % EVAL_FREQUENCY == 0:
					predictions,test_loss_array = eval_in_batches(sess)
					print "Test Loss is " + str(np.mean(test_loss_array))
					average_test_loss.append(np.mean(test_loss_array))
					#also svae the predictions to get
					checkpoint_num = step // EVAL_FREQUENCY
					#use the checkpoint_num to specify the correct directory to save an image
					checkpoint_dir = ROOT_DIR + "Checkpoint" + str(checkpoint_num) + "/"
					if not os.path.exists(checkpoint_dir):
						os.makedirs(checkpoint_dir)
					for i in range(TRAIN_SIZE):
						plt.imsave(checkpoint_dir + "output_image" + str(i) + ".png", predictions[i,...], cmap = "Greys_r")
						plt.imsave(checkpoint_dir + "target_image" + str(i) + ".png", target_image_array_eval[i,...], cmap = "Greys_r")
						plt.imsave(checkpoint_dir + "input_image" + str(i) + ".png", input_image_array_eval[i,...], cmap = "Greys_r")

			predictions,test_loss_array = eval_in_batches(sess)
			#now get the learned variable values and dump to a list
			variable_list = sess.run(image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
			with open(ROOT_DIR + "learned_variable_list.npy","wb") as f:
				pickle.dump(variable_list,f)
		return predictions,training_loss_array,test_loss_array,average_test_loss

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

predictions,training_loss_array,test_loss_array,average_test_loss = train_graph()

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

with open(ROOT_DIR + "average_test_loss.npy","wb") as f:
	pickle.dump(average_test_loss,f)


print "Test Loss is " + str(np.mean(test_loss_array))

#now write a function that can use the predictions to calculate a percentage for IoU
#I like a code
#Go to ETH
#first construct an array of pixels for the
threshold_list = np.arange(0,0.9,step = 0.025)
IoU_list = []
for i,threshold in enumerate(threshold_list):
	good_mapping_count = 0
	bad_mapping_count = 0
	for i in range(TRAIN_SIZE):
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


with open(ROOT_DIR + "percentage_correct.npy","wb") as f:
	pickle.dump(IoU_list,f)

#happy birthday.py
