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

#Parameters for Image encoder
FC_UNITS_IMAGE = 1024 - 56
#model globals
NUM_SAMPLES = 5000
IMAGE_SIZE = 64
BATCH_SIZE = 1000
learning_rate = 1e-3
display_num = 10
EVAL_BATCH_SIZE = 200
EPOCHS = 3000
EVAL_SIZE = 200
TRAIN_SIZE = NUM_SAMPLES - EVAL_SIZE
ROOT_DIR = "Joints_to_Image/"
SUMMARY_DIR = "/tmp/summary_logs"
SAVE_DIR = "/tmp/model.cpkt"
EVAL_FREQUENCY = 2000
DISPLAY = False
KEEP_PROB = 1.0
LAMBDA = 1e-3

############################DEFINE PARAMETERS FOR JOINT TO SEQ MAP#######################################

observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 20}
joint_encoder_parameters = {"fc_1" : 200 , "fc_2" : 56}
output_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 200}
output_image_decoder_parameters = {"deconv_output_channels_1" : 32, "deconv_output_channels_2" : 16, "deconv_output_channels_3" : 8, "deconv_output_channels_4" : 4, "deconv_output_channels_5" : 1}


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
print "Joint State Array Shape",np.shape(joint_state_array)
print "Tareget Image Array Shape",np.shape(target_image_array)
print "Input Image Array Shape",np.shape(input_image_array)
#split this data into a training and validation set
joint_state_array_train = joint_state_array[EVAL_SIZE:,...]
target_image_array_train = target_image_array[EVAL_SIZE:,...]
input_image_array_train = input_image_array[EVAL_SIZE:,...]
#now specify the eval set
joint_state_array_eval = joint_state_array[:EVAL_SIZE,...]
target_image_array_eval = target_image_array[:EVAL_SIZE,...]
input_image_array_eval = input_image_array[:EVAL_SIZE,...]

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
		W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_conv1")
		#initiaize the biases
		b = tf.Variable(tf.constant(0.1,shape = [weight_shape[-1]]), trainable = trainable, name = "b_conv1")
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

def deconv(x,weight_shape,output_shape,scope,strides = [1,2,2,1], stddev = 0.1,trainable = True, reuse_variables = False):
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
		h = tf.nn.relu(tf.nn.bias_add(deconv,b))

	return h,W,b


def encode_previous_output_image(previous_output_image):
	"""
	Takes an input placeholder for an image
	"""

	#expand the dimensionality of the input image
	x_image = tf.expand_dims(previous_output_image, -1)
	#find the activations of the first conv layer
	h_conv1,W_conv1,b_conv1 = conv(x_image,[3,3,1,observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_output",trainable = True)
	#find the activations of the second conv layer
	h_conv2,W_conv2,b_conv2 = conv(h_conv1,[3,3,observed_image_encoder_parameters["conv1_kernels"],observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_output",trainable = True)
	#find the activations of the third conv layer
	h_conv3,W_conv3,b_conv3 = conv(h_conv2,[3,3,observed_image_encoder_parameters["conv2_kernels"],observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_output",trainable = True)
	#find the activations of the second conv layer
	h_conv4,W_conv4,b_conv4 = conv(h_conv3,[3,3,observed_image_encoder_parameters["conv3_kernels"],observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_output",trainable = True)
	#find the activations of the second conv layer
	h_conv5,W_conv5,b_conv5 = conv(h_conv4,[3,3,observed_image_encoder_parameters["conv4_kernels"],observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_output",trainable = True)
	#flatten the activations in the final conv layer in order to obtain an output image
	h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*observed_image_encoder_parameters["conv5_kernels"]])
	#pass flattened activations to a fully connected layer
	h_fc1,W_fc1,b_fc1 = fc_layer(h_conv5_reshape,[4*observed_image_encoder_parameters["conv5_kernels"],1024 - 56],"fc_layer_encode_output",trainable = True)
	output_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]

	return h_fc1,output_image_encoder_variable_list 

def encode_joints(x_joints):
	"""
	Takes joint states and encodes them in order to generate an image
	"""
	h_fc1,W_fc1,b_fc1 = fc_layer(x_joints,[DOF,joint_encoder_parameters["fc_1"]],"fc_joint_encoder_1",trainable = True)
	#pass the activations to a second fc layer
	h_fc2,W_fc2,b_fc2 = fc_layer(h_fc1,[joint_encoder_parameters["fc_1"], joint_encoder_parameters["fc_2"]],"fc_joint_encoder_2",trainable = True)
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
	h_deconv1,W_deconv1,b_deconv1 = deconv(hidden_image,[2,2,output_image_decoder_parameters['deconv_output_channels_1'],64],[batch_size,4,4,output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1])
	#calculate activations for second deconv layer
	h_deconv2,W_deconv2,b_deconv2 = deconv(h_deconv1,[3,3,output_image_decoder_parameters['deconv_output_channels_2'],output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2")
	#calculate activations for third deconv layer
	h_deconv3,W_deconv3,b_deconv3 = deconv(h_deconv2,[3,3,output_image_decoder_parameters['deconv_output_channels_3'],output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3")
	#calculate activations for fourth deconv layer
	h_deconv4,W_deconv4,b_deconv4 = deconv(h_deconv3,[3,3,output_image_decoder_parameters['deconv_output_channels_4'],output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4")
	#calculate activations for fifth deconv layer
	h_deconv5,W_deconv5,b_deconv5 = deconv(h_deconv4,[3,3,output_image_decoder_parameters['deconv_output_channels_5'],output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5")
	decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

	return tf.squeeze(h_deconv5),decoder_variable_list


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


x_image = tf.placeholder(tf.float32,shape = [None,64,64])
x_joint = tf.placeholder(tf.float32,shape = [None,DOF])
y_ = tf.placeholder(tf.float32,shape = [None,64,64])

#pass the input image and joint angle tensor to jointangle2image to get y_before_sigmoid
y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list = jointangle2image(x_joint,x_image)
#apply sigmoid to get y
y = tf.nn.sigmoid(y_before_sigmoid)
#define the loss op using the y before sigmoid and in the cross entropy sense
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_before_sigmoid,y_))
#add a summary node to record the los
tf.scalar_summary("loss summary",loss)
#define the optimizer with the specified learning rate
opt = tf.train.AdamOptimizer(learning_rate)
variable_names = ["W_conv1","W_conv2","W_conv3","W_conv4","W_conv5","b_conv1","b_conv2","b_conv3","b_conv4", "b_conv5","W_image_fc1","b_image_fc1","W_joint_fc1","b_joint_fc1","W_joint_fc2","b_joint_fc2","W_deconv1","W_deconv2","W_deconv3","W_deconv4","W_deconv5","b_deconv1","b_deconv2","b_deconv3","b_deconv4","b_deconv5"]
grads_and_vars = opt.compute_gradients(loss, joint_encoder_variable_list + image_encode_variable_list + decoder_variable_list)
gradient_summary_nodes = [tf.histogram_summary(variable_names[i] + "_gradients",gv[0]) for i,gv in enumerate(grads_and_vars)]
var_summary_nodes = [tf.histogram_summary(variable_names[i],gv[1]) for i,gv in enumerate(grads_and_vars)]
merged = tf.merge_all_summaries()
train_op = opt.apply_gradients(grads_and_vars)
#define an op to initialize variables
init_op = tf.initialize_all_variables()
# Add ops to save and restore all the variables.
saver = tf.train.Saver(joint_encoder_variable_list+image_encode_variable_list+decoder_variable_list)

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


			#initialize a training loss array
			training_loss_array = [0] * (int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE)
			for step in xrange(int(EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (TRAIN_SIZE)
				joint_batch = joint_state_array_train[offset:(offset + BATCH_SIZE),...]
				input_image_batch = input_image_array_train[offset:(offset + BATCH_SIZE),...]
				target_image_batch = target_image_array_train[offset:(offset + BATCH_SIZE),...]
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
					for i in range(EVAL_SIZE):
						plt.imsave(checkpoint_dir + "output_image" + str(i) + ".png", predictions[i,...], cmap = "Greys_r")
						plt.imsave(checkpoint_dir + "target_image" + str(i) + ".png", target_image_array_eval[i,...], cmap = "Greys_r")
						plt.imsave(checkpoint_dir + "input_image" + str(i) + ".png", input_image_array_eval[i,...], cmap = "Greys_r")

			predictions,test_loss_array = eval_in_batches(sess)
			#now get the learned variable values and dump to a list
			variable_list = sess.run(image_encode_variable_list + joint_encoder_variable_list + decoder_variable_list)
			with open(ROOT_DIR + "learned_variable_list.npy","wb") as f:
				pickle.dump(variable_list,f)

			save_path = saver.save(sess, ROOT_DIR + SAVE_DIR)
  			print("Model saved in file: %s" % save_path)
		return predictions,training_loss_array,test_loss_array,average_test_loss

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
for i in range(EVAL_SIZE):
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


with open(ROOT_DIR + "percentage_correct.npy","wb") as f:
	pickle.dump(IoU_list,f)

#happy birthday.py
