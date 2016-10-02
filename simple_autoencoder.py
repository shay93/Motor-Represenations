import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import png
import os
#define a model based off mnist that is able to take your shapes and reconstruct them

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 12
IMAGE_SIZE = 64
NUM_EPOCHS = 40
EVAL_BATCH_SIZE = 12
BATCH_SIZE = 12
EVAL_FREQUENCY = 20 #num of steps between evaluations
shape_str_array = ['Rectangle', 'Square', 'Triangle']
ROOT_DIR = "Simple_Autoencoder_Outputs/"


def extract_data(data_directory,num_of_images):
	"""
	extract the specified number of Images into a numpy array

	"""
	
	#initialize a numpy array to hold the data
	image_array = np.zeros([num_of_images,IMAGE_SIZE,IMAGE_SIZE,1])
	for image_num in xrange(num_of_images):
		#figure out which shape needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		#this information may now be combined to designate a file path and load the right image
		image_path = data_directory + shape_str_array[shape_name_index] + str(shape_index) + ".png"
		#load a single image and add it to image array
		image_array[image_num,:,:,0] = plt.imread(image_path)


	return image_array


train_data = extract_data("Training_Images/", 3000)

#generate a validation set
validation_data = train_data[:VALIDATION_SIZE, ...]
train_data = train_data[VALIDATION_SIZE:, ...]
num_epochs = NUM_EPOCHS
train_size = train_data.shape[0]


#Network Parameters
n_kernels_1 = 64
kernel_width_1 = 10
n_hidden_fc1 =  256 # 1st layer num features
n_hidden_fc2 = 128 # 2nd layer num features
n_input = 64*64 # num of pixels of single image input

weights = {
	'encoder_conv1' : tf.Variable(tf.random_normal([kernel_width_1,kernel_width_1,1,n_kernels_1])),
	'encoder_fc1' : tf.Variable(tf.random_normal([n_kernels_1*n_input,n_hidden_fc1])),
	'encoder_fc2': tf.Variable(tf.random_normal([n_hidden_fc1, n_hidden_fc2])),
	'decoder_fc1' : tf.Variable(tf.random_normal([n_hidden_fc2, n_hidden_fc1])),
	'decoder_fc2': tf.Variable(tf.random_normal([n_hidden_fc1, n_input]))
}

biases = {
	"encoder_conv1" : tf.Variable(tf.random_normal([n_kernels_1])),
	"encoder_fc1": tf.Variable(tf.random_normal([n_hidden_fc1])),
	"encoder_fc2": tf.Variable(tf.random_normal([n_hidden_fc2])),
	'decoder_fc1': tf.Variable(tf.random_normal([n_hidden_fc1])),
	'decoder_fc2': tf.Variable(tf.random_normal([n_input]))
}

#Build the encoder
def encoder(x):
	"""
	x is a tensor of shape [None n_input]
	"""
	#Encoder Hidden layer with relu activation
	conv1 = tf.nn.conv2d(x,weights['encoder_conv1'],strides = [1,1,1,1],padding = "SAME")
	h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,biases['encoder_conv1']))
	h_conv1_reshape = tf.reshape(h_conv1,shape = [-1,n_kernels_1*n_input])
	fc_layer_1 = tf.nn.relu(tf.add(tf.matmul(h_conv1_reshape,weights['encoder_fc1']), biases["encoder_fc1"]))
	fc_layer_2 = tf.nn.relu(tf.add(tf.matmul(fc_layer_1,weights['encoder_fc2']), biases['encoder_fc2']))
	return fc_layer_2

def decoder(x):
	"""
	Takes an input of shape [None,n_hidden_2]
	"""
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['decoder_fc1']), biases['decoder_fc1']))
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['decoder_fc2']), biases['decoder_fc2']))
	return layer_2

def variable_summaries(var,name):
	"""
	Attach summaries to a tensor
	"""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name,mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)

#construct model
#This is where input data is fed into the graph
X = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
#Encoder the input
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


def normalize_by_max(data):
	max_batch_element = tf.reduce_max(data,[1])
	#reshape max_batch element
	max_batch_element_reshape = tf.reshape(max_batch_element, shape = [-1,1])
	#tile the max batch element so that an element to element division may be carried out
	max_batch_tiled = tf.tile(max_batch_element_reshape,[1,IMAGE_SIZE*IMAGE_SIZE])
	#take the norm of y and set as the output
	return tf.div(data,max_batch_tiled)

#Prediction
y_pred = normalize_by_max(decoder_op)
X_reshape = tf.reshape(X,shape = [-1,n_input])
#define a cost node
loss = tf.reduce_mean(tf.pow(X_reshape - y_pred,2))

#define a learning rate
learning_rate = 1e-4

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#utility function to evaluate a dataset by feedin batches of data pulling results from eval_loss
def eval_in_batches(data,sess):
	"""Get combined loss for dataset by running in batches"""
	size = data.shape[0]

	if size < EVAL_BATCH_SIZE:
		raise ValueError("batch size for evals larger than dataset: %d" % size)

	predictions = np.ndarray(shape = (size,IMAGE_SIZE*IMAGE_SIZE), dtype = np.float32)
	for begin in xrange(0,size,EVAL_BATCH_SIZE):
		end = begin + EVAL_BATCH_SIZE
		
		if end <= size:
			predictions[begin:end, ...] = sess.run(y_pred,feed_dict={X: data[begin:end, ...]})
		else:
			batch_prediction = sess.run(y_pred,feed_dict = {X : data[-EVAL_BATCH_SIZE:, ...]})
			predictions[begin:, ...] = batch_prediction[-(size - begin):,...]

	
	return predictions


def unwrap_eval_prediction(predictions,eval_num):
	for image_num in xrange(VALIDATION_SIZE):
		#figure out which shape needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		#this information may now be combined to designate a file path and load the right image
		my_dir = ROOT_DIR 
		try:
			os.stat(my_dir)
		except:
			os.mkdir(my_dir)
		image_path = my_dir + shape_str_array[shape_name_index] + str(shape_index) + "_evalnum_%d" %(eval_num) + ".png"
		#reshape the image and scale it 
		image = np.reshape(predictions[image_num,:],[IMAGE_SIZE,IMAGE_SIZE])
		temp = np.round(image * PIXEL_DEPTH)
		png.from_array(temp.tolist(),'L').save(image_path)


#initialize a numpy array to store loss
loss_array = np.ndarray(shape = [int(num_epochs * train_size) // BATCH_SIZE], dtype = np.float32)


with tf.Session() as sess:
	#initialize the variables
	log_dir = ROOT_DIR + "/tmp/summary_logs"
	train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

	tf.initialize_all_variables().run()
	for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
		#compute the offset of the current minibatch in the data
		offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
		batch_data = train_data[offset:(offset + BATCH_SIZE),...]
		feed_dict = {X : batch_data}

		#run the graph
		_, l = sess.run(
			[optimizer,loss],
			feed_dict=feed_dict)
		loss_array[step] = l


		if step % EVAL_FREQUENCY == 0:
			predictions = eval_in_batches(validation_data, sess)
			unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY)
			print step,l

	save_as_npy(sess,loss_array,weigts['encoder_conv1'])

def save_as_npy(sess,training_loss_array,W_conv1_op):
	"""
	Saves the training loss and evaluation loss as an npy in addition to the weights prescribed as W_conv1 and W_conv2
	inputs: training_loss and testing_loss are both numpy arrays  
	"""
	file_path_list = ["training_loss.npy","W_conv1.npy"]
	#evaluate the weight tensors
	W_conv1 = sess.run(W_conv1_op)
	#construct value list
	value_list = [training_loss_array,W_conv1]

	for file_path,value in zip(file_path_list,value_list):
		with open(self.output_root_directory + file_path,'w') as f:
			pickle.dump(value,f)
			f.close() 














