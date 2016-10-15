import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import png
import os
#define a model based off mnist that is able to take your shapes and reconstruct them

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 180
IMAGE_SIZE = 64
NUM_EPOCHS =2
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
	image_array = np.zeros([num_of_images,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
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
n_hidden_1 =  256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 64*64 # num of pixels of single image input

weights = {
	'encoder_h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
	"encoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
	"encoder_b2": tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}

#Build the encoder
def encoder(x):
	"""
	x is a tensor of shape [None n_input]
	"""
	#Encoder Hidden layer with relu activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['encoder_h1']), biases["encoder_b1"]))
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['encoder_h2']), biases['encoder_b2']))
	return layer_2

def decoder(x):
	"""
	Takes an input of shape [None,n_hidden_2]
	"""
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['decoder_h1']), biases['decoder_b1']))
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['decoder_h2']), biases['decoder_b2']))
	return layer_2


#construct model
#This is where input data is fed into the graph
X = tf.placeholder(tf.float32,[None,n_input])
#Encoder the input
encoder_op = encoder(X)
decoder_op = decoder(decoder_op)


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

#define a cost node
cost = tf.reduce_mean(tf.pow(X - y_pred,2))

#define a learning rate
learning_rate = 0.01

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


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
			predictions[begin:end, ...] = sess.run(eval_prediction,feed_dict={X: data[begin:end, ...]})
		else:
			batch_prediction = sess.run(eval_prediction,feed_dict = {X : data[-EVAL_BATCH_SIZE:, ...]})
			predictions[begin:, ...] = batch_prediction

	
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
		image = np.reshape(predictions[image_num,:],[Image_Shape,Image_Shape])
		temp = np.round(image * PIXEL_DEPTH)
		png.from_array(temp.tolist(),'L').save(image_path)




with tf.Session() as sess:
	#initialize the variables
	tf.initialize_all_variables().run()
	for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
		#compute the offset of the current minibatch in the data
		offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
		batch_data = train_data[offset:(offset + BATCH_SIZE)]
		feed_dict = {X : batch_data}

		#run the graph
		_, l = sess.run(
			[optimizer,loss],
			feed_dict=feed_dict)


		if step % EVAL_FREQUENCY == 0:
			predictions = eval_in_batches(validation_data, sess)
			unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY)
			print step,l











