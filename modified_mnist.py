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
NUM_EPOCHS = 60
EVAL_BATCH_SIZE = 9
BATCH_SIZE = 256
EVAL_FREQUENCY = 20 #num of steps between evaluations
shape_str_array = ['Rectangle', 'Square', 'Triangle']
ROOT_DIR = "Modified_Mnist_Outputs/"


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


#This is where training samples are fed into the graph
train_data_node = tf.placeholder(tf.float32,shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

eval_data = tf.placeholder(tf.float32, shape = [EVAL_BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

conv1_weights = tf.Variable(tf.truncated_normal([5,5,NUM_CHANNELS,32],stddev = 0.1))
conv1_biases = tf.Variable(tf.zeros([32]))

conv2_weights = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
conv2_biases = tf.Variable(tf.zeros([64]))

fc1_weights = tf.Variable(
	tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
						stddev = 0.1))

fc1_biases = tf.Variable(tf.constant(0.1,shape = [512]))

fc2_weights = tf.Variable(tf.truncated_normal([512,IMAGE_SIZE*IMAGE_SIZE]))

fc2_biases = tf.Variable(tf.constant(0.1,shape = [IMAGE_SIZE*IMAGE_SIZE]))


def model(data,train = False):

	conv = tf.nn.conv2d(data,conv1_weights,strides = [1,1,1,1],padding = 'SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv1_biases))

	pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME")
	conv = tf.nn.conv2d(pool,conv2_weights,strides = [1,1,1,1], padding = "SAME")
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv2_biases))
	pool = tf.nn.max_pool(relu, ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME")
	reshape = tf.reshape(pool,[-1,IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64])

	hidden = tf.nn.relu(tf.matmul(reshape,fc1_weights) + fc1_biases)

	if train:
		hidden = tf.nn.dropout(hidden,0.9)

	return tf.reshape(tf.nn.relu(tf.matmul(hidden,fc2_weights) + fc2_biases),shape = [-1,IMAGE_SIZE,IMAGE_SIZE,1])


def normalize_by_max(data):
	max_batch_element = tf.reduce_max(data,[1,2,3])
	#reshape max_batch element
	max_batch_element_reshape = tf.reshape(max_batch_element, shape = [-1,1,1,1])
	#tile the max batch element so that an element to element division may be carried out
	max_batch_tiled = tf.tile(max_batch_element_reshape,[1,IMAGE_SIZE,IMAGE_SIZE,1])
	#take the norm of y and set as the output
	return tf.div(data,max_batch_tiled)



output_image_node = normalize_by_max(model(train_data_node,True))
diff = tf.sub(output_image_node,train_data_node)
loss = tf.reduce_mean(tf.square(diff)) #take an L1 loss of the tensor

#l2 regularization for the fully connected parameters
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
				tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

#Add the regularization term to the loss
loss += 5e-3 * regularizers

#Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay
batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
	10.,
	batch * BATCH_SIZE,
	train_size,
	0.95,
	staircase = True)

#Use simple momentum for optimization
optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step = batch)

train_prediction = output_image_node

eval_prediction = normalize_by_max(model(eval_data))
eval_loss = tf.reduce_sum(tf.abs(tf.sub(eval_prediction,eval_data)))

#utility function to evaluate a dataset by feedin batches of data pulling results from eval_loss
def eval_in_batches(data,sess):
	"""Get combined loss for dataset by running in batches"""
	#initialize a batch number

	size = data.shape[0]

	if size < EVAL_BATCH_SIZE:
		raise ValueError("batch size for evals larger than dataset: %d" % size)

	predictions = np.ndarray(shape = (size,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS), dtype = np.float32)
	for begin in xrange(0,size,EVAL_BATCH_SIZE):
		end = begin + EVAL_BATCH_SIZE
		
		if end <= size:
			predictions[begin:end, ...] = sess.run(eval_prediction,feed_dict={eval_data: data[begin:end, ...]})
		else:
			batch_prediction = sess.run(eval_prediction,feed_dict = {eval_data : data[-EVAL_BATCH_SIZE:, ...]})
			predictions[begin:, ...] = batch_prediction

	
	return predictions


def unwrap_eval_prediction(predictions,eval_num):
	for image_num in xrange(VALIDATION_SIZE):
		#figure out which shape needs to be loaded
		shape_name_index = image_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_num // len(shape_str_array)
		#this information may now be combined to designate a file path and load the right image
		my_dir = ROOT_DIR + "Predictions%d" % (eval_num)
		try:
			os.stat(my_dir)
		except:
			os.mkdir(my_dir)
		image_path = my_dir + "/" + shape_str_array[shape_name_index] + str(shape_index) + ".png"
		#save the image
		temp = np.round(predictions[image_num,:,:,0] * PIXEL_DEPTH)
		png.from_array(temp.tolist(),'L').save(image_path)




with tf.Session() as sess:
	#initialize the variables
	tf.initialize_all_variables().run()
	for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
		#compute the offset of the current minibatch in the data
		offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
		batch_data = train_data[offset:(offset + BATCH_SIZE)]
		feed_dict = {train_data_node : batch_data}

		#run the graph
		_, l, lr = sess.run(
			[optimizer,loss, learning_rate],
			feed_dict=feed_dict)


		if step % EVAL_FREQUENCY == 0:
			predictions = eval_in_batches(validation_data, sess)
			unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY)
			print step,l,lr











