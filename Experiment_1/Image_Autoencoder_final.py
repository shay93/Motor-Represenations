from __future__ import division 
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt
import png
import pickle
import os


#Globals
BATCH_SIZE = 256
IMAGE_SIZE = 64
PIXEL_DEPTH = 255
EVAL_BATCH_SIZE = 12
EPOCHS = 500
FC_2_UNITS = 2000
EVAL_FREQUENCY = 50
NUM_CHANNELS = 1
VALIDATION_SIZE = 256
shape_str_array = ['Rectangle', 'Square', 'Triangle']
ROOT_DIR = "Image_Autoencoder_Ver2_Outputs/"


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


train_data = extract_data("Training_Images_Thick_Lines/", 3000)

#generate a validation set
validation_data = train_data[:VALIDATION_SIZE, ...]
train_data = train_data[VALIDATION_SIZE:, ...]
num_epochs = EPOCHS
train_size = train_data.shape[0]


class Shape_Autoencoder:

	def __init__(self):
		"""
		initialize the shape Autoencoders hyperparameters
		"""
		self.batch_size = BATCH_SIZE
		self.img_width = IMAGE_SIZE
		self.conv_kernels_1 = 32
		self.conv_kernels_2 = 32
		self.op_dict = {}
		self.parameter_dict = {}
		self.dropout_prob = 0.95
		#intialize some directory names
		self.output_root_directory = "Image_Autoencoder_Ver2_Outputs/"
		self.output_image_directory = self.output_root_directory + "Output_Images/"
	

	def build_graph(self):
		"""
		Responsible for defining the operations that comprise the graph
		inputs: -IMG_WIDTH = 64

		outputs: A session objecta and a operation dictionary
		"""
		
		#first specify a placeholder for the input image which is of size 64 by 64
		with tf.name_scope('Input_placeholder') as scope:
			self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None,self.img_width,self.img_width,1])
		
		#define a place holder for the outputs
		with tf.name_scope('Output_placeholder') as scope:
			self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [None,self.img_width,self.img_width,1])

		with tf.name_scope("Conv1") as scope:	
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_conv1'] = tf.Variable(tf.truncated_normal([5,5,1,self.conv_kernels_1],stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_conv1'] = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_1]))
			with tf.name_scope("Conv_Output") as scope:
				conv1 = tf.nn.conv2d(self.op_dict['x'],self.parameter_dict['W_conv1'],strides = [1,1,1,1],padding = 'SAME')
			with tf.name_scope("Activation") as scope:	
				h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,self.parameter_dict['b_conv1']))
		
		with tf.name_scope("Pool1") as scope:		
			pool1 = tf.nn.max_pool(h_conv1, ksize =[1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
		
		with tf.name_scope("Conv2") as scope:
			#define parameters for the second convolutional layer
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_conv2'] = tf.Variable(tf.truncated_normal([5,5,self.conv_kernels_1,self.conv_kernels_2],stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_conv2'] = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_2]))
			with tf.name_scope("Conv_Output") as scope:	
				#consider second layer
				conv2 = tf.nn.conv2d(pool1,self.parameter_dict['W_conv2'],strides = [1,1,1,1],padding = 'SAME')
			with tf.name_scope("Activation") as scope:	
				h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2,self.parameter_dict['b_conv2']))

		with tf.name_scope("Pool2") as scope:		
			pool2 = tf.nn.max_pool(h_conv2, ksize = [1,3,3,1],strides = [1,2,2,1], padding = 'SAME')
		
		with tf.name_scope("Reshape_h_conv2") as scope:
			#Reshape the output from pooling layers to pass to fully connected layers
			h_conv2_reshape = tf.reshape(pool2, shape = [-1,self.img_width*self.img_width*self.conv_kernels_2 // 16])

		with tf.name_scope("FC1") as scope:
			#define parameters for full connected layer
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_fc1'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width*self.conv_kernels_2 // 16,FC_2_UNITS],stddev = 0.1)) 
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_fc1'] = tf.Variable(tf.constant(0.,shape = [FC_2_UNITS])) 
			with tf.name_scope("Activation") as scope:
				h_fc1 = tf.nn.relu(tf.matmul(h_conv2_reshape, self.parameter_dict['W_fc1']) + self.parameter_dict['b_fc1'])

		with tf.name_scope("Dropout") as scope:
			h_dropout = tf.nn.dropout(h_fc1,self.dropout_prob)


		with tf.name_scope("FC2") as scope:
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_fc2'] = tf.Variable(tf.truncated_normal(shape = [FC_2_UNITS,self.img_width*self.img_width],stddev = 0.1))
			with tf.name_scope("Biases") as scope:	
				self.parameter_dict['b_fc2'] = tf.Variable(tf.constant(0.,shape = [self.img_width*self.img_width]))
			with tf.name_scope("Activation") as scope: 
				h_fc2 = tf.nn.relu(tf.matmul(h_dropout,self.parameter_dict['W_fc2']) + self.parameter_dict['b_fc2'])

		with tf.name_scope("Reshape_h_fc2") as scope:
			#now reshape such that it may be used by the other convolutional layers
			h_fc2_reshape = tf.reshape(h_fc2, shape = [-1,self.img_width,self.img_width, 1])
	
		with tf.name_scope("y") as scope:
			#first calculate the max of each batch 
			max_batch_element = tf.reduce_max(h_fc2_reshape,[1,2,3])
			#reshape max_batch element
			max_batch_element_reshape = tf.reshape(max_batch_element, shape = [-1,1,1,1])
			#tile the max batch element so that an element to element division may be carried out
			max_batch_tiled = tf.tile(max_batch_element_reshape,[1,self.img_width,self.img_width,1])
			#take the norm of y and set as the output
			self.op_dict['y'] = tf.div(h_fc2_reshape,max_batch_tiled)		#define a loss function 
		
		with tf.name_scope("Loss") as scope:
			self.op_dict['meansq'] = tf.reduce_mean(tf.square(self.op_dict['y'] - self.op_dict['y_']))
		
		#define a learning rate with an exponential decay,a batch variable is needed in order to prevent 
		self.op_dict['learning_rate'] = 1e-4
		
		#define a training operation
		with tf.name_scope("Train") as scope:
			self.op_dict['train_op'] = tf.train.AdamOptimizer(self.op_dict['learning_rate']).minimize(self.op_dict['meansq'])
		

		#add the tensorboard ops
		self.Add_Tensorboard_ops()

	def train_graph(self):
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

			tf.initialize_all_variables().run()
			#initialize a training loss array
			loss_array = [0] * (int(num_epochs * train_size) // BATCH_SIZE)
			for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
				batch_data = train_data[offset:(offset + BATCH_SIZE),...]
				feed_dict = { self.op_dict['x']: batch_data, self.op_dict['y_'] : batch_data}

				#run the graph
				_, l = sess.run(
					[self.op_dict['train_op'],self.op_dict['meansq']],
					feed_dict=feed_dict)
				loss_array[step] = l


				if step % EVAL_FREQUENCY == 0:
					predictions,test_loss_array = self.eval_in_batches(validation_data, sess)
					self.unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY,"Checkpoints/")
					print step,l
			predictions,test_loss_array = self.eval_in_batches(validation_data,sess)
			self.unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY,"Validation_on_Test/")
			predictions, _ = self.eval_in_batches(train_data,sess)
			self.unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY,"Validation_on_Train/")
			self.save_as_npy(sess,loss_array,test_loss_array)
		



	def eval_in_batches(self,data,sess):
		"""Get combined loss for dataset by running in batches"""
		size = data.shape[0]

		if size < EVAL_BATCH_SIZE:
			raise ValueError("batch size for evals larger than dataset: %d" % size)

		predictions = np.ndarray(shape = (size,IMAGE_SIZE,IMAGE_SIZE,1), dtype = np.float32)
		test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
		i = 0
		for begin in xrange(0,size,EVAL_BATCH_SIZE):
			end = begin + EVAL_BATCH_SIZE
			
			if end <= size:
				predictions[begin:end, ...],l = sess.run([self.op_dict['y'],self.op_dict['meansq']],feed_dict={self.op_dict['x']: data[begin:end, ...], self.op_dict['y_'] : data[begin:end, ...]})
			else:
				batch_prediction,l = sess.run([self.op_dict['y'],self.op_dict['meansq']],feed_dict = {self.op_dict['x'] : data[-EVAL_BATCH_SIZE:, ...],self.op_dict['y_']:data[-EVAL_BATCH_SIZE:,...]})
				predictions[begin:, ...] = batch_prediction[-(size - begin):,...]

			test_loss_array[i] = l
			i += 1
		return predictions,test_loss_array


	def unwrap_eval_prediction(self,predictions,eval_num,lower_dir):
		for image_num in xrange(VALIDATION_SIZE):
			#figure out which shape needs to be loaded
			shape_name_index = image_num % len(shape_str_array)
			#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
			shape_index = image_num // len(shape_str_array)
			#this information may now be combined to designate a file path and load the right image
			my_dir = ROOT_DIR + lower_dir
			try:
				os.stat(my_dir)
			except:
				os.mkdir(my_dir)
			image_path = my_dir + shape_str_array[shape_name_index] + str(shape_index) + "_evalnum_%d" %(eval_num) + ".png"
			#reshape the image and scale it 
			image = np.reshape(predictions[image_num,:],[IMAGE_SIZE,IMAGE_SIZE])
			temp = np.round(image * PIXEL_DEPTH)
			png.from_array(temp.tolist(),'L').save(image_path)



	def variable_summaries(self,var,name):
		"""
		Attach summaries to a tensor
		"""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + name,mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.scalar_summary('stddev/' + name, stddev)
			#tf.histogram_summary(name,var)


	def Add_Tensorboard_ops(self):
		"""
		Calls on the variable summaries helper function to generate ops for the graph in order to visualize them in tensorboard 
		"""
		for label,op in self.parameter_dict.items() :
			self.variable_summaries(op,label)

		#merge the summaries
		self.op_dict['merged'] = tf.merge_all_summaries()


	def save_as_npy(self,sess,training_loss_array,testing_loss_array):
		"""
		Saves the training loss and evaluation loss as an npy in addition to the weights prescribed as W_conv1 and W_conv2
		inputs: training_loss and testing_loss are both numpy arrays  
		"""
		file_path_list = ["training_loss.npy","W_conv1.npy","W_conv2.npy","testing_loss.npy"]
		#evaluate the weight tensors
		W_conv1,W_conv2 = sess.run([self.parameter_dict['W_conv1'],self.parameter_dict['W_conv2']])
		#construct value list
		value_list = [training_loss_array,W_conv1,W_conv2,testing_loss_array]

		for file_path,value in zip(file_path_list,value_list):
			with open(self.output_root_directory + file_path,'w') as f:
				pickle.dump(value,f)
				f.close() 





my_autoencoder = Shape_Autoencoder()
my_autoencoder.build_graph()
my_autoencoder.train_graph()
