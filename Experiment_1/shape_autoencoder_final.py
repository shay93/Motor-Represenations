from __future__ import division
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import png



#Globals
BATCH_SIZE = 10
IMAGE_SIZE = 64
PIXEL_DEPTH = 255
CONV_KERNELS_1 = 4
CONV_KERNELS_2 = 8
EVAL_BATCH_SIZE = 10
EPOCHS = 1
EVAL_FREQUENCY = 5
NUM_CHANNELS = 1
VALIDATION_SIZE =5
shape_str_array = ['Rectangle', 'Square', 'Triangle']
ROOT_DIR = "Image_AutoEncoder_Ver1_Outputs/"


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


train_data = extract_data("Training_Images/", 60)

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
		self.img_width = IMG_WIDTH
		self.conv_kernels_1 = CONV_KERNELS_1
		self.conv_kernels_2 = CONV_KERNELS_2
		self.op_dict = {}
		self.upsample_factor = 4
		self.parameter_dict = {}
		self.dropout_prob = 0.8

		#initialize some directory names
		self.output_root_directory = "Image_Autoencoder_Ver1_Outputs/"
		self.output_image_directory = self.output_root_directory + "Output_Images/"

	def build_graph(self):
		"""
		Responsible for defining the operations that comprise the graph
		inputs: -IMG_WIDTH = 64

		outputs: A session objecta and a operation dictionary
		"""
		#first specify a placeholder for the input image which is of size 64 by 64

		self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None,self.img_width,self.img_width,1])
		
		#define a place holder for the outputs
		self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [None,self.img_width,self.img_width,1])
		
		 
		x_reshape = tf.reshape(self.op_dict['x'], shape = [-1,self.img_width*self.img_width])
		
		with tf.name_scope("FC1") as scope:	
			#project the input into a higher dimension first hence initialize weights
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_fc1'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width,self.img_width*self.img_width*self.upsample_factor], stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_fc1'] = tf.Variable(tf.constant(0., shape = [self.img_width*self.img_width*self.upsample_factor]))
			with tf.name_scope("Activations") as scope:
				h_fc1 = tf.nn.relu(tf.matmul(x_reshape,self.parameter_dict['W_fc1']) + self.parameter_dict['b_fc1'])


		#add a dropout layer between FC1 and FC2
		with tf.name_scope("Dropout") as scope:
			h_dropout = tf.nn.dropout(h_fc1,self.dropout_prob)
		
		with tf.name_scope("FC2") as scope:
			with tf.name_scope("Weights") as scope:
				#initialize a weight variable that will be used to down sample by a factor of 4
				self.parameter_dict['W_fc2'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width*self.upsample_factor,self.img_width*self.img_width // 16], stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_fc2'] = tf.Variable(tf.constant(0., shape = [self.img_width*self.img_width // 16]))
			with tf.name_scope("Activations") as scope:
				#compute the output of the fully connected layer
				h_fc2 = tf.nn.relu(tf.matmul(h_dropout,self.parameter_dict['W_fc2']) + self.parameter_dict['b_fc2'])
		
		with tf.name_scope("Reshape_h_fc1") as scope:
			#reshape the hidden layer so that it may be fed into a 2d convolver 
			h_fc2_reshape = tf.reshape(h_fc2,shape = [-1,self.img_width // 4,self.img_width // 4,1])
		
		with tf.name_scope("Conv1") as scope:	
			with tf.name_scope("Weights"):
				#now initialize some Weights for the convolutional layer
				self.parameter_dict['W_conv1'] = tf.Variable(tf.truncated_normal(shape = [2, 2, 1, self.conv_kernels_1], stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				#initialize a bias variable for the convolutional layer 
				self.parameter_dict['b_conv1'] = tf.Variable(tf.constant(0.,shape = [self.conv_kernels_1]))
			with tf.name_scope("Conv_Output") as scope:			
				conv1 = tf.nn.conv2d(h_fc2_reshape,self.parameter_dict['W_conv1'],strides = [1,1,1,1], padding = 'SAME')
			with tf.name_scope("Activations") as scope:
				#now compute output from first conv kernel
				h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1,self.parameter_dict['b_conv1']))
		
		with tf.name_scope("Pool1") as scope:
			#now pool the output of the first convolutional layer
			pool1 = tf.nn.max_pool(h_conv1,ksize = [1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
		
		with tf.name_scope("Conv2") as scope:
			with tf.name_scope("Weights") as scope:
				#initialize weights for second convolutional layer
				self.parameter_dict['W_conv2'] = tf.Variable(tf.truncated_normal(shape = [2,2,self.conv_kernels_1,self.conv_kernels_2] , stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				#initialize a bias variable 
				self.parameter_dict['b_conv2'] = tf.Variable(tf.constant(0.,shape = [self.conv_kernels_2]))
			with tf.name_scope("Conv_Output") as scope:
				#calculate the second conv layer
				conv2 = tf.nn.conv2d(pool1,self.parameter_dict['W_conv2'],strides = [1,1,1,1], padding = 'SAME')
			with tf.name_scope("Activations") as scope:
				h_conv2 = tf.nn.relu((tf.nn.bias_add(conv2,self.parameter_dict['b_conv2'])))
		
		with tf.name_scope("Pool2") as scope:
			#pool the output from h_conv2
			pool2 = tf.nn.max_pool(h_conv2,ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
		
		with tf.name_scope("Pool2_Flat") as scope:
			#flatten the output of pool 2
			pool2_flat = tf.reshape(pool2, shape = [-1,self.img_width*self.img_width*self.conv_kernels_2 // (16*16)])
		
		with tf.name_scope("FC3") as scope:
			with tf.name_scope("Weights") as scope:
				#initialize weights for last fully connected layer
				self.parameter_dict['W_fc3'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width*self.conv_kernels_2 // (16*16), self.img_width * self.img_width],stddev = 0.1))
			with tf.name_scope("Biases") as scope:		
				self.parameter_dict['b_fc3'] = tf.Variable(tf.constant(0.,shape = [self.img_width*self.img_width]))
			with tf.name_scope("Activations") as scope:
				self.op_dict['y_not_normed'] = tf.reshape(tf.nn.relu(tf.matmul(pool2_flat,self.parameter_dict['W_fc3']) + self.parameter_dict['b_fc3']),shape = [-1,self.img_width,self.img_width,1])
		
		with tf.name_scope("y") as scope:
			#first calculate the max of each batch 
			max_batch_element = tf.reduce_max(self.op_dict['y_not_normed'],[1,2,3])
			#reshape max_batch element
			max_batch_element_reshape = tf.reshape(max_batch_element, shape = [-1,1,1,1])
			#tile the max batch element so that an element to element division may be carried out
			max_batch_tiled = tf.tile(max_batch_element_reshape,[1,self.img_width,self.img_width,1])
			#take the norm of y and set as the output
			self.op_dict['y'] = tf.div(self.op_dict['y_not_normed'],max_batch_tiled)
		
		with tf.name_scope("loss") as scope:
			#now define a loss for training purposes
			self.op_dict['meansq'] =  tf.reduce_mean(tf.square(self.op_dict['y_'] - self.op_dict['y']))
		
		#define a learning rate this may be made adaptive later but for the moment keep it fixed
		self.op_dict['learning_rate'] = 1e-4
		
		with tf.name_scope("Train") as scope:
			self.op_dict['train_op'] = tf.train.AdamOptimizer(self.op_dict['learning_rate']).minimize(self.op_dict['meansq'])
		
		#add the tensorboard ops
		#self.Add_Tensorboard_ops()



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
			loss_array = [] * (int(num_epochs * train_size) // BATCH_SIZE)
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
					predictions = self.eval_in_batches(validation_data, sess)
					self.unwrap_eval_prediction(predictions,step // EVAL_FREQUENCY)
					print step,l

			return loss_array,sess



	def eval_in_batches(self,data,sess):
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


	def unwrap_eval_prediction(self,predictions,eval_num):
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


	def save_as_npy(self,sess,training_loss_array):
		"""
		Saves the training loss and evaluation loss as an npy in addition to the weights prescribed as W_conv1 and W_conv2
		inputs: training_loss and testing_loss are both numpy arrays  
		"""
		file_path_list = ["training_loss.npy","W_conv1.npy","W_conv2.npy"]
		#evaluate the weight tensors
		W_conv1,W_conv2 = sess.run([self.parameter_dict['W_conv1'],self.parameter_dict['W_conv2']])
		#construct value list
		value_list = [training_loss_array,W_conv1,W_conv2]

		for file_path,value in zip(file_path_list,value_list):
			with open(self.output_root_directory + file_path,'w') as f:
				pickle.dump(value,f)
				f.close() 





my_autoencoder = Shape_Autoencoder()
#my_autoencoder.build_graph()
#training_loss_array = my_autoencoder.train_graph()

#my_autoencoder.save_as_npy(sess,training_loss_array)


