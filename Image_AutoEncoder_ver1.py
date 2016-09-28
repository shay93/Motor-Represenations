from __future__ import division
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import png



#Globals
BATCH_SIZE = 48
IMG_WIDTH = 64
PIXEL_DEPTH = 255
CONV_KERNELS_1 = 32
CONV_KERNELS_2 = 64

EPOCHS = 60
DIRECTORY_NAME = 'Training_Images/'

#define a datasplit ratio to separate training data from validation data
data_split_ratio = 0.8

def extract_batch(batch_num):
	"""
	Reads into directory containing training images and reads a number of them (specified by batch size)
	into a numpy array so that it may be then be fed into the tensorflow graph
	inputs: batch_num an integer specifying what the batch number is
	outputs:data_batch a 4d numpy array of size [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]
	"""
	#initialize numpy array to hold batch of images
	data_batch = np.zeros([BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1])
	shape_str_array = ['Rectangle', 'Square', 'Triangle']
	#the number of batches in one Epoch may be used to calculate a batch index
	batches_in_Epoch = (len(shape_str_array) * 1000) // BATCH_SIZE
	#batch index is used to index the batch wrt to the data available
	batch_index = batch_num % batches_in_Epoch
	#it is also useful to know the number of each shape per batch since this is the same for all shapes just call it squares per batch
	squares_per_batch = BATCH_SIZE // len(shape_str_array)
	for j,shape_str in enumerate(shape_str_array):
		for i in range(squares_per_batch):
			shape_index = batch_index*squares_per_batch + i
			data_batch[i + squares_per_batch*j,:,:,0] = plt.imread(DIRECTORY_NAME + shape_str + str(shape_index) + '.png')

	return data_batch


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
		self.upsample_factor = 20
		self.parameter_dict = {}
		self.dropout_prob = 0.8

		#initialize some directory names
		self.checkpoint_images_directory = "Image_Checkpoints/"
		self.output_root_directory = "Image_Autoencoder_Ver1_Outputs/"
		self.output_image_directory = self.output_root_directory + "Output_Images/"

	def build_graph(self):
		"""
		Responsible for defining the operations that comprise the graph
		inputs: -IMG_WIDTH = 64

		outputs: A session objecta and a operation dictionary
		"""
		#first specify a placeholder for the input image which is of size 64 by 64
		with tf.name_scope('Input_placeholder') as scope:
			self.op_dict['x'] = tf.placeholder(tf.float32,shape = [self.batch_size,self.img_width,self.img_width,1])
		
		#define a place holder for the outputs
		with tf.name_scope('Output_placeholder') as scope:
			self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [self.batch_size,self.img_width,self.img_width,1])
		
		with tf.name_scope('Reshape_x') as scope:
			#reshape x so that you can downsample it 
			x_reshape = tf.reshape(self.op_dict['x'], shape = [self.batch_size,self.img_width*self.img_width])
		
		with tf.name_scope("FC1") as scope:	
			#project the input into a higher dimension first hence initialize weights
			with tf.name_scope("Weights") as scope:
				self.parameter_dict['W_fc1'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width,self.img_width*self.img_width*self.upsample_factor], stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_fc1'] = tf.Variable(tf.constant(0., shape = [self.batch_size,self.img_width*self.img_width*self.upsample_factor]))
			with tf.name_scope("Activations") as scope:
				h_fc1 = tf.nn.relu(tf.matmul(x_reshape,self.parameter_dict['W_fc1']) + self.parameter_dict['b_fc1'])
				print h_fc1


		#add a dropout layer between FC1 and FC2

		with tf.name_scope("Dropout") as scope:
			h_dropout = tf.nn.dropout(h_fc1,self.dropout_prob)
			print h_dropout
		
		with tf.name_scope("FC2") as scope:
			with tf.name_scope("Weights") as scope:
				#initialize a weight variable that will be used to down sample by a factor of 4
				self.parameter_dict['W_fc2'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width*self.upsample_factor,self.img_width*self.img_width // 16], stddev = 0.1))
			with tf.name_scope("Biases") as scope:
				self.parameter_dict['b_fc2'] = tf.Variable(tf.constant(0., shape = [self.batch_size,self.img_width*self.img_width // 16]))
			with tf.name_scope("Activations") as scope:
				#compute the output of the fully connected layer
				h_fc2 = tf.nn.relu(tf.matmul(h_dropout,self.parameter_dict['W_fc2']) + self.parameter_dict['b_fc2'])
		
		with tf.name_scope("Reshape_h_fc1") as scope:
			#reshape the hidden layer so that it may be fed into a 2d convolver 
			h_fc2_reshape = tf.reshape(h_fc2,shape = [self.batch_size,self.img_width // 4,self.img_width // 4,1])
		
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
			pool2_flat = tf.reshape(pool2, shape = [self.batch_size,-1])
		
		with tf.name_scope("FC3") as scope:
			with tf.name_scope("Weights") as scope:
				#initialize weights for last fully connected layer
				self.parameter_dict['W_fc3'] = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.img_width*self.conv_kernels_2 // (16*16), self.img_width * self.img_width],stddev = 0.1))
			with tf.name_scope("Biases") as scope:		
				self.parameter_dict['b_fc3'] = tf.Variable(tf.constant(0.,shape = [self.batch_size, self.img_width*self.img_width]))
			with tf.name_scope("Activations") as scope:
				self.op_dict['y_not_normed'] = tf.reshape(tf.nn.relu(tf.matmul(pool2_flat,self.parameter_dict['W_fc3']) + self.parameter_dict['b_fc3']),shape = [self.batch_size,self.img_width,self.img_width,1])
		
		with tf.name_scope("y") as scope:
			#first calculate the max of each batch 
			max_batch_element = tf.reduce_max(self.op_dict['y_not_normed'],[1,2,3])
			print max_batch_element
			#reshape max_batch element
			max_batch_element_reshape = tf.reshape(max_batch_element, shape = [self.batch_size,1,1,1])
			#tile the max batch element so that an element to element division may be carried out
			max_batch_tiled = tf.tile(max_batch_element_reshape,[1,self.img_width,self.img_width,1])
			#take the norm of y and set as the output
			self.op_dict['y'] = tf.div(self.op_dict['y_not_normed'],max_batch_tiled)
		
		with tf.name_scope("loss") as scope:
			#now define a loss for training purposes
			self.op_dict['L1_Norm'] =  tf.reduce_mean(tf.square(self.op_dict['y_'] - self.op_dict['y']))
		
		#define a learning rate this may be made adaptive later but for the moment keep it fixed
		self.op_dict['learning_rate'] = 1e-4
		
		with tf.name_scope("Train") as scope:
			self.op_dict['train_op'] = tf.train.AdamOptimizer(self.op_dict['learning_rate']).minimize(self.op_dict['L1_Norm'])
		
		#add the tensorboard ops
		self.Add_Tensorboard_ops()

		#initialize a sessions object and then all variables
		sess = tf.Session()
		log_dir = self.output_root_directory + "/tmp/summary_logs"
		self.op_dict['train_writer'] = tf.train.SummaryWriter(log_dir, sess.graph)
		#initialize another operation which comprises the test time summary writer
		self.op_dict['test_writer'] = tf.train.SummaryWriter(log_dir, sess.graph)
		sess.run(tf.initialize_all_variables())
		
		return sess


	def train_graph(self,sess,data_split_ratio):
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		#using the number of EPOCHS and the batch size figure out the number of
		#training steps that are required
		num_training_batches = int((EPOCHS * 3000 * data_split_ratio) // self.batch_size)
		#num_batches_per_Epoch
		num_training_batches_per_Epoch = int((3000 * data_split_ratio) // (self.batch_size))
		#initialize a list to record the loss at each step
		training_loss_array = [0] * num_training_batches
		#iterate over the steps training at each step and recording the loss
		for batch_num in range(num_training_batches):
			#calculate the epoch number 
			epoch_index = batch_num // num_training_batches_per_Epoch + 1
			#get the data batch by specifying the batch index as step % BATCH_SIZE
			if batch_num % 100 == 0:
				#evaluate the batch and save the outputs
				batch_index = batch_num % num_training_batches_per_Epoch
				self.evaluate_graph(sess,batch_index,(batch_num + 1) % num_training_batches_per_Epoch, checkpoint_boolean = True ,epoch_index = epoch_index)

			data_batch = extract_batch(batch_num)
			feed = {self.op_dict['x'] : data_batch , self.op_dict['y_'] : data_batch}
			loss, _,summary = sess.run([self.op_dict['L1_Norm'],self.op_dict['train_op'],self.op_dict['merged']], feed_dict = feed)
			if batch_num % 20 == 0:
				self.op_dict['train_writer'].add_summary(summary,batch_num)
				print batch_num,loss
			training_loss_array[batch_num] = loss
		return training_loss_array

	def evaluate_graph(self,sess,start_batch_index,end_batch_index,checkpoint_boolean = False,epoch_index = None):
		"""
		Pass the testing data through the graph and save the output image for each input image
		to the output image directory.
		input: start_batch_index and end_batch_index indicate start and end point for evaluations
				checkpoint_boolean specifies if the evaluation is being used for checkpointing. 
		output: -
		"""
		shape_str_array = ['Rectangle', 'Square', 'Triangle']
		#iterate over the batches
		if checkpoint_boolean:
			output_directory = self.output_root_directory + self.checkpoint_images_directory
		else:
			output_directory = self.output_image_directory
			#initialize a loss array as well if not checkpointing and evaluating
			testing_loss_array = [0] * int(end_batch_index - start_batch_index)
		
		for batch_index in range(start_batch_index,end_batch_index):
			#call on the batch generator
			data_batch = extract_batch(batch_index)
			output,testing_loss,summary = np.array(sess.run([self.op_dict['y'],self.op_dict['L1_Norm'],self.op_dict['merged']],feed_dict = {self.op_dict['x'] : data_batch,self.op_dict['y_'] : data_batch}))

			for j in range(self.batch_size):
				#in order to separate the batch into its separate shapes we need
				#the j mod 4 gives the index of the 
				shape_str = shape_str_array[j // (self.batch_size // 3)]
				#once the shape name is known determine the shape index
				shape_index = j%(self.batch_size // 3) + batch_index*(self.batch_size // 3)
				if checkpoint_boolean:
					save_name = output_directory + shape_str + str(shape_index) + "_epoch_" + str(epoch_index) + '.png'
				else:
					save_name = output_directory + shape_str + str(shape_index) + '.png'					
				temp = output[j,:,:,0] * PIXEL_DEPTH
				png.from_array((temp).tolist(),'L').save(save_name)

			if not(checkpoint_boolean):
				#i.e if evaluating then append to the testing loss array
				testing_loss_array[batch_index - start_batch_index] = testing_loss

		if not(checkpoint_boolean):
			return testing_loss_array

	
	def save_normalized_weights(self,sess):
		"""
		Takes an input of weights and saves them as images so that training may be observed
		inputs: A sessions object to evaluate the weights
		"""
		W_conv1,W_conv2 = sess.run([self.parameter_dict['W_conv1'],self.parameter_dict['W_conv2']])
		#initialize a figure to store the images
		conv1_fig = plt.figure(1,(20.,20.))
		#initialize an image grid
		conv1_grid = ImageGrid(conv1_fig, 111,nrows_ncols=(self.conv_kernels_1 // 8,8),axes_pad = 0.1) 
		for i in range(self.conv_kernels_1):
			kernel = W_conv1[:,:,0,i]
			kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
			conv1_grid[i].imshow(kernel_normed, cmap = "Greys_r")
		
		conv1_fig.savefig("Image_Autoencoder_Ver1_Outputs/Conv1_Kernels.png")
		plt.close(conv1_fig)
		#perform the above for second conv layer as well
		conv2_fig = plt.figure(1,(20.,20.))
		conv2_grid = ImageGrid(conv2_fig, 111,nrows_ncols=(self.conv_kernels_2 // 8,8),axes_pad = 0.1)
		for j in range(self.conv_kernels_2):
			kernel = W_conv2[:,:,0,j]
			kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
			conv2_grid[j].imshow(kernel_normed,cmap = "Greys_r")

		conv2_fig.savefig("Image_Autoencoder_Ver1_Outputs/Conv2_Kernels.png")
		plt.close(conv2_fig)


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
		file_path_list = ["testing_loss.npy","training_loss.npy","W_conv1.npy","W_conv2.npy"]
		#evaluate the weight tensors
		W_conv1,W_conv2 = sess.run([my_autoencoder.parameter_dict['W_conv1'],my_autoencoder.parameter_dict['W_conv2']])
		#construct value list
		value_list = [testing_loss_array,training_loss_array,W_conv1,W_conv2]

		for file_path,value in zip(file_path_list,value_list):
			with open(self.output_root_directory + file_path,'w') as f:
				pickle.dump(value,f)
				f.close() 





my_autoencoder = Shape_Autoencoder()
sess = my_autoencoder.build_graph()
training_loss_array = my_autoencoder.train_graph(sess,data_split_ratio)
testing_index_start = int((3000 * data_split_ratio) // BATCH_SIZE)
end_index = int(3000 // BATCH_SIZE)
testing_loss_array = my_autoencoder.evaluate_graph(sess,testing_index_start,end_index)
my_autoencoder.save_as_npy(sess,training_loss_array,testing_loss_array)


