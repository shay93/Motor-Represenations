from __future__ import division
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import png



#Globals
BATCH_SIZE = 12
IMG_WIDTH = 64
PIXEL_DEPTH = 255
CONV_KERNELS_1 = 32
CONV_KERNELS_2 = 64

EPOCHS = 20
DIRECTORY_NAME = 'Training_Images/'
EVALUATION_SIZE = 600


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
		self.upsample_factor = 16

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
		self.op_dict['x'] = tf.placeholder(tf.float32,shape = [self.batch_size,self.img_width,self.img_width,1])
		#define a place holder for the labels
		self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [self.batch_size,self.img_width,self.img_width,1])
		#reshape x so that you can downsample it 
		x_reshape = tf.reshape(self.op_dict['x'], shape = [self.batch_size*self.img_width,self.img_width])
		#project the input into a higher dimension first hence initialize weights
		W_fc1 = tf.Variable(tf.truncated_normal(shape = [self.img_width,self.img_width*self.upsample_factor], stddev = 0.1))
		b_fc1 = tf.Variable(tf.constant(0.1, shape = [self.batch_size*self.img_width,self.img_width*self.upsample_factor]))
		h_fc1 = tf.nn.tanh(tf.matmul(x_reshape,W_fc1) + b_fc1)
		
		#initialize a weight variable that will be used to down sample by a factor of 4
		W_fc2 = tf.Variable(tf.truncated_normal(shape = [self.img_width*self.upsample_factor,self.img_width // 16], stddev = 0.1))
		b_fc2 = tf.Variable(tf.constant(0.1, shape = [self.batch_size*self.img_width,self.img_width // 16]))
		#compute the output of the fully connected layer
		h_fc2 = tf.nn.tanh(tf.matmul(h_fc1,W_fc2) + b_fc2)
		#reshape the hidden layer so that it may be fed into a 2d convolver 
		
		h_fc2_reshape = tf.reshape(h_fc2,shape = [self.batch_size,self.img_width // 4,self.img_width // 4,1])
		#now initialize some Weights for teh convolutional layer
		self.op_dict['W_conv1'] = tf.Variable(tf.truncated_normal(shape = [2, 2, 1, CONV_KERNELS_1], stddev = 0.1))
		#initialize a bias variable for the convolutional layer 
		b_conv1 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_1]))
		conv1 = tf.nn.conv2d(h_fc2_reshape,self.op_dict['W_conv1'],strides = [1,1,1,1], padding = 'SAME')
		#now compute output from first conv kernel
		h_conv1 = tf.sigmoid(tf.nn.bias_add(conv1,b_conv1))
		
		#now pool the output of the first convolutional layer
		pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides = [1,1,1,1],padding = 'SAME')
		
		#initialize weights for second convolutional layer
		self.op_dict['W_conv2'] = tf.Variable(tf.truncated_normal(shape = [2,2,CONV_KERNELS_1,CONV_KERNELS_2] , stddev = 0.1))
		#initialize a bias variable 
		b_conv2 = tf.Variable(tf.constant(0.1,shape = [CONV_KERNELS_2]))
		#calculate the second conv layer
		conv2 = tf.nn.conv2d(pool1,self.op_dict['W_conv2'],strides = [1,1,1,1], padding = 'SAME')
		h_conv2 = tf.sigmoid((tf.nn.bias_add(conv2,b_conv2)))
		
		#pool the output from h_conv2
		pool2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
		#flatten the output of pool 2
		pool2_flat = tf.reshape(pool2, shape = [self.img_width,-1])
		#initialize weights for last fully connected layer
		W_fc3 = tf.Variable(tf.truncated_normal(shape = [self.batch_size*self.img_width*self.img_width*CONV_KERNELS_2 //(self.img_width*16),self.img_width * self.batch_size],stddev = 0.1))
		b_fc3 = tf.Variable(tf.constant(0.1,shape = [self.img_width, self.img_width*self.batch_size]))
		
		self.op_dict['y'] = tf.reshape(tf.sigmoid(tf.matmul(pool2_flat,W_fc3) + b_fc3),shape = [self.batch_size,self.img_width,self.img_width,1])
		#now define a loss for training purposes
		self.op_dict['meansq'] =  tf.reduce_mean(tf.square(self.op_dict['y_'] - self.op_dict['y']))
		#define a learning rate this may be made adaptive later but for the moment keep it fixed
		
		self.op_dict['batch'] = tf.Variable(0,trainable = False)

  		self.op_dict['learning_rate'] = tf.train.exponential_decay(1e-3,self.op_dict['batch'],200,0.9,staircase = True)
		self.op_dict['train_op'] = tf.train.AdamOptimizer(self.op_dict['learning_rate']).minimize(self.op_dict['meansq'],global_step = self.op_dict['batch'])
		#initialize a sessions object and then all variables
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		
		return sess


	def train_graph(self,sess):
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		#using the number of EPOCHS and the batch size figure out the number of
		#training steps that are required
		num_batches = int((EPOCHS * 3000) // self.batch_size)
		#num_batches_per_Epoch
		num_batches_per_Epoch = int(3000 // self.batch_size)
		#initialize a list to record the loss at each step
		loss_array = [0] * num_batches
		#iterate over the steps training at each step and recording the loss
		for batch_num in range(num_batches):
			#calculate the epoch number 
			epoch_index = batch_num // num_batches_per_Epoch + 1
			#get the data batch by specifying the batch index as step % BATCH_SIZE
			if batch_num % 100 == 0:
				#evaluate the batch and save the outputs
				self.evaluate_graph(sess,batch_num % num_batches,(batch_num + 1) % num_batches,True,epoch_index = epoch_index)
			self.op_dict['batch'].assign(batch_num)
			data_batch = extract_batch(batch_num)
			feed = {self.op_dict['x'] : data_batch , self.op_dict['y_'] : data_batch}
			loss, _,learning = sess.run([self.op_dict['meansq'],self.op_dict['train_op'],self.op_dict['learning_rate']], feed_dict = feed)
			if batch_num % 20 == 0:
				print batch_num,loss,learning
			loss_array[batch_num] = loss
		return loss_array

	def evaluate_graph(self,sess,start_batch_index,end_batch_index,checkpoint_boolean,epoch_index = None):
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
		for batch_index in range(start_batch_index,end_batch_index):
			#call on the batch generator
			data_batch = extract_batch(batch_index)
			output = np.array(sess.run(self.op_dict['y'],feed_dict = {self.op_dict['x'] : data_batch}))

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

	
	def save_normalized_weights(self,sess):
		"""
		Takes an input of weights and saves them as images so that training may be observed
		inputs: A sessions object to evaluate the weights
		"""
		W_conv1,W_conv2 = sess.run([self.op_dict['W_conv1'],self.op_dict['W_conv2']])
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



my_autoencoder = Shape_Autoencoder()
sess = my_autoencoder.build_graph()
loss = my_autoencoder.train_graph(sess)
f = plt.figure()
plt.title("Loss")
plt.plot(loss)
f.savefig("Image_Autoencoder_Ver1_Outputs/Loss_Array.png")
plt.close(f)
my_autoencoder.evaluate_graph(sess,0,3000 // BATCH_SIZE,False)
my_autoencoder.save_normalized_weights(sess)

