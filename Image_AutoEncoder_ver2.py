from __future__ import division 
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
FC_2_UNITS = 64*64*5

EPOCHS = 20
DIRECTORY_NAME = 'Training_Images/'
OUTPUT_DIRECTORY = 'Output_Images_ver2/'
EVALUATION_SIZE = 200


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
		self.conv_kernels_3 = 32
		self.conv_kernels_4 = 64
		self.op_dict = {}
	

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


		self.op_dict['W_conv1'] = tf.Variable(tf.truncated_normal([10,10,1,self.conv_kernels_1],stddev = 0.1))
		b_conv1 = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_1]))
		
			
		conv1 = tf.nn.conv2d(self.op_dict['x'],self.op_dict['W_conv1'],strides = [1,1,1,1],padding = 'SAME')
		h_conv1 = tf.sigmoid(tf.nn.bias_add(conv1,b_conv1))
		pool1 = tf.nn.max_pool(h_conv1, ksize =[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
	
		#define parameters for the second convolutional layer
		W_conv2 = tf.Variable(tf.truncated_normal([10,10,self.conv_kernels_1,self.conv_kernels_2],stddev = 0.1))
		b_conv2 = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_2]))

		#consider second layer
		conv2 = tf.nn.conv2d(pool1,W_conv2,strides = [1,1,1,1],padding = 'SAME')
		h_conv2 = tf.sigmoid(tf.nn.bias_add(conv2,b_conv2))
		pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1],strides = [1,1,1,1], padding = 'SAME')
		#Reshape the output from pooling layers to pass to fully connected layers
		h_conv2_reshape = tf.reshape(pool2, shape = [self.img_width*self.img_width, self.batch_size*self.conv_kernels_2 // 4])

		#define parameters for full connected layer"
		W_fc1 = tf.Variable(tf.truncated_normal(shape = [self.batch_size*self.conv_kernels_2 // 4,FC_2_UNITS],stddev = 0.1)) 
		b_fc1 = tf.Variable(tf.constant(0.1,shape = [self.img_width*self.img_width,FC_2_UNITS])) 
		h_fc1 = tf.sigmoid(tf.matmul(h_conv2_reshape,W_fc1) + b_fc1)

		W_fc2 = tf.Variable(tf.truncated_normal(shape = [FC_2_UNITS,self.batch_size],stddev = 0.1))
		b_fc2 = tf.Variable(tf.constant(0.1,shape = [self.img_width*self.img_width,self.batch_size]))
		#Add the final layer 
		h_fc2 = tf.sigmoid(tf.matmul(h_fc1,W_fc2) + b_fc2)

		#reshape h_fc2
		h_fc2_reshape = tf.reshape(h_fc2,shape = [self.img_width*self.img_width // 16, self.batch_size*16])
		W_fc3 = tf.Variable(tf.truncated_normal(shape = [self.batch_size*16,self.batch_size],stddev = 0.1))
		b_fc3 = tf.Variable(tf.constant(0.1, shape = [self.img_width*self.img_width // 16,self.batch_size]))
		h_fc3 = tf.sigmoid(tf.matmul(h_fc2_reshape,W_fc3) + b_fc3)

		#now reshape such that it may be used by the other convolutional layers
		h_fc3_reshape = tf.reshape(h_fc3, shape = [self.batch_size,self.img_width // 4, self.img_width // 4, 1])
		#pass this into a set of convolutional layers
		W_conv3 = tf.Variable(tf.truncated_normal(shape = [2, 2, 1, self.conv_kernels_3], stddev = 0.1))
		#initialize a bias variable for the convolutional layer 
		b_conv3 = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_3]))
		conv3 = tf.nn.conv2d(h_fc3_reshape,W_conv3,strides = [1,1,1,1], padding = 'SAME')
		#now compute output from first conv kernel
		h_conv3 = tf.sigmoid(tf.nn.bias_add(conv3,b_conv3))
		
		#now pool the output of the first convolutional layer
		pool3 = tf.nn.max_pool(h_conv3,ksize = [1,2,2,1],strides = [1,1,1,1],padding = 'SAME')
		
		#initialize weights for second convolutional layer
		W_conv4 = tf.Variable(tf.truncated_normal(shape = [2,2,self.conv_kernels_3,self.conv_kernels_4] , stddev = 0.1))
		#initialize a bias variable 
		b_conv4 = tf.Variable(tf.constant(0.1,shape = [self.conv_kernels_4]))
		#calculate the second conv layer
		conv4 = tf.nn.conv2d(pool3,W_conv4,strides = [1,1,1,1], padding = 'SAME')
		h_conv4 = tf.sigmoid((tf.nn.bias_add(conv4,b_conv4)))
		
		#pool the output from h_conv2
		pool4 = tf.nn.max_pool(h_conv4,ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
		#flatten the output of pool 2
		pool4_flat = tf.reshape(pool4, shape = [self.img_width,-1])
		#initialize weights for last fully connected layer
		W_fc5 = tf.Variable(tf.truncated_normal(shape = [self.batch_size*self.img_width*self.img_width*CONV_KERNELS_2 //(self.img_width*16),self.img_width * self.batch_size],stddev = 0.1))
		b_fc5 = tf.Variable(tf.constant(0.1,shape = [self.img_width, self.img_width*self.batch_size]))

		self.op_dict['y'] = tf.reshape(tf.sigmoid(tf.matmul(pool4_flat,W_fc5) + b_fc5),shape = [self.batch_size,self.img_width,self.img_width,1])

		
		#define a loss function 
		self.op_dict['meansq'] = tf.reduce_mean(tf.square(self.op_dict['y'] - self.op_dict['y_']))
		
		#define a learning rate with an exponential decay,a batch variable is needed in order to prevent 
		self.op_dict['batch'] = tf.Variable(0,trainable = False)
		self.op_dict['learning_rate'] = tf.train.exponential_decay(
      				1.,                		# Base learning rate.
      				self.op_dict['batch'],  	# Current index into the dataset.
      				200,      		# Decay step.
      				0.5,             			# Decay rate.
      				staircase=True)
		
		#define a training operation
		self.op_dict['train_op'] = tf.train.MomentumOptimizer(self.op_dict['learning_rate'],1.).minimize(self.op_dict['meansq'],global_step = self.op_dict['batch'])
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		
		return sess


	def train_graph(self,sess,test_data_index):
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		#using the number of EPOCHS and the batch size figure out the number of
		#training steps that are required
		num_batches = int((EPOCHS * 3000) // self.batch_size)
		#initialize a list to record the loss at each step
		loss_array = [0] * num_batches
		#iterate over the steps training at each step and recording the loss
		for batch_num in range(num_batches):
			#get the data batch by specifying the batch index as step % BATCH_SIZE
			self.op_dict['batch'].assign(batch_num)
			data_batch = extract_batch(batch_num)
			feed = {self.op_dict['x'] : data_batch , self.op_dict['y_'] : data_batch}
			loss, _,learning = sess.run([self.op_dict['meansq'],self.op_dict['train_op'],self.op_dict['learning_rate']], feed_dict = feed)
			if batch_num % 20 == 0:
				print batch_num,loss,learning
			loss_array[batch_num] = loss
		return loss_array

	def evaluate_graph(self,sess,test_data_index):
		"""
		Pass the testing data through the graph and save the output image for each input image
		to the output image directory.
		input: test_data_index is an integer specifying what the starting index of the test data is
				a sess object is needed to evaluate the graph and the op_dict provides the nodes
		output: -
		"""
		shape_str_array = ['Rectangle', 'Square', 'Triangle']
		#iterate over the batches
		for batch_index in range(EVALUATION_SIZE // self.batch_size):
			#call on the batch generator
			data_batch = extract_batch(batch_index)
			output = np.array(sess.run(self.op_dict['y'],feed_dict = {self.op_dict['x'] : data_batch}))

			for j in range(self.batch_size):
				#in order to separate the batch into its separate shapes we need
				#the j mod 4 gives the index of the 
				shape_str = shape_str_array[j // (self.batch_size // 3)]
				#once the shape name is known determine the shape index
				shape_index = j%(self.batch_size // 3) + batch_index*(self.batch_size // 3)
				save_name = OUTPUT_DIRECTORY + shape_str + str(shape_index) + '.png'
				temp = output[j,:,:,0] * PIXEL_DEPTH
				png.from_array((temp).tolist(),'L').save(save_name)


	def save_normalized_weights(self,sess):
		"""
		Takes an input of weights and saves them as images so that training may be observed
		inputs:  
		"""
		W_conv1 = sess.run(self.op_dict['W_conv1'])
		for i in range(self.conv_kernels_1):
			kernel = W_conv1[:,:,0,i]
			kernel_normed = np.divide(kernel,np.mean(kernel))
			plt.imsave("Kernels_Ver2/" + "kernel" + str(i) + ".png",kernel_normed)



test_data_index = 220
my_autoencoder = Shape_Autoencoder()
sess = my_autoencoder.build_graph()
loss = my_autoencoder.train_graph(sess,test_data_index)
my_autoencoder.evaluate_graph(sess,test_data_index)
my_autoencoder.save_normalized_weights(sess)

