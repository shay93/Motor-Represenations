from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import png


IMG_WIDTH = 64
DIRECTORY_NAME = 'Training_Images/'
BATCH_SIZE = 1
EPOCHS = 0.005
PIXEL_DEPTH = 255.
KERNELS_LAYER_1 = 8
KERNELS_LAYER_2 = 4
OUTPUT_DIRECTORY = 'Output_Images/'
EVALUATION_SIZE =  2

def extract_data():
	#initialize numpy array to hold batch of images
	data_batch = np.zeros([BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1])
	for i in range(BATCH_SIZE):
		shape_ind = np.round(np.random.rand()*2)
		if shape_ind == 0:
			shape_str = 'Triangle'
		elif shape_ind == 1:
			shape_str = 'Rectangle'
		else:
			shape_str = 'Square'
		
		index = int(np.round(np.random.rand()*999)) 
		#read the input image and place it in in corresponding position of data_batch array
		data_batch[i,:,:,0] = (plt.imread(DIRECTORY_NAME + shape_str + str(index) + '.png'))/PIXEL_DEPTH
	return data_batch

def variable_summaries(var,name):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/'+ name,mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var-mean)))
		tf.scalar_summary('sttdev/' + name,stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name,var)

#define placeholder for input data
x = tf.placeholder('float',shape = [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]) #BATCH_SIZE
#b_size = tf.shape(x)[0]

#define parameters for first convolutional layer
with tf.name_scope("Conv_1"):
	with tf.name_scope("weights"):
		W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,KERNELS_LAYER_1],stddev = 0.1))
		variable_summaries(W_conv1,'Conv_1' + '/weights')
	with tf.name_scope("biases"):
		b_conv1 = tf.Variable(tf.constant(0.1,shape = [KERNELS_LAYER_1]))
		variable_summaries(b_conv1, 'Conv_1' + '/biases')
	
#define parameters for the second convolutional layer
with tf.name_scope("Conv_2"):
	with tf.name_scope("weights"):
		W_conv2 = tf.Variable(tf.truncated_normal([3,3,KERNELS_LAYER_1,KERNELS_LAYER_2],stddev = 0.1))
		variable_summaries(W_conv2,'Conv_2' + '/weights')
	with tf.name_scope("biases"):
		b_conv2 = tf.Variable(tf.constant(0.1,shape = [KERNELS_LAYER_2]))
		#Add summary operator for second convolutional layer
		variable_summaries(b_conv2,'Conv_2' + '/biases')

with tf.name_scope("FC1"):
#define parameters for full connected layer"
	with tf.name_scope("weights"):
		W_fc = tf.Variable(tf.truncated_normal(shape = [IMG_WIDTH*IMG_WIDTH*KERNELS_LAYER_2*BATCH_SIZE,BATCH_SIZE*IMG_WIDTH*IMG_WIDTH],stddev = 0.1)) #IMG_WIDTH*IMG_WIDTH*KERNELS_LAYER_2*BATCH_SIZE #BATCH_SIZE*IMG_WIDTH*IMG_WIDTH
		variable_summaries(W_fc,'FC1' + '/weights')
		tf.image_summary
	with tf.name_scope("biases"):
		b_fc = tf.Variable(tf.constant(0.1,shape = [BATCH_SIZE*IMG_WIDTH*IMG_WIDTH])) 
		variable_summaries(b_fc,'FC1' + '/biases')

#now build the operations for the graph
h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides = [1,1,1,1],padding = 'SAME'))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides = [1,1,1,1],padding = 'SAME'))
h_conv2_flat = tf.reshape(h_conv2, shape = [1,IMG_WIDTH*IMG_WIDTH*BATCH_SIZE*KERNELS_LAYER_2]) 
y_reshape = tf.nn.relu(tf.matmul(h_conv2_flat,W_fc) + b_fc)
y = tf.reshape(y_reshape, shape = [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]) #BATCH_SIZE
#now define output place holder and loss function
y_ = tf.placeholder('float', shape = [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]) #BATCH_SIZE
#define loss function for training

with tf.name_scope('Reduce_Mean'):
	meansq = tf.reduce_mean(tf.square(y_-y))
	tf.scalar_summary('loss',meansq)

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      EPOCHS * 3000,          			# Decay step.
      0.95,                # Decay rate.
      staircase=True)
  
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(meansq,global_step = batch)


sess = tf.Session()
merged = tf.merge_all_summaries()

def train_graph():
	train_writer = tf.train.SummaryWriter("/tmp/AutoEncoder_Log", sess.graph)
	sess.run(tf.initialize_all_variables())
	data_batch = extract_data()
	num_steps = int((EPOCHS * 3 * 1000) // BATCH_SIZE)  

	for step in range(num_steps):
		data_batch = extract_data()
		feed = {x : data_batch, y_ : data_batch}
		if step % 10 == 0:
			summary, _ = sess.run([merged,train_step], feed_dict = feed)
			train_writer.add_summary(summary,step)
		else:
			 sess.run([train_step], feed_dict = feed)
			print step,sess.run(meansq, feed_dict = feed)
			


def evaluate_graph():
	#read all the images and produce output images
	shape_str = 'Square'
	for batch_index in range(EVALUATION_SIZE // BATCH_SIZE):
		data_batch = np.zeros([BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1])
		start_point = (batch_index)*BATCH_SIZE
		end_point = (batch_index + 1)*BATCH_SIZE
		for j in range(start_point,end_point):
			data_batch[j - start_point,:,:,0] = (plt.imread(DIRECTORY_NAME + shape_str + str(j) + '.png'))/PIXEL_DEPTH
		
		output = np.array(sess.run(y,feed_dict = {x : data_batch}))*PIXEL_DEPTH
		
		
		
		for j in range(start_point,end_point):
			save_name = OUTPUT_DIRECTORY + shape_str + str(j) + '.png'
			png.from_array((output[j - start_point,:,:,0]).tolist(),'L').save(save_name)

	

train_graph()
evaluate_graph()


