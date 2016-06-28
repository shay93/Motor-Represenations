from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import png


IMG_WIDTH = 64
DIRECTORY_NAME = 'Training_Images/'
BATCH_SIZE = 2
EPOCHS = 30
PIXEL_DEPTH = 255.
KERNELS_LAYER_1 = 20
KERNELS_LAYER_2 = 4
FC_2_UNITS = 64*64*2*BATCH_SIZE
OUTPUT_DIRECTORY = 'Output_Images/'
EVALUATION_SIZE =  200

def extract_data():
	#initialize numpy array to hold batch of images
	data_batch = np.zeros([BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1])
	for i in range(BATCH_SIZE):
		shape_ind = np.round(np.random.rand())
		if shape_ind == 0:
			shape_str = 'Square'
		elif shape_ind == 1:
			shape_str = 'Rectangle'
		else:
			shape_str = 'Square'
		
		index = int(np.round(np.random.rand()*999)) 
		#read the input image and place it in in corresponding position of data_batch array
		data_batch[i,:,:,0] = (plt.imread(DIRECTORY_NAME + shape_str + str(index) + '.png'))/(PIXEL_DEPTH)
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
		W_conv1 = tf.Variable(tf.truncated_normal([10,10,1,KERNELS_LAYER_1],stddev = 0.1))
		variable_summaries(W_conv1,'Conv_1' + '/weights')
	with tf.name_scope("biases"):
		b_conv1 = tf.Variable(tf.constant(0.1,shape = [KERNELS_LAYER_1]))
		variable_summaries(b_conv1, 'Conv_1' + '/biases')
	
#define parameters for the second convolutional layer
with tf.name_scope("Conv_2"):
	with tf.name_scope("weights"):
		W_conv2 = tf.Variable(tf.truncated_normal([10,10,KERNELS_LAYER_1,KERNELS_LAYER_2],stddev = 0.1))
		variable_summaries(W_conv2,'Conv_2' + '/weights')
	with tf.name_scope("biases"):
		b_conv2 = tf.Variable(tf.constant(0.1,shape = [KERNELS_LAYER_2]))
		#Add summary operator for second convolutional layer
		variable_summaries(b_conv2,'Conv_2' + '/biases')

with tf.name_scope("FC1"):
#define parameters for full connected layer"
	with tf.name_scope("weights"):
		W_fc1 = tf.Variable(tf.truncated_normal(shape = [IMG_WIDTH *IMG_WIDTH // 4 *KERNELS_LAYER_2*BATCH_SIZE,FC_2_UNITS],stddev = 0.1)) #IMG_WIDTH*IMG_WIDTH*KERNELS_LAYER_2*BATCH_SIZE #BATCH_SIZE*IMG_WIDTH*IMG_WIDTH
		variable_summaries(W_fc1,'FC1' + '/weights')
		tf.image_summary
	with tf.name_scope("biases"):
		b_fc1 = tf.Variable(tf.constant(0.1,shape = [FC_2_UNITS])) 
		variable_summaries(b_fc1,'FC1' + '/biases')

with tf.name_scope("FC2"):
	with tf.name_scope("weights"):
		W_fc2 = tf.Variable(tf.truncated_normal(shape = [FC_2_UNITS,BATCH_SIZE*IMG_WIDTH*IMG_WIDTH],stddev = 0.1))
	with tf.name_scope("biases"):
		b_fc2 = tf.Variable(tf.constant(0.1,shape = [BATCH_SIZE*IMG_WIDTH*IMG_WIDTH]))

#now build the operations for the graph
#consider first layer
conv1 = tf.nn.conv2d(x,W_conv1,strides = [1,1,1,1],padding = 'SAME')
h_conv1 = tf.sigmoid(tf.nn.bias_add(conv1,b_conv1))
pool1 = tf.nn.max_pool(h_conv1, ksize =[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
#now consider second layer
conv2 = tf.nn.conv2d(pool1,W_conv2,strides = [1,1,1,1],padding = 'SAME')
h_conv2 = tf.sigmoid(tf.nn.bias_add(conv2,b_conv2))
pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1],strides = [1,1,1,1], padding = 'SAME')
#Reshape the output from pooling layers to pass to fully connected layers
h_conv2_flat = tf.reshape(pool2, shape = [1,IMG_WIDTH*IMG_WIDTH // 4*BATCH_SIZE*KERNELS_LAYER_2])
h_fc1 = tf.sigmoid(tf.matmul(h_conv2_flat,W_fc1) + b_fc1)
#Add the final layer 
y_reshape = tf.sigmoid(tf.matmul(h_fc1,W_fc2) + b_fc2)
y = tf.reshape(y_reshape, shape = [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]) #BATCH_SIZE
#now define output place holder and loss function
y_ = tf.placeholder('float', shape = [BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1]) #BATCH_SIZE
#define loss function for training


with tf.name_scope('Reduce_Mean'):
	loss = tf.reduce_mean(tf.square(tf.square(y_-y))) + (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))*1e-9
	tf.scalar_summary('loss',loss)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices = [1]))

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
      10.,                # Base learning rate.
      batch*BATCH_SIZE,  # Current index into the dataset.
      EPOCHS * 3000,          			# Decay step.
      1e-1,                # Decay rate.
      staircase=True)
  
with tf.name_scope('train'):
	train_step = tf.train.MomentumOptimizer(learning_rate,0.5).minimize(loss)


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
			batch.assign(step)
			sess.run(train_step,feed_dict = feed)		
			#summary, _ = sess.run([merged,train_step], feed_dict = feed)
			#train_writer.add_summary(summary,step)
		else:
			batch.assign(step)
			sess.run([train_step], feed_dict = feed)
			print "For step %d the loss was" %(step),sess.run(loss, feed_dict = feed)
			#print "The learning rate was %f" %(sess.run(learning_rate, feed_dict = feed))

			


def evaluate_graph():
	#read all the images and produce output images
	string_list = ['Square','Rectangle','Triangle']
	for shape_str in string_list:
		for batch_index in range(EVALUATION_SIZE // BATCH_SIZE):
			data_batch = np.zeros([BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,1])
			start_point = (batch_index)*BATCH_SIZE
			end_point = (batch_index + 1)*BATCH_SIZE
			for j in range(start_point,end_point):
				data_batch[j - start_point,:,:,0] = (plt.imread(DIRECTORY_NAME + shape_str + str(j) + '.png'))/PIXEL_DEPTH
		
			output = np.array(sess.run(y,feed_dict = {x : data_batch}))*PIXEL_DEPTH
			print np.count_nonzero(output)
		
		
			for j in range(start_point,end_point):
				save_name = OUTPUT_DIRECTORY + shape_str + str(j) + '.png'
				temp = output[j - start_point,:,:,0]
				png.from_array((temp).tolist(),'L').save(save_name)

	

train_graph()
evaluate_graph()


