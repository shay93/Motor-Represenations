import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


#first define some global variables needed for the graph
#set the length of the input vector at each step using Batch Size
BATCH_SIZE = 1
#Defien the total number of sinuisoid points that you want to generate
TOTAL_POINTS = 10
#this is the number of steps
NUM_TIMESTEPS = 10
#This is the length of the column vector fed in at each step
NUM_OF_BATCHES = TOTAL_POINTS // NUM_TIMESTEPS
#This is the number of times one trains over the whole training dataset
EPOCHS = 100
#This is the length of the hidden layer vector, the number of hidden units must be enough to generate all teh outputs from the batch hence
NUM_HIDDEN_UNITS = NUM_TIMESTEPS

def generate_data_batches():
	'''Generates a sinuisoid of data with specified number of points
		Returns a numpy array with of size(points,)
	'''
	#get equally spaced points between zero and BATCH_SIZE
	x = np.linspace(0,20*np.pi,num = TOTAL_POINTS)
	#apply sin operation on output
	y1 = x#np.sin(x)
	y2 = x + 5 #np.sin(x)
	# generate batches of data
	batch_list_y1 = np.split(y1,NUM_OF_BATCHES)
	batch_list_y2 = np.split(y2,NUM_OF_BATCHES)
	#let x-batch represent the data fed into the rnn this is the sequence of sinuisoidal data excluding the last entry
	for i in range(0,NUM_OF_BATCHES):
		batch_list_y1[i] = np.reshape(batch_list_y1[i],[1,NUM_TIMESTEPS])
		batch_list_y2[i] = np.reshape(batch_list_y2[i],[1,NUM_TIMESTEPS])
	return batch_list_y1,batch_list_y2
def build_model():
	'''This should construct graph of operations that corresponds to rnn'''
	#get the data by calling on generate data function
	x = tf.placeholder(tf.float32,shape =[1,NUM_TIMESTEPS])
	x_input = tf.split(1,NUM_TIMESTEPS,x)
	y_ = tf.placeholder(tf.float32, shape = [1,NUM_TIMESTEPS])
	# define a cell by specifying number of hidden units
	cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN_UNITS,state_is_tuple = True) #hidden layer outputs consist of list of length equal to number of time steps
	#construct a network using the cell and passing in the data
	output,states = tf.nn.rnn(cell,x_input, dtype = tf.float32)
	#initialize weights for the transformation from the final output 
	W_fc = tf.Variable(tf.truncated_normal(shape = [NUM_HIDDEN_UNITS,NUM_TIMESTEPS]))
	b_fc = tf.Variable(tf.constant(0.1, shape = [NUM_TIMESTEPS]))
	#y = tf.nn.tanh(tf.matmul(output[-1],W_fc) + b_fc)
	y = tf.reshape(output[-1], shape = [1,NUM_TIMESTEPS])
	output_test = tf.reduce_mean(states)
	#define a training nodes in the graph
	#define a loss function to train the nueral network
	meansq= tf.reduce_mean(tf.square(y - y_))
	train_step = tf.train.AdamOptimizer(1).minimize(meansq)
	return meansq,output,x,y_,y,states,train_step

def train_model():
	x_batch_list,y_batch_list = generate_data_batches()
	meansq,output,x,y_,y,states,train_step = build_model()
	#initialize a session and then intitialize all variables
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	for j in range(EPOCHS):
		for i in range(0,NUM_OF_BATCHES - 1):
			test1,test2,loss,_ = sess.run([states[1],output[-1],meansq,train_step],feed_dict = {x : x_batch_list[i], y_ : y_batch_list[i]})
			if (i+(j*NUM_OF_BATCHES - 1)) % 50 == 0:
				print loss,i+(j*NUM_OF_BATCHES - 1)
				#print np.mean(test1),np.mean(test2)

	#once optimization complete find the y output
	output_list = [0] * (NUM_OF_BATCHES)
	for j in range(0,NUM_OF_BATCHES):
		output_list[j] = np.array(sess.run(y, feed_dict = {x:x_batch_list[j]}))
	print output_list
	print np.array(output_list).shape
	print np.array(x_batch_list).shape
	output = np.hstack(output_list)
	x_input = np.hstack(x_batch_list)
	plt.figure()
	plt.plot(output.flatten(), label = 'Predicted')
	plt.plot(x_input.flatten(), label = 'Baseline')
	plt.legend()
	plt.show()


train_model()