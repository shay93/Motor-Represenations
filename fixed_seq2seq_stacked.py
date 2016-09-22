from __future__ import division
import numpy as np
import tensorflow as tf


#define a class for a stacked lstm
class stacked_lstm():
	
	def _init_(self,num_stacks,batch_size,input_features,output_features,sequence_length,hidden_units):
		"""
		Define some attributes for the class
		These may also be regarded as globals for the object and its methods
		"""
		self.batch_size = batch_size
		self.num_stacks = num_stacks
		self.img_width = 64
		self.input_features = input_features
		self.output_features = output_features
		self.sequence_length = sequence_length
		self.hidden_units = hidden_units
		self.op_dict = {}	
	
	def build_model(self):
		"""
		Responsible for defining the computational operations
		that comprise the graph
		inputs: self
		outputs: An operation dictionary with operations that may be useful
			for training purposes and evaluation. 
		"""	
		#define a placeholder
		self.op_dict['x'] = tf.placeholder(tf.float32, shape=[self.batch_size,self.input_features,self.sequence_length)
		#split x so that it may each time step may be fed into a seperate stacked lstm cell
		x_list = tf.split(2,self.sequence_length,self.op_dict['x'])
		#initialize a weight for a fully connected layer for each timestep 
		W_fc1 = tf.Variable(tf.truncated_normal(shape = [self.input_features,self.hidden_units,stddev = 0.1))
		#also define a bias term for the fully connected layer
		b_fc1 = tf.Variable(tf.constant(0.1,shape = [self.batch_size,self.hidden_units]))
		#initialize a list to record outputs from the first fc layer
		h_fc1_list = []
		#now run through the time steps of x and pass them into the fully connected layer to calculate the first hidden layer output
		for x_timestep in x_list:
			#first reshape the x_timestep since it is 3d
			x_timestep = tf.reshape(x_timestep, shape = [self.batch_size,self.input_features])
			h_fc_timestep = tf.sigmoid(tf.matmul(W_fc1,x_timestep) + b_fc)
			h_fc1_list.append(h_fc_timestep)
		#also initialize a multirnncell, this requires defining a list of lstm ccells first
		lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units,state_is_tuple = True)] * self.num_stacks
		#pass this list into a multirnn_cell
		multicell = tf.nn.rnn_cell(lstm_list)
		#the aim is now to propogate this multicell rnn forward in time capturing the output at each step
		#to this end initialize an output list
		y_list = []
		#initialize a state for the multicell
		cur_state = multicell.zero_state(self.batch_size,tf.float32)
		#define another fully connected layer for the output
		W_fc2 = tf.Variable(tf.truncated_normal(shape = [self.hidden_units,self.output_features], stddev = 0.1))
		#now define a bias for this fully connected layer
		b_fc2 = tf.Variable(tf.constant(0.1,shape = [self.batch_size,self.output_features))
		#propogate the rnn by iterating through the outputs from the first fully connected layer
		for h_fc1_timestep in h_fc1_list:
			rnn_timestep,cur_state = multicell(cur_state,h_fc1_timestep)
			#pass the output from the rnn to fc layer
			y_timestep = tf.sigmoid(tf.matmul(h_fc1_timestep,W_fc2) + b_fc2)
			#append this to the output list
			y_list.append(y_timestep)

		#reshape the list in order to get the output y
		self.op_dict['y'] = tf.reshape(y_list, shape = [self.batch_size,self.output_features,self.sequence_length])
		#specify a placeholder for the above
		self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [self.batch_size,self.output_features,self.sequence_length)
		#define a meansq loss
		self.op_dict['meansq'] = tf.reduce_mean(tf.square(self.op_dict['y'] - self.op_dict['y_']))
		#specify a training node
		self.op_dict['train_op'] = tf.train.AdamOptimizer(self.learning_rate).minmize(self.op_dict['meansq'])
		#initialize a graph and the variables via a sessions object
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		return sess
		


