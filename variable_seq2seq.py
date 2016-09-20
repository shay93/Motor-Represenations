from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import training_tools as tt
import pickle

DECODER_MAX_LENGTH = 250
ENCODER_MAX_LENGTH = 250
BATCH_SIZE = 12
HIDDEN_UNITS = 500
OUTPUT_FEATURES = 2
INPUT_FEATURES = 2
NUM_BATCHES = 12
LEARNING_RATE = 1e-3
number_of_shapes = 3
num_shapes_per_Epoch = 3000
num_each_shape_per_batch = BATCH_SIZE // number_of_shapes

def extract_data_lstm(EPOCHS):
	"""
	Given the number of EPOCHS the number of shapes being used and the batch size
	this function should be able to extract the state of the arms used to draw shapes
	in a format that is acceptable for variable_seq2seq.py
	inputs: -
	returns: Three lists with length equal to the number of batches required to traverse the EPOCHS.
			 The lists are the termination tstep list, the x_list and the y_list  
	"""
	
	#first things first load the saved state array
	rectangle_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Rectangle_50.npy', 'rb'))
	square_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Square_50.npy', 'rb'))
	triangle_state_first_arm = pickle.load(open('Training_Data_First_Arm/saved_state_Triangle_50.npy', 'rb'))

	rectangle_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Rectangle_80.npy', 'rb'))
	square_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Square_80.npy', 'rb'))
	triangle_state_second_arm = pickle.load(open('Training_Data_Second_Arm/saved_state_Triangle_80.npy', 'rb'))

	#each of these states are lists with a thousand elements, the aim is to now create batches out of these
	#lets first initialize the lists that we are dealing with the length of the list should be equal to 
	num_batches_in_Epoch =  num_shapes_per_Epoch // BATCH_SIZE
	#now we know the length of x_list and y_list
	x_list = [0] * (num_batches_in_Epoch * EPOCHS)
	y_list = [0] * (num_batches_in_Epoch * EPOCHS)
	termination_tstep_list = [0] * (num_batches_in_Epoch * EPOCHS)

	#inorder to make looping easier define a shape array
	shape_state_array_first_arm = [rectangle_state_first_arm,square_state_first_arm,triangle_state_first_arm]
	shape_state_array_second_arm = [rectangle_state_second_arm,square_state_second_arm,triangle_state_second_arm]
	
	for batch_num in range(num_batches_in_Epoch * EPOCHS):
		#initialize an empty array of zeros as x_list
		x_temp = np.zeros([BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH])
		y_temp = np.zeros([BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		
		batch_index = batch_num % num_batches_in_Epoch
		termination_tstep = 0
		
		for j,shape_state in enumerate(shape_state_array_first_arm):
			for i in range(num_each_shape_per_batch):
				shape_index = batch_index*4 + i
				#get the number of time steps in each input
				_,timesteps = np.shape(shape_state[shape_index])
				if timesteps > termination_tstep:
					termination_tstep = timesteps
				x_temp[j*num_each_shape_per_batch + i,0:INPUT_FEATURES,:timesteps] = shape_state[shape_index]

		for j,shape_state in enumerate(shape_state_array_second_arm):
			for i in range(num_each_shape_per_batch):
				shape_index = batch_index*4 + i
				#get the number of time steps in each input
				_,timesteps = np.shape(shape_state[shape_index])
				y_temp[j*num_each_shape_per_batch + i,0:OUTPUT_FEATURES,:timesteps] = shape_state[shape_index]
		
		termination_tstep_list[batch_num] = [termination_tstep]
		x_list[batch_num] = x_temp
		y_list[batch_num] = y_temp

	return x_list,y_list,termination_tstep_list

class variable_lstm:

	def __init__(self):
		#initalize global variables
		self.hidden_units = HIDDEN_UNITS
		self.batch_size = BATCH_SIZE
		self.input_features = INPUT_FEATURES
		self.output_features = OUTPUT_FEATURES
		self.encoder_max_length = ENCODER_MAX_LENGTH
		self.decoder_max_length = DECODER_MAX_LENGTH
		#initialize some variables
		self.cur_tstep = tf.constant(1,shape = [1])
		#specify the termination tstep 
		self.termination_tstep = tf.placeholder(tf.int32,shape = [1])
		#initialize LSTM cell
		self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS,state_is_tuple = True)
		#define the input of shape [Batch_size,Num of hidden units,outputs]
		self.x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH])
		self.y_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH]) 

	def construct_decoder(self,old_state):
		"""
		Takes the cell and hidden state from the encoder and feeds it to a decoder in order to get
		an output equal to the decoder length
		input : old_state - tuple with first entry corresponding to the cell state and second the hidden state
							each entry has shape [Batch_Size, Number of Units]

		output : decoder_output - A list of outputs at each cell of the LSTM. The shape is [DECODER_MAX_LENGTH,BATCH_SIZE,HIDDEN_UNITS]
		"""

		with tf.variable_scope("Decoder_RNN") as scope:
			#initialize the decoder lstm cell
			decoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS, state_is_tuple = True)
			#initialize a list for decoder output
			decoder_output = [0] * DECODER_MAX_LENGTH
			#now unroll the decoder lstm plugging in the previous input at each time step
			last_output = old_state[0]
			for tstep in range(DECODER_MAX_LENGTH):
				cur_output,new_state = decoder_lstm(last_output,old_state,scope = "Cell" + str(tstep))
				#append the cur_output to the decoder_output array
				decoder_output[tstep] = cur_output
				#now reassign the old_state and last_output to the new_state and cur_output
				last_output = cur_output
				old_state = new_state

		return decoder_output


	def body(self,tstep,old_m_state,old_c_state):
		new_tstep = tf.add(tstep,tf.constant(1, shape = [1]))
		#concatentate the new tstep with the
		index = tf.concat(0,[tf.constant([0,0]),new_tstep])
		size = tf.constant([-1,-1,1])
		x_sliced = tf.slice(self.x,index,size)
		x_reshape = tf.reshape(x_sliced,shape = [BATCH_SIZE,INPUT_FEATURES])
		#now feed this into lstm
		output,state = self.lstm_cell(x_reshape,(old_m_state,old_c_state))
		return new_tstep,state[0],state[1]

	
	def cond(self,tstep,old_m_state,old_c_state):
		return tf.less(tstep,self.termination_tstep)[0]


	def build_graph(self):
		"""
		specifies the operations that build the computational graph
		output : op_list - a list of tensors corresponding to operations that may be required in another function
				 sess - a tensorflow object required to evaluate operations that constitute the graph
		"""
		#initialize the state tuple
		state_tuple = self.lstm_cell.zero_state(BATCH_SIZE,tf.float32)
		loop_var = [self.cur_tstep,state_tuple[0],state_tuple[1]]
		#use the above to construct the while loop 
		r = tf.while_loop(lambda tstep,old_m_state,old_c_state : self.cond(tstep,old_m_state,old_c_state),lambda tstep,old_m_state,old_c_state : self.body(tstep,old_m_state,old_c_state), loop_var)
		#use the encoder output to initialize the decoder
		decoder_output = self.construct_decoder((r[1],r[2]))
		#reshape the decoder output such that it may be used to compute a loss
		#add a fully connected layer after the decoder_output
		#in order to this you must first reshape the decoder output
		decoder_output2d = tf.reshape(decoder_output,shape = [BATCH_SIZE*HIDDEN_UNITS,DECODER_MAX_LENGTH])
		#initialize a weight matrix for the fully connected layer
		W_fc = tf.Variable(tf.truncated_normal(shape = [BATCH_SIZE*OUTPUT_FEATURES,BATCH_SIZE*HIDDEN_UNITS], stddev = 0.1))
		#Initialize a bias for the fully connected layer
		b_fc = tf.Variable(tf.constant(0.0, shape = [DECODER_MAX_LENGTH]))
		#now compute the output after it passes through the full connected layer
		fc_output = tf.nn.tanh(tf.matmul(W_fc,decoder_output2d) + b_fc)
		#reshape the output from the fully connected layer to get y
		y = tf.reshape(fc_output, shape = [BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		#compute a loss
		meansq = tf.reduce_mean(tf.square(tf.sub(y,self.y_)))
		#use the loss to define a training operation 
		train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(meansq)
		#define a list of operations that would be useful 
		op_list = [self.y_,y,meansq,train_op]
		#initialize graph
		sess = tf.Session()
		#initialize variables
		sess.run(tf.initialize_all_variables())
		return sess,op_list


	def train_graph(self,sess,op_list,x_list,y_list,termination_tstep_list):
		"""
		Tune the parameters of the graph so that it learns the right transformation
		inputs: sess - object required in order to evaluate a graph operation.
				op_list - A list of operations
				x_list - A list of input arrays of dimension [Batch Size, Input_Features, Encoder_Max_Length] list length equal to EPOCHS*num of batches in each EPOCH
				y_list - A list of output arrays of dimentions [Batch Size, Output_Features, Decoder_Max_Length] list length equal to EPOCHS*num of batches in each EPOCH
				termination_tstep_list - A list of arrays of size one corresponding to the termination time_step for each batch
		"""
		loss_array = [0] * len(x_list)
		for batch_num in range(len(x_list)):
			loss,_ = sess.run([op_list[2],op_list[-1]],feed_dict = {op_list[0] : y_list[batch_num], self.termination_tstep : termination_tstep_list[batch_num], self.x : x_list[batch_num] })
			loss_array[batch_num] = loss
			if batch_num % 20 == 0:
				print batch_num,loss

		return loss_array


	def evaluate_graph(self,sess,op_list):
		"""
		"""
		#first get the data from lstm
		x_list,y_list,termination_tstep_list = extract_data_lstm(1)
		#initialize an output array
		output_list = [0] * len(x_list)
		#initialize an arm of the right link length
		my_arm = tt.two_link_arm(80)
		#now use the data in x to evaluate the graph
		for batch_num,x in enumerate(x_list):
			y_temp = sess.run(op_list[1],feed_dict = {self.x : x, self.termination_tstep : termination_tstep_list[batch_num]})
			output_list[batch_num] = y_temp
			#initialize a grid
			my_grid = tt.grid('output' + str(batch_num),'Output_Images_LSTM/')
			pos_list = my_arm.forward_kinematics(y_temp[0,:,:])
			grid_array = my_grid.draw_figure(pos_list)
			print len(pos_list)
			my_grid.save_image()
			#print np.shape(sessu.run(op_list[1],feed_dict = {self.x : np.random.rand(BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH),self.termination_tstep : [6]}))



#need to read encoder data into 
my_graph = variable_lstm()
sess,op_list = my_graph.build_graph()
x_list,y_list,termination_tstep_list = extract_data_lstm(5)
my_graph.train_graph(sess,op_list,x_list,y_list,termination_tstep_list)
my_graph.evaluate_graph(sess,op_list)


