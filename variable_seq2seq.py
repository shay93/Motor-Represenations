from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#first thing to do is to define some global variables
DECODER_MAX_LENGTH = 20
BATCH_SIZE = 15
HIDDEN_UNITS = 1 
INPUT_FEATURES = 1
OUTPUT_FEATURES = 1
LEARNING_RATE = 1e-3
FC_UNITS = DECODER_MAX_LENGTH*BATCH_SIZE*HIDDEN_UNITS


#first define a function that constructs the decoder
class variable_lstm:

	def __init__(self):
		self.hidden_units = HIDDEN_UNITS
		self.batch_size = BATCH_SIZE
		self.input_features = INPUT_FEATURES
		self.learning_rate = LEARNING_RATE
		self.encoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS)
		#initialize a tensor for the curren time step
		self.cur_tstep =tf.constant(1,shape = [1])
				
		#instead of using a boolean array just define a termination point in the sequence
		self.termination_tstep = tf.placeholder(tf.int32,shape = [1],name = "test")
		#define input place holder for the variable length input
		self.x_input = tf.placeholder(tf.float32, shape = [BATCH_SIZE,INPUT_FEATURES,None])
		#self.W_fc = tf.Variable(tf.truncated_normal(shape = [FC_UNITS,DECODER_MAX_LENGTH*OUTPUT_FEATURES*BATCH_SIZE]))
		#self.b_fc = tf.Variable(tf.constant(0.0,shape = [DECODER_MAX_LENGTH*OUTPUT_FEATURES*BATCH_SIZE]))
		self.op_dict = {}
	
	def construct_decoder(self,old_state):
		"""
		Constructs the operations for the decoder sub-graph which is initialized by the last state of the encoder.
		The subgraph hidden state serves as the output at each timestep.
		Args: old_state - Tensor of shape [Batch_Size,Hidden_Units * 2] 
			  last_output - Tensor of shape [Batch_Size,Hidden_Units]
		Returns: decoder_output - List of Tensors of shape [Batch_Size,Features], length of Decoder is DECODER_MAX_LENGTH
		"""
		with tf.variable_scope("Decoder_RNN") as scope:
			#initialize the decoder lstm cell
			decoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS)
			#initialize a list for decoder output
			decoder_output = [0] * DECODER_MAX_LENGTH
			#get the last output by splitting the state
			print old_state
			last_output,_ = tf.split(1,2,old_state)
			#now unroll the decoder lstm plugging in the previous input at each time step
			for tstep in range(DECODER_MAX_LENGTH):
				cur_output,new_state = decoder_lstm(last_output,old_state,scope = "Cell" + str(tstep))
				#append the cur_output to the decoder_output array
				decoder_output[tstep] = cur_output
				#now reassign the old_state and last_output to the new_state and cur_output
				last_output = cur_output
				old_state = new_state

		return decoder_output


	def unroll_one_tstep(self,x,cur_tstep,termination_tstep,cell,state):
		"""
		Takes RNN cell its previous state and current input and unrolls it for one time step
		Args: cell - a Tensorflow cell to unroll
			  state - A 2d tensor of shape [BATCH_SIZE,HIDDEN_UNITS*2]
			  input - A 2d tensor of shape [BATCH_SIZE,FEATURES]
		Returns: output - A 2d tensor of shape [BATCH_SIZE,HIDDEN_UNITS]
				 new_state - A 2d tensor of shape [BATCH_SIZE,HIDDEN_UNITS*2]
		"""
		#take the input x and slice out the x_input that belongs to the right tstep
		#first define the beginning index
		index_start = tf.concat(0,[tf.constant([0,0]),cur_tstep])
		slice_size = tf.constant([-1,-1,1])
		x_slice = tf.slice(x,index_start,slice_size)
		#now reshape x_slice into a 2d tensor
		x_reshape = tf.reshape(x_slice,shape = [BATCH_SIZE,INPUT_FEATURES])
		output,state = cell(x_reshape,state,scope = "Cell" + str(cur_tstep))
		return state

	#initialize the encoder lstm cell

	def cond(self,cur_tstep,termination_tstep,state,x_input):
		"""
		Takes the termination step and returns a boolean specifying if the condition has been met
		Args: termination_tstep - An integer tensor specifying the end of the input variable length sequence
			  cur_tstep - The current tstep in the position
		Returns: An integer Boolean tensor specifying if the end of the while loop has been reached
		"""
		boolean = tf.less(cur_tstep ,termination_tstep)[0]
		return boolean

	def body(self,cur_tstep,termination_tstep,state,x_input):
		"""
		This function takes the loop variables and updates them for every timestep
		Args: 
		Returns: The loop variables which are updated 
		"""
		#first call on unroll_one_tstep in order to get the new state
		state = self.unroll_one_tstep(x_input,cur_tstep,termination_tstep,self.encoder_lstm,state)
		#termination_tstep remains fixed so does the input_x and so does lstm_cell
		cur_tstep = tf.add(cur_tstep ,tf.constant(1,shape = [1]))
		return cur_tstep,termination_tstep,state,x_input

	#now construct the encoder part of the RNN
	def construct_graph(self,state):
		"""
		Constructs tensorflow graph using the weights of
		"""
		#define the loop variable list that will be used by tf.while_loop
		loop_variable = [self.cur_tstep,self.termination_tstep,state,self.x_input]
		#use tf.while_loop to get the required output
		r = tf.while_loop(lambda  cur_tstep,termination_tstep,state,x_input : self.cond(cur_tstep,termination_tstep,state,x_input),
			lambda cur_tstep,termination_tstep,state,x_input : self.body(cur_tstep,termination_tstep,state,x_input), loop_variable )
		self.op_dict['fc'] = tf.pack(self.construct_decoder(r[2]))
		print self.op_dict['fc']
		# #reshape the fully connected layer to get a 2d array
		#self.op_dict['fc_reshape'] = tf.reshape(self.op_dict['fc'],[-1,FC_UNITS])
		# #define a weight variable to transform the output to what is required
		
		# #use the above to obtain the y that you want
		# y_2d = tf.nn.tanh(tf.matmul(self.op_dict['fc_reshape'],self.W_fc))
		# #now reshape to obtain the correctly shaped y
		self.op_dict['y'] = tf.reshape(self.op_dict['fc'],shape = [BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		# #define an input for the label 
		self.op_dict['y_'] = tf.placeholder(tf.float32,shape = [BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		# #now define a loss node and a training node using
		self.op_dict['meansq'] = tf.reduce_mean(tf.square(tf.sub(self.op_dict['y_'],self.op_dict['y'])))
		# #now define a training node using the above
		self.op_dict['train_op'] = tf.train.AdamOptimizer(self.learning_rate).minimize(self.op_dict['meansq'])
		sess = tf.Session()
		#sess.run(tf.initialize_all_variables())
		return sess

	def train_graph(self,Epoch,x_list,y_list,termination_tstep_list,sess):
		"""
		Treat the x_list and y_list to be lists of tensors of shape =[batch_size,features,sequence_length]
		returns a session object that can be used to evaluate the graph for a certain input

		"""
		#first get a sessions object and initialize all the variables
		#initialize a loss array
		loss_array = [0] * len(x_list)
		#now run the train operation for each batch of inputs
		for index in range(Epoch):
			for batch_num in range(len(x_list)):
				loss = sess.run(self.op_dict['meansq'],feed_dict = {self.x_input: x_list[batch_num],self.op_dict['y_'] : y_list[batch_num], 
					self.termination_tstep : [termination_tstep_list[batch_num]]})
				loss_array[batch_num] = loss
		
		return sess,loss_array

	def evaluate_graph(self,y,x,termination_tstep,sess):
		"""
		use the sessions object to run the graph for the data that is given for evaluation
		"""
		#use the sess object to get the predicted data
		predicted_y = sess.run(y,feed_dict = {self.op_dict['x_'] : x, self.op_dict['termination_tstep'] : termination_tstep })
		#pick a random batch out of the predicted
		random_batch = np.random.rand() * BATCH_SIZE
		#plot the random batch against the 
		y_sample_batch = np.reshape(predicted_y[random_batch,:,:],[OUTPUT_FEATURES,DECODER_MAX_LENGTH])
		plt.figure()
		plt.plot(x[random_batch,:,:],label = "input")
		plt.plot(y_sample_batch,label = "Predicted Output")
		plt.legend()



state = tf.truncated_normal(shape = [BATCH_SIZE,HIDDEN_UNITS*2],stddev = 0.1)
my_lstm = variable_lstm()
sess = my_lstm.construct_graph(state)
x_list = [np.random.rand(BATCH_SIZE,INPUT_FEATURES,6)] * 5
y_list = [np.random.rand(BATCH_SIZE,OUTPUT_FEATURES,DECODER_MAX_LENGTH)] * 5
termination_tstep_list = [3] * 5
sess,loss_array = my_lstm.train_graph(1,x_list,y_list,termination_tstep_list,sess)