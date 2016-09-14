from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DECODER_MAX_LENGTH = 20
ENCODER_MAX_LENGTH = 15
BATCH_SIZE = 15
HIDDEN_UNITS = INPUT_FEATURES = 1
OUTPUT_FEATURES = 1
NUM_BATCHES = 10
LEARNING_RATE = 1e-3



class variable_lstm:

	def __init__(self):
		#initialize some variables
		self.cur_tstep = tf.constant(1,shape = [1])
		#specify the termination tstep 
		self.termination_tstep = tf.placeholder(tf.int32,shape = [1])
		#initialize LSTM cell
		self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS,state_is_tuple = True)
		#define the input of shape [Batch_size,Num of hidden units,outputs]
		self.x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH])


	def construct_decoder(self,old_state):

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
		#initialize the state tuple
		state_tuple = self.lstm_cell.zero_state(BATCH_SIZE,tf.float32)
		loop_var = [self.cur_tstep,state_tuple[0],state_tuple[1]]
		#use the above to construct the while loop 
		r = tf.while_loop(lambda tstep,old_m_state,old_c_state : self.cond(tstep,old_m_state,old_c_state),lambda tstep,old_m_state,old_c_state : self.body(tstep,old_m_state,old_c_state), loop_var)
		#initialize a session
		decoder_output = self.construct_decoder((r[1],r[2]))
		sess = tf.InteractiveSession()
		sess.run(tf.initialize_all_variables())
		print decoder_output[2].eval(feed_dict = {self.termination_tstep : [6],self.x : np.random.rand(BATCH_SIZE,INPUT_FEATURES,ENCODER_MAX_LENGTH)})
		return sess

my_graph = variable_lstm()
my_graph.build_graph()


