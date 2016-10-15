from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import training_tools as tt
import pickle

DECODER_MAX_LENGTH = 200
BATCH_SIZE = 50
HIDDEN_UNITS = 50
EVAL_BATCH_SIZE = 50 

OUTPUT_FEATURES = 2
INPUT_FEATURES = 2
NUM_BATCHES = 12
LEARNING_RATE = 1e-3

VALIDATION_SIZE = 200
NUM_OF_WAVES = 3000
EPOCHS = 5
DISPLAY_NUM_EXAMPLES = 10
ROOT_DIR = "Logs/"

EVAL_FREQUENCY = 10

#generate time array  
time_array = np.linspace(0,4*np.pi,num = DECODER_MAX_LENGTH)

def get_training_data(num_of_waves):
    """
    generate all the waves that may then be split into batches
    """
    sin_wave_array = np.zeros([num_of_waves,DECODER_MAX_LENGTH])
    cos_wave_array = np.zeros([num_of_waves,DECODER_MAX_LENGTH])

    #generate a random phase shift array
    phase_array = np.pi*np.random.rand(num_of_waves)
    #keep amplitude and frequency constant at 1
    #generate random lengths as well
    length_array = np.round(DECODER_MAX_LENGTH*np.random.rand(num_of_waves))
    for i in range(num_of_waves):
        sin_wave_array[i,:length_array[i]] = np.sin(time_array[:length_array[i]] - phase_array[i])
        cos_wave_array[i,:length_array[i]] = np.cos(time_array[:length_array[i]] - phase_array[i])

    return sin_wave_array,cos_wave_array,length_array


train_x_data,train_y_data,termination_tstep_array = get_training_data(NUM_OF_WAVES)
#generate a validation set
validation_x_data = train_x_data[:VALIDATION_SIZE, ...]
validation_y_data = train_y_data[:VALIDATION_SIZE, ...]
validation_tstep_array = termination_tstep_array[:VALIDATION_SIZE]
#get the training data required 
train_x_data = train_x_data[VALIDATION_SIZE:, ...]
train_y_data = train_y_data[VALIDATION_SIZE:, ...]
training_tstep_array = termination_tstep_array[VALIDATION_SIZE:]

num_epochs = EPOCHS
train_size = train_x_data.shape[0]

class variable_lstm:

	def __init__(self):
		#initalize global variables
		self.hidden_units = HIDDEN_UNITS
		self.batch_size = BATCH_SIZE
		self.input_features = INPUT_FEATURES
		self.output_features = OUTPUT_FEATURES
		self.decoder_max_length = DECODER_MAX_LENGTH
		#initialize some variables
		self.cur_tstep = tf.constant([0])
		#specify the termination tstep 
		self.batch_termination_tstep = tf.placeholder(tf.int32,shape = [1])
		#initialize LSTM cell
		self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS,state_is_tuple = True)
		#define an operation dictionary
		self.op_dict = {}
		self.parameter_dict = {}

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
			decoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units, state_is_tuple = True)
			#initialize a list for decoder output
			decoder_output = [0] * DECODER_MAX_LENGTH
			#now unroll the decoder lstm plugging in the previous input at each time step
			last_output = old_state[0]
			#define a weight array 
			for tstep in range(DECODER_MAX_LENGTH):
				cur_output,new_state = decoder_lstm(last_output,old_state,scope = "Cell" + str(tstep))
				#append the cur_output to the decoder_output array
				decoder_output[tstep] = tf.nn.tanh(tf.matmul(cur_output,self.parameter_dict['W_decoder']) + self.parameter_dict['b_decoder'])
				#now reassign the old_state and last_output to the new_state and cur_output
				last_output = cur_output
				old_state = new_state

			#now join the decoder
			decoder_tensor = tf.concat(1,decoder_output)
			#now use the the termination tstep to slice out the right array
			begin = tf.constant([0,0])
			size = tf.concat(0,[tf.constant([-1]),self.batch_termination_tstep])
			decoder_tensor_sliced = tf.slice(decoder_tensor, begin, size)

		return decoder_tensor_sliced


	def body(self,tstep,old_m_state,old_c_state):
		
		#concatentate the new tstep with the
		#index = tf.concat(0,[tf.constant([0]),new_tstep])
		#print index
		size = tf.constant([-1,1])
		x_sliced = self.op_dict['x'][:,1]
		x_reshape = tf.reshape(x_sliced,shape = [-1,1])
		output,state = self.lstm_cell(x_reshape,(old_m_state,old_c_state))
		new_tstep = tstep + 1
		return new_tstep,state[0],state[1]

	
	def cond(self,tstep,old_m_state,old_c_state):
		return tf.less(tstep,self.batch_termination_tstep)[0]


	def build_graph(self):
		"""
		specifies the operations that build the computational graph
		output : op_list - a list of tensors corresponding to operations that may be required in another function
				 sess - a tensorflow object required to evaluate operations that constitute the graph
		"""
		#initialize placeholders
		self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None, None])
		self.op_dict['y_'] = tf.placeholder(tf.float32, shape = [None, None])
		#define a weight variable for a decoder
		self.parameter_dict['W_decoder'] = tf.truncated_normal(shape = [self.hidden_units,self.output_features],stddev = 0.1)
		self.parameter_dict['b_decoder'] = tf.constant(0., shape = [self.output_features])
		#initialize the state tuple
		state_tuple = self.lstm_cell.zero_state(self.batch_size,tf.float32)
		loop_var = [self.cur_tstep,state_tuple[0],state_tuple[1]]
		#use the above to construct the while loop 
		r = tf.while_loop(lambda tstep,old_m_state,old_c_state : self.cond(tstep,old_m_state,old_c_state),lambda tstep,old_m_state,old_c_state : self.body(tstep,old_m_state,old_c_state), loop_var)
		#use the encoder output to initialize the decoder
		decoder_output = self.construct_decoder((r[1],r[2]))
		self.op_dict['y'] = decoder_output
		#compute a loss
		self.op_dict['meansq'] = tf.reduce_mean(tf.square(self.op_dict['y'] - self.op_dict['y_']))
		#use the loss to define a training operation 
		self.op_dict['train_op'] = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.op_dict['meansq'])
		

	def eval_in_batches(self,x_data,y_data,sess):
		"""Get combined loss for dataset by running in batches"""
		size = x_data.shape[0]

		if size < EVAL_BATCH_SIZE:
			raise ValueError("batch size for evals larger than dataset: %d" % size)

		predictions = np.ndarray(shape = (size,DECODER_MAX_LENGTH), dtype = np.float32)
		test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
		i = 0
		for begin in xrange(0,size,EVAL_BATCH_SIZE):
			end = begin + EVAL_BATCH_SIZE
			
			if end <= size:
				batch_termination_tstep = np.max(validation_tstep_array[begin:end])
				predictions[begin:end, :batch_termination_tstep],l = sess.run([self.op_dict['y'],self.op_dict['meansq']],feed_dict={self.op_dict['x']: x_data[begin:end, :batch_termination_tstep], self.op_dict['y_'] : y_data[begin:end, :batch_termination_tstep], self.batch_termination_tstep : [batch_termination_tstep]})
			else:
				batch_termination_tstep = np.max(validation_tstep_array[-EVAL_BATCH_SIZE:])
				batch_prediction,l = sess.run([self.op_dict['y'],self.op_dict['meansq']],feed_dict = {self.op_dict['x'] : x_data[-EVAL_BATCH_SIZE:, :batch_termination_tstep],self.op_dict['y_']:y_data[-EVAL_BATCH_SIZE:, :batch_termination_tstep],self.batch_termination_tstep : [batch_termination_tstep]})
				predictions[begin:, :batch_termination_tstep] = batch_prediction[(begin - size):,...]

			test_loss_array[i] = l
			i += 1
		return predictions,test_loss_array


	def train_graph(self):
		"""
		Tune the weights of the graph so that you can learn the right results
		inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
		outputs: a loss array for the purposes of plotting
		"""
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config = config) as sess:
			#initialize the variables
			log_dir = ROOT_DIR + "/tmp/summary_logs"
			train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

			tf.initialize_all_variables().run()
			#initialize a training loss array
			loss_array = [0] * (int(num_epochs * train_size) // BATCH_SIZE)
			for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
				#compute the offset of the current minibatch in the data
				offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
				batch_termination_tstep = np.max(training_tstep_array[offset:(offset + BATCH_SIZE)])
				batch_x_data = train_x_data[offset:(offset + BATCH_SIZE), :batch_termination_tstep]
				batch_y_data = train_y_data[offset:(offset + BATCH_SIZE), :batch_termination_tstep]
				feed_dict = { self.op_dict['x']: batch_x_data, self.op_dict['y_'] : batch_y_data, self.batch_termination_tstep : [batch_termination_tstep] }
				#run the graph
				_, l = sess.run(
					[self.op_dict['train_op'],self.op_dict['meansq']],
					feed_dict=feed_dict)
				loss_array[step] = l


				if step % EVAL_FREQUENCY == 0:
					predictions,test_loss_array = self.eval_in_batches(validation_x_data,validation_y_data, sess)
					print step,l
			
			output_examples = validation_y_data[:DISPLAY_NUM_EXAMPLES]
			output_predictions = predictions[:DISPLAY_NUM_EXAMPLES] 
			f, axarr = plt.subplots(2, DISPLAY_NUM_EXAMPLES)
			for i in xrange(DISPLAY_NUM_EXAMPLES):
				axarr[0,i].plot(time_array,output_examples[i,:])
				axarr[1,i].plot(time_array,predictions[i,:])



#need to read encoder data into 
my_graph = variable_lstm()
my_graph.build_graph()
my_graph.train_graph()
plt.show()


