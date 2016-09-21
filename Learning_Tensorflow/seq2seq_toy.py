import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


#Specify global variables including hyper-parameters
TOTAL_NUM_TIMESTEPS = 60
ENCODER_TIMESTEPS = 40
DECODER_TIMESTEPS = TOTAL_NUM_TIMESTEPS - ENCODER_TIMESTEPS
NUM_OF_PERIODS = 3
VECTOR_LENGTH = 1
BATCH_SIZE = 10
NUM_OF_BATCHES = 1
HIDDEN_UNITS = 30
learning_rate = 1e-4
#first generate the training data by determining
time_array = np.linspace(0,NUM_OF_PERIODS*np.pi*2,num = TOTAL_NUM_TIMESTEPS)
encoder_time_array = time_array[:ENCODER_TIMESTEPS]
decoder_time_array = time_array[ENCODER_TIMESTEPS:]
print len(decoder_time_array)
#now generate training data using sines of random phase between -pi and pi
#training data should consist of a [BATCH_SIZE,NUM_TIMESTEPS] array
#first initialize array
encoder_train = [[np.ndarray(shape = (VECTOR_LENGTH,ENCODER_TIMESTEPS),dtype = float)]*BATCH_SIZE] * NUM_OF_BATCHES
decoder_train_input = [[np.ndarray(shape = (VECTOR_LENGTH,DECODER_TIMESTEPS),dtype = float)]*BATCH_SIZE] * NUM_OF_BATCHES
decoder_train_output = [[np.ndarray(shape = (VECTOR_LENGTH,DECODER_TIMESTEPS),dtype = float)]*BATCH_SIZE] * NUM_OF_BATCHES
#generate array of random phases
phase_array = [(np.random.rand(BATCH_SIZE) - 0.5) * 2 * np.pi] * NUM_OF_BATCHES
for batch_num in range(0,NUM_OF_BATCHES):
	for element in range(0,BATCH_SIZE):
		for index in range(0,VECTOR_LENGTH):
			encoder_train[batch_num][element][index,:] = np.sin(encoder_time_array - phase_array[batch_num][element])
			decoder_train_input[batch_num][element][index,:] = np.sin(decoder_time_array - phase_array[batch_num][element])
			decoder_train_output[batch_num][element][index,:] = np.cos(decoder_time_array - phase_array[batch_num][element])

#convert the training data into tensor flow operations
tf_encoder_train = tf.constant(encoder_train)
tf_decoder_train_input = tf.constant(decoder_train_input)
y_ = tf.constant(decoder_train_output[0])
#once you have generated the training data construct the model
#define the number of hidden units in the LSTM cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS)
#now define the rnn around this that outputs tensors corresponding to outputs and states
outputs,states = tf.nn.seq2seq.basic_rnn_seq2seq(tf_encoder_train[0],tf_decoder_train_input[0],lstm_cell) 
#now define a loss function by using the frobenius norm
#inorder to to take difference and define loss its first necessary to convert list to tensor
y = tf.convert_to_tensor(outputs)
loss = tf.reduce_sum(tf.squared_difference(y,y_))
#now add training node to graph
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#initialize graph
sess = tf.Session()
#initialize variables
sess.run(tf.initialize_all_variables())
for batch_num in range(0,NUM_OF_BATCHES):
	_,error = sess.run([train_step,loss])
	print error
