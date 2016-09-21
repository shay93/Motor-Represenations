import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#aim of this toy will be to map a sin wave to a cosine wave via a recurrent net

#first define some global constants these include encoder timesteps decoder time steps and batch size
ENCODER_TIMESTEPS = 30
DECODER_TIMESTEPS = 20
TOTAL_TIMESTEPS = ENCODER_TIMESTEPS + DECODER_TIMESTEPS
BATCH_SIZE = 20
NUM_HIDDEN_UNITS = 5
NUM_OF_BATCHES = 500
NUM_OF_PERIODS = 2
EPOCHS = 5

#define the learning rate
learning_rate = 0.01

#first generate the training data by determining
time_array = np.linspace(0,NUM_OF_PERIODS*np.pi*2,num = TOTAL_TIMESTEPS)

#now start constructing graph for performing such a computations
x_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE,TOTAL_TIMESTEPS])
#define the a placeholder for the decoder output as well
y_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE,DECODER_TIMESTEPS])
#split the input to create a list of column vectors
x_list = tf.split(1,TOTAL_TIMESTEPS,x_)
#split the list into those that correspond to encoder and those that correspond to the decoder
x_list_encoder = x_list[:ENCODER_TIMESTEPS]
x_list_decoder = x_list[ENCODER_TIMESTEPS:]
#now define the lstm cell that will be used
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN_UNITS)
#feed the cell along with the specified inputs into the rnn constructor
outputs,states = tf.nn.seq2seq.basic_rnn_seq2seq(x_list_encoder,x_list_decoder,lstm_cell)
#now define a loss function but in order to do this first cast the output list into a 2D tensor
fc = tf.convert_to_tensor(outputs)
fc_reshape = tf.reshape(fc,shape = [BATCH_SIZE,DECODER_TIMESTEPS*NUM_HIDDEN_UNITS])
W_fc = tf.Variable(tf.truncated_normal(shape = [DECODER_TIMESTEPS*NUM_HIDDEN_UNITS,DECODER_TIMESTEPS],stddev = 0.1))
b_fc = tf.Variable(tf.constant(0.1,shape = [DECODER_TIMESTEPS]))
y = tf.nn.tanh(tf.matmul(fc_reshape,W_fc) + b_fc)
loss = tf.reduce_sum(tf.squared_difference(y,y_))
#use the loss tensor and learning rate to define a training operation
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#initialize graph
sess = tf.Session()
#initialize variables
sess.run(tf.initialize_all_variables())


#now define the training data that is fed into the graph
#first define a 2d list that randomly generates phases in radians that can be used to generate the sinuisoids
phase_array = [(np.random.rand(BATCH_SIZE) - 0.5) * 2 * np.pi] * NUM_OF_BATCHES
#input data should be a list of 2d tensors of dimension [Batch_size,Total Time steps,num of batches] hence initialize a list of input
input_data = [0] * NUM_OF_BATCHES
output_data = [0] * NUM_OF_BATCHES
#now run a for loop that inputs data into these lists
for batch_num in range(0,NUM_OF_BATCHES):
	#initialize a 2d array of size [Batch_size,Total Time steps]
	temp_x = np.zeros([BATCH_SIZE,TOTAL_TIMESTEPS])
	temp_y = np.zeros([BATCH_SIZE,DECODER_TIMESTEPS])

	for index in range(0,BATCH_SIZE):
		temp_x[index,:] = np.sin(time_array - phase_array[batch_num][index])
		temp_y[index,:] = np.cos(time_array[ENCODER_TIMESTEPS:] - phase_array[batch_num][index])

	#now append these to the lists
	input_data[batch_num] = temp_x
	output_data[batch_num] = temp_y


#now actually train the network
#define a loss array to record loss
loss_array = [0] * (EPOCHS * NUM_OF_BATCHES)
for epoch in range(0,EPOCHS):
	for batch_num in range(0,NUM_OF_BATCHES):
		_,error = sess.run([train_step,loss],feed_dict = {x_ : input_data[batch_num], y_ : output_data[batch_num]})
		loss_array[epoch*NUM_OF_BATCHES + batch_num] = error

plt.plot(loss_array)

#visualize a single batch example
example_input = input_data[5]
model_output = output_data[5][2,:]
test_output = sess.run(y, feed_dict = {x_ : example_input})
test_output = test_output[2,:]
plt.figure()
plt.plot(test_output,label = 'Predicted')
plt.plot(model_output,label = 'Truth')
plt.legend()
plt.show()