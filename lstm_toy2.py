import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('Agg')

#generate some training and test data. Feed in a random sequence of numbers and specify output to be mx+c
BATCH_SIZE = 10
TOTAL_NUM_POINTS = 20
TIME_STEPS = TOTAL_NUM_POINTS
NUM_HIDDEN_UNITS = 10
FC_UNITS = 20	
learning_rate = 0.5 * 1e-3
EPOCHS = 10
NUM_OF_BATCHES = 1000
m = 5
c = 1

def transform(x):
	return 0.5*np.sin(x) + 0.25

def generate_data(num):
	x_data_list = [0] * num
	y_data_list = [0] * num
	for j in range(0,num):
		#generate randome sequence equal to number of points
		x_data = np.array([np.linspace(0,4*np.pi,num = TOTAL_NUM_POINTS)] * BATCH_SIZE)
		#x_data = np.random.rand(BATCH_SIZE,TOTAL_NUM_POINTS)
		y_data = transform(x_data)
		#y_data = np.reshape(y_data,[BATCH_SIZE,TOTAL_NUM_POINTS])
		x_data_list[j] = x_data
		y_data_list[j] = y_data
	
	return x_data_list,y_data_list

#now build the model that will be used to generate this data
#first specify place holders for input data
x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,TIME_STEPS])
#need to split the input so that it fits with model specification
x_input = tf.split(1,TIME_STEPS,x)
y_ = tf.placeholder(tf.float32,shape = [BATCH_SIZE,TIME_STEPS])
#now specify the RNN cell that will be used
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN_UNITS)
#now define the rnn around this that outputs tensors corresponding to outputs and states
outputs,states = tf.nn.rnn(lstm_cell, x_input,dtype = tf.float32)
#now reshape output such that it is same shape as placeholder for y
W_fc1 = tf.Variable(tf.truncated_normal(shape = [NUM_HIDDEN_UNITS, FC_UNITS],stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.0, shape = [FC_UNITS]))
fc_layer_1 = tf.nn.tanh(tf.matmul(outputs[-1],W_fc1) + b_fc1)
#Add a dropout layer
dropout_layer = tf.nn.dropout(fc_layer_1,keep_prob = 0.5)
#Add a second fully connected layer to the drop out layer from the layer one
W_fc2 = tf.Variable(tf.truncated_normal(shape = [FC_UNITS,TOTAL_NUM_POINTS],stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape = [TOTAL_NUM_POINTS]))
fc_layer_2 = tf.matmul(dropout_layer,W_fc2) + b_fc2
#Reshape output from layer 2
y = tf.reshape(fc_layer_2,[BATCH_SIZE,TOTAL_NUM_POINTS])
#Define the loss via the L2 norm of the difference
meansq = tf.reduce_mean(tf.square(y -y_)) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1)
#add training nodes based on this loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(meansq)
#initialize a session
sess = tf.Session()
#initialize variables using session
sess.run(tf.initialize_all_variables())


#generate data
x_data_list,y_data_list = generate_data(NUM_OF_BATCHES)
#initialize a list to add loss
loss_list = []
for step in range(0,EPOCHS):
	for batch_num in range(0,NUM_OF_BATCHES):
		x_data = x_data_list[batch_num]
		y_data = y_data_list[batch_num]
		loss,_ = sess.run([meansq,train_step],feed_dict = {x : x_data, y_ : y_data})
		if (step*NUM_OF_BATCHES + batch_num) % 20 == 0:
			print (step*NUM_OF_BATCHES + batch_num),loss
			loss_list.append(loss)


#now test the model
#generate random sequence equal to number of points
x_model,y_model = generate_data(1)
with open('y_model.csv','wb') as f:
	np.savetxt(f,np.array(y_model[0]),delimiter = ",")

y_pred = sess.run(y,feed_dict = {x : x_model[0]})
with open('y_pred.csv','wb') as f:
	np.savetxt(f,np.array(y_pred),delimiter = ",")

