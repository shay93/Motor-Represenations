import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#generate some training and test data. Feed in a random sequence of numbers and specify output to be mx+c
BATCH_SIZE = 10
TOTAL_NUM_POINTS = 20
TIME_STEPS = TOTAL_NUM_POINTS
NUM_HIDDEN_UNITS = 1000	
learning_rate = 1e-4
EPOCHS = 3
NUM_OF_BATCHES = 1000
m = 5
c = 1

def transform(x):
	return 0.5*np.sin(x)

def generate_data(num):
	x_data_list = [0] * num
	y_data_list = [0] * num
	for j in range(0,num):
		#generate randome sequence equal to number of points
		x_data = np.array([np.linspace(0,10*np.pi,num = TOTAL_NUM_POINTS) + 0.3*(np.random.rand(TOTAL_NUM_POINTS)-0.5)] * BATCH_SIZE)
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
W_fc = tf.Variable(tf.truncated_normal(shape = [NUM_HIDDEN_UNITS,TOTAL_NUM_POINTS]))
b_fc = tf.Variable(tf.constant(0.1, shape = [TOTAL_NUM_POINTS]))
fc_layer = (tf.matmul(outputs[-1],W_fc) + b_fc)
dropout_layer = tf.nn.dropout(fc_layer,keep_prob = 0.6)
y = tf.reshape(dropout_layer,[BATCH_SIZE,TOTAL_NUM_POINTS])
#Define the loss via the L2 norm of the difference
meansq = tf.reduce_mean(tf.square(y -y_))
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
plt.plot(y_model[0][1,:] , label = 'Model')
y_pred = sess.run(y,feed_dict = {x : x_model[0]})
test = plt.plot(y_pred[1,:], label = 'Predicted')
plt.legend()
plt.imsave("test.png",test)

plt.figure()
loss_plot = plt.plot(loss_list, label = "loss")
plt.legend()
plt.imsave("loss.png",loss_plot)
