import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data = np.random.rand(15,15)
W_conv1 = tf.Variable(tf.truncated_normal([2,2,1,3])
b_conv1 = tf.Variable(tf.constant(0.1,shape = [3])
x = tf.placeholder('float',shape = [15,15])
x_reshape = tf.reshape(x,[1,15,15,1])
h_conv1 = tf.nn.tanh(tf.nn.conv2d(x_reshape,W_conv1,strides = [1,1,1,1],padding = 'SAME'))

h_conv1_flat = tf.reshape(h_conv1, shape = [1,15*15*3])
W_fc = tf.Variable(tf.truncated_normal(shape = [15*15*3,15*15])
h_fc = tf.Variable(tf.constant(0.1,shape = [15*15])
y = tf.reshape(h_fc,shape = [15,15])
sess = tf.InteractiveSession()
y_ = tf.placeholder('float', shape = [img_height,img_width])

cross_entropy = -tf.reduce_sum(y_*tf.log(y_fc1))
train_step = tf.train.Optimizer(1e-4).minimize(cross_entropy)
sess.run(tf.inialize_all_variables())
sess.run(train_step)
