import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

#first read input image into an array

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

image_str = 'grey_cat.jpg'
img = rgb2gray(plt.imread(image_str))
input_data = img

#set output equal to the input data
output = input_data


#scale input data to in between 0 and 1 
input_data = np.divide(input_data,255)
output_data = np.divide(output,255)

#Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#first convolution layer initialize variables
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#get dimensions of input image
in_height,in_width = np.shape(input_data)
x = tf.placeholder("float", [-1,in_height,in_width,1])
#reshape input image to 4d tensor
x_image = tf.reshape(input_data, [-1,in_height,in_width,1])

#Now convolve image with weight tensor add bias and compute relu function
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Now initialize variables for second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#perform convolution for second layer and add bias and then pool
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#now add densely connected layer with 1024 units
W_fc1 = weight_variable([in_height*in_width*64./4, 1024])
b_fc1 = bias_variable([1024])

#compute output of fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, in_height*in_width*64./4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Add dropout layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Add readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_ = tf.placeholder("float", shape=(None,n_input))
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 1000



for i in range(n_rounds):
    #sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = x_image
    batch_ys = x_image
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    #if i % 100 == 0:
        #print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})

print "Target:"
print output_data
print "Final activations:"
output_image = sess.run(y_conv, feed_dict={x: x_image})
print output_image
plt.imsave('output_image.jpg',255.*output_image,cmap='Greys_r')
#print "Final weights (input => hidden layer)"
#print sess.run(Wh_1)
#print "Final biases (input => hidden layer)"
#print sess.run(bh)
print "Final biases (hidden layer => output)"
print sess.run(bo)

