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
input_data = rgb2gray(plt.imread(image_str))
input_data = input_data.astype(np.float32)
#scale input data to in between 0 and 1 
input_data = np.divide(input_data,255)
img_height,img_width = np.shape(input_data)
#reshape image so that it is compatible with the tensor
input_data = tf.reshape(input_data, [-1,in_height,in_width,1])
#set output equal to the input data
output = input_data


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

#INITIALIZE VARIABLES IN ALL THE LAYERS
with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([img_height*img_width*32,1024])
    b_fc1 = bias_variable([1024])
    

#Now convolve image with weight tensor add bias and compute relu function
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

#compute output of fully connected layer
h_conv1_flat = tf.reshape(h_conv1, [-1, img_height*img_width*32])
y_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

y_ = tf.placeholder("float", shape=[img_height,img_width])
cross_entropy = -tf.reduce_sum(y_*tf.log(y_fc1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 2

for i in range(n_rounds):
    #sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_image
    batch_ys = input_image
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

print "Final activations:"
output_image = sess.run(y_conv, feed_dict={x: x_image})
print output_image
plt.imsave('output_image.jpg',255.*output_image,cmap='Greys_r')


