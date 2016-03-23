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
output_data = np.divide(ouput,255)

# Autoencoder with 1 hidden layer
n_samp, n_input = input_data.shape
#set number of hidden layer units to 20 
n_hidden = 20

x = tf.placeholder("float", [None, n_input])
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
# Weights and biases to hidden layer
Wo = tf.transpose(Wh) # tied weights
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
# Objective functions
y_ = tf.placeholder("float", [None,n_input])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 500
batch_size = min(50, n_samp)

for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data[sample][:]
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})

print "Target:"
print output_data
print "Final activations:"
print sess.run(y, feed_dict={x: input_data})
print "Final weights (input => hidden layer)"
print sess.run(Wh)
print "Final biases (input => hidden layer)"
print sess.run(bh)
print "Final biases (hidden layer => output)"
print sess.run(bo)
print "Final activations of hidden layer"
print sess.run(h, feed_dict={x: input_data})