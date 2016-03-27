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

# Autoencoder with 1 hidden layer
n_samp, n_input = input_data.shape
#set number of hidden layer units to 20 
n_hidden_1 = 200
n_hidden_2 = 200

#figure out the batch size based on which we may define the input and output
batch_size= min(50,n_samp)

x = tf.placeholder("float", shape=(None,n_input))
# Weights and biases to hidden layer
Wh_1 = tf.Variable(tf.random_uniform((n_input, n_hidden_1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh_1 = tf.Variable(tf.zeros([n_hidden_1]))
h_1 = tf.nn.tanh(tf.matmul(x,Wh_1) + bh_1)
# Weights and biases to hidden layer
Wh_2 = tf.Variable(tf.random_uniform((n_hidden_1,n_hidden_2),-1.0/math.sqrt(n_hidden_1),1.0/math.sqrt(n_hidden_2))) # tied weights
bh_2 = tf.Variable(tf.zeros([n_hidden_2]))
h_2 = tf.nn.tanh(tf.matmul(h_1,Wh_2) + bh_2)

#now add final set of weightsw to obtain the output
Wo = tf.Variable(tf.random_uniform((n_hidden_2,n_input),-1.0/math.sqrt(n_hidden_2),1.0/math.sqrt(n_input)))
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.matmul(h_2,Wo) + bo)
# Objective functions
y_ = tf.placeholder("float", shape=(None,n_input))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 5000


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
output_image = sess.run(y, feed_dict={x: input_data})
print output_image
plt.imsave('output_image.jpg',255.*output_image,cmap='Greys_r')
#print "Final weights (input => hidden layer)"
#print sess.run(Wh_1)
#print "Final biases (input => hidden layer)"
#print sess.run(bh)
print "Final biases (hidden layer => output)"
print sess.run(bo)
#print "Final activations of hidden layer"
#print sess.run(h, feed_dict={x: input_data})
