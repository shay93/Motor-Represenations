import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.int32,shape = [None,5])
c = lambda i,bin_array : (i < tf.shape(x)[0])[0]
length = tf.constant([1,2,3,4,5])
sess = tf.InteractiveSession()

def body(i,bin_array):
	length_slice = tf.slice(length,i,tf.constant([1]))[0]
	one_array = tf.ones([length_slice])
	print one_array
	zero_array = tf.zeros([5 - length_slice])
	print zero_array
	row = tf.concat(0,[one_array,zero_array])
	row_expanded = tf.expand_dims(row, 0)
	bin_array = tf.concat(1,[bin,row_expanded])
	i += 1
	return i,bin_array

i = tf.constant([1])
bin_array = tf.expand_dims(tf.concat(0,[tf.ones([1]), tf.zeros([5 - 1])]), 0)
loop_vars = [i,bin_array]
b = lambda i,bin_array : body(i,bin_array)
r = tf.while_loop(c,b,loop_vars)

data = np.random.randint(1,7,size = [5,5])
print data


sess.eval(r[1],feed_dict = {x : data})
