
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


#generate time array

SEQ_MAX_LENGTH = 200
BATCH_SIZE = 100
HIDDEN_UNITS = 100
EVAL_BATCH_SIZE = 50 

OUTPUT_FEATURES = 1
INPUT_FEATURES = 1
LEARNING_RATE = 1e-3

VALIDATION_SIZE = 200
NUM_OF_WAVES = 3000
EPOCHS = 20
DISPLAY_NUM_EXAMPLES = 10
ROOT_DIR = "Logs/"

EVAL_FREQUENCY = 20
#generate time array  


time_array = np.linspace(0,4*np.pi,num = SEQ_MAX_LENGTH)

def get_training_data(num_of_waves):
    """
    generate all the waves that may then be split into batches
    """
    sin_wave_array = np.zeros([num_of_waves,SEQ_MAX_LENGTH,1])
    cos_wave_array = np.zeros([num_of_waves,SEQ_MAX_LENGTH,1])

    #generate a random phase shift array
    phase_array = np.pi*np.random.rand(num_of_waves)
    #keep amplitude and frequency constant at 1
    #generate random lengths as well
    length_array = np.round(SEQ_MAX_LENGTH*np.random.rand(num_of_waves))
    for i in range(num_of_waves):
        sin_wave_array[i,:length_array[i],0] = np.sin(time_array[:length_array[i]] - phase_array[i])
        cos_wave_array[i,:length_array[i],0] = np.cos(time_array[:length_array[i]] - phase_array[i])

    return sin_wave_array,cos_wave_array,length_array

train_x_data,train_y_data,termination_tstep_array = get_training_data(NUM_OF_WAVES)
#generate a validation set
validation_x_data = train_x_data[:VALIDATION_SIZE, ...]
validation_y_data = train_y_data[:VALIDATION_SIZE, ...]
validation_tstep_array = termination_tstep_array[:VALIDATION_SIZE]
#get the training data required 
train_x_data = train_x_data[VALIDATION_SIZE:, ...]
train_y_data = train_y_data[VALIDATION_SIZE:, ...]
training_tstep_array = termination_tstep_array[VALIDATION_SIZE:]

num_epochs = EPOCHS
train_size = train_x_data.shape[0]

#build the graph 
x = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH,INPUT_FEATURES])
y_ = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH,OUTPUT_FEATURES])
seq_length = tf.placeholder(tf.float32, shape = [None]) #this specifies the seq length for each sample in the batch
# x will have to be split into a list of size SEQ_MAX_LENGTH with each element of size [Batch Size, 1]
x_transpose = tf.transpose(x,[1,0,2])
#reshape x
x_reshape = tf.reshape(x_transpose, [-1,INPUT_FEATURES])
#initialize an lstm cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS, state_is_tuple = True)
#split the x input into a list as specified above
x_list = tf.split(0,SEQ_MAX_LENGTH,x_reshape)
print len(x_list)
#now pass this into a function that generates the rnn graph
outputs,states = tf.nn.rnn(lstm_cell,x_list,dtype = tf.float32)
#now we want to take each element of this output list and pass it into fc layer 
W_fc = tf.Variable(tf.truncated_normal(shape = [HIDDEN_UNITS,OUTPUT_FEATURES],stddev = 0.1))
b_fc = tf.Variable(tf.constant(0., shape = [OUTPUT_FEATURES]))
#loop through the outputs
#but first initialize y_list
#while samples in the batch remain and while timesteps in the sample remain compute the output and append zeros to the output
output_tensor = tf.pack(outputs)
#now slice this to get not only a single sample lets deal with the fist sample first
for output_timestep in outputs:
	y_timestep = tf.nn.elu(tf.matmul(output_timestep,W_fc) + b_fc)
	y_list.append(y_timestep)

#now combine the elements in y_list to get a tensor of shape [SEQ_MAX_LENGTH,None,OUTPUT_FEATURES]
y_tensor = tf.pack(y_list)
y = tf.transpose(y_tensor, [1,0,2])
#now define a loss node for training
meansq = tf.reduce_mean(tf.square(y - y_))
#define a training node
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(meansq)




def eval_in_batches(x_data,y_data,sess):
		"""Get combined loss for dataset by running in batches"""
		size = x_data.shape[0]

		if size < EVAL_BATCH_SIZE:
			raise ValueError("batch size for evals larger than dataset: %d" % size)

		predictions = np.ndarray(shape = (size,SEQ_MAX_LENGTH,OUTPUT_FEATURES), dtype = np.float32)
		test_loss_array = [0] * ((size // EVAL_BATCH_SIZE) + 1)
		i = 0
		for begin in xrange(0,size,EVAL_BATCH_SIZE):
			end = begin + EVAL_BATCH_SIZE
			
			if end <= size:
				predictions[begin:end, ...],l = sess.run([y,meansq],feed_dict={x : x_data[begin:end, ...], y_ : y_data[begin:end, ...]})
			else:
				batch_prediction,l = sess.run([y,meansq],feed_dict = {x : x_data[-EVAL_BATCH_SIZE:, ...], y_ : y_data[-EVAL_BATCH_SIZE:, ...]})
				predictions[begin:, ...] = batch_prediction[(begin - size):,...]

			test_loss_array[i] = l
			i += 1
		return predictions,test_loss_array


def train_graph():
	"""
	Tune the weights of the graph so that you can learn the right results
	inputs: A sessions object and an operation dictionary along with an integer specifying the end of the training data
	outputs: a loss array for the purposes of plotting
	"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:
		#initialize the variables
		log_dir = ROOT_DIR + "/tmp/summary_logs"
		train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

		tf.initialize_all_variables().run()
		#initialize a training loss array
		loss_array = [0] * (int(num_epochs * train_size) // BATCH_SIZE)
		for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
			#compute the offset of the current minibatch in the data
			offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
			batch_x_data = train_x_data[offset:(offset + BATCH_SIZE), ...]
			batch_y_data = train_y_data[offset:(offset + BATCH_SIZE), ...]
			feed_dict = { x: batch_x_data, y_ : batch_y_data}
			#run the graph
			_, l = sess.run(
				[train_op,meansq],
				feed_dict=feed_dict)
			loss_array[step] = l


			if step % EVAL_FREQUENCY == 0:
				predictions,test_loss_array = eval_in_batches(validation_x_data,validation_y_data, sess)
				print step,l
		
		output_examples = validation_y_data[:DISPLAY_NUM_EXAMPLES]
		output_predictions = predictions[:DISPLAY_NUM_EXAMPLES] 
		f, axarr = plt.subplots(2, DISPLAY_NUM_EXAMPLES, sharex='col',sharey='row')
		for i in xrange(DISPLAY_NUM_EXAMPLES):
			axarr[0,i].plot(time_array,output_examples[i,:],label = 'label')
			axarr[0,i].plot(time_array,predictions[i,:], color = 'r', label = 'predictions')
			axarr[1,i].plot(time_array,predictions[i,:])
		return loss_array


loss_array = train_graph()
plt.show()