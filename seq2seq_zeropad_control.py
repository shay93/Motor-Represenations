
from __future__ import division
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import training_tools as tt


#generate time array

BATCH_SIZE = 256
HIDDEN_UNITS = 200
EVAL_BATCH_SIZE = 256

OUTPUT_FEATURES = 3
INPUT_FEATURES = 2
LEARNING_RATE = 1e-4

VALIDATION_SIZE = 256
EPOCHS = 200
ROOT_DIR = "home/shay93/Motor-Represenations/Seq2Seq_Outputs/"

EVAL_FREQUENCY = 20
Layers = 5


link_length_2 = 60
link_length_1 = 50
arm1_path = "/home/shayaan/Research/Redwood/Motor-Represenations/Training_Data_First_Arm/"
arm2_path = "/home/shayaan/Research/Redwood/Motor-Represenations/Training_Data_Second_Arm/"
shape_str_array = ['Rectangle', 'Square', 'Triangle']

data_dict = {}

def load_data():
	for shape_name in shape_str_array:
		
		with open("Training_Data_First_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_1) + '.npy',"rb") as f:
			key = "arm1_" + shape_name
			data_dict[key] = pickle.load(f)
			f.close()

		with open("Training_Data_3DOF_Arm/" + shape_name + '.npy',"rb") as f:
			key = "arm2_" + shape_name
			data_dict[key] = pickle.load(f)
			f.close()
	return data_dict


def find_seq_max_length(data_dict):
	"""
	loop through the lists in the data list in order to figure out the maximum length of each sequence
	"""
	max_length = 1
	for sequence_list in data_dict.values():
		for sequence in sequence_list:
			if np.shape(sequence)[1] > max_length:
				max_length = np.shape(sequence)[1]

	return max_length


def extract_data(num,data_dict,max_seq_length):

	control_array_arm1 = np.zeros([num,max_seq_length,INPUT_FEATURES])
	control_array_arm2 = np.zeros([num,max_seq_length,OUTPUT_FEATURES])
	length_array = np.ndarray(num)
	for image_control_num in xrange(num):
		#figure out which shape control needs to be loaded
		shape_name_index = image_control_num % len(shape_str_array)
		#next figure out the index of the shape being read in i.e. is it Triangle1 or Triangle100
		shape_index = image_control_num // len(shape_str_array)
		#find the sequence length for the sample being loaded
		length_array[image_control_num] = np.shape(data_dict["arm1_" + shape_str_array[shape_name_index]][shape_index])[1]
		#this information may now be combined to load the right control space value
		control_array_arm1[image_control_num, :length_array[image_control_num], :] = np.transpose(data_dict["arm1_" + shape_str_array[shape_name_index]][shape_index])
		control_array_arm2[image_control_num, :length_array[image_control_num], :] = np.transpose(data_dict["arm2_" + shape_str_array[shape_name_index]][shape_index])
			

	return control_array_arm1,control_array_arm2,length_array

data_dict = load_data()
SEQ_MAX_LENGTH = find_seq_max_length(data_dict)
train_x_data,train_y_data,length_array = extract_data(3000,data_dict,SEQ_MAX_LENGTH)



def get_bin_array(length_array):
	"""
	This function should take the length array as an input and zero outputs for which the sequence ends
	"""
	#initialize the output bin array
	bin_array = np.zeros((len(length_array),SEQ_MAX_LENGTH,3))
	for i,sample_length in enumerate(length_array):
		bin_array[i,:sample_length,0] = np.ones(sample_length,dtype = np.float32)
		bin_array[i,:sample_length,1] = np.ones(sample_length,dtype = np.float32)
		bin_array[i,:sample_length,2] = np.ones(sample_length,dtype = np.float32)		
	return bin_array

bin_array = get_bin_array(length_array)
print np.shape(bin_array)
#generate a validation set
validation_x_data = train_x_data[:VALIDATION_SIZE, ...]
validation_y_data = train_y_data[:VALIDATION_SIZE, ...]
validation_tstep_array = length_array[:VALIDATION_SIZE]
validation_bin_array = bin_array[:VALIDATION_SIZE, ...]
#get the training data required 
train_x_data = train_x_data[VALIDATION_SIZE:, ...]
train_y_data = train_y_data[VALIDATION_SIZE:, ...]
training_tstep_array = length_array[VALIDATION_SIZE:]
training_bin_array = bin_array[VALIDATION_SIZE:,...]


num_epochs = EPOCHS
train_size = train_x_data.shape[0]



#build the graph 
bin_tensor = tf.placeholder(tf.float32, shape = [None,SEQ_MAX_LENGTH,OUTPUT_FEATURES])
x = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH,INPUT_FEATURES])
y_ = tf.placeholder(tf.float32,shape = [None,SEQ_MAX_LENGTH,OUTPUT_FEATURES])
seq_length = tf.placeholder(tf.float32, shape = [None]) #this specifies the seq length for each sample in the batch
# x will have to be split into a list of size SEQ_MAX_LENGTH with each element of size [Batch Size, 1]
x_transpose = tf.transpose(x,[1,0,2])
#reshape x
x_reshape = tf.reshape(x_transpose, [-1,INPUT_FEATURES])
#initialize an lstm cell
lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS,state_is_tuple = True)] * Layers
#pass this list into a multirnn_cell
lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list, state_is_tuple = True)
#split the x input into a list as specified above
x_list = tf.split(0,SEQ_MAX_LENGTH,x_reshape)
#now pass this into a function that generates the rnn graph
outputs,states = tf.nn.rnn(lstm_cell,x_list,dtype = tf.float32)
#now we want to take each element of this output list and pass it into fc layer 
W_fc = tf.Variable(tf.truncated_normal(shape = [HIDDEN_UNITS,OUTPUT_FEATURES],stddev = 0.1))
b_fc = tf.Variable(tf.constant(0., shape = [OUTPUT_FEATURES]))
#loop through the outputs
#but first initialize y_list
#while samples in the batch remain and while timesteps in the sample remain compute the output and append zeros to the output
y_list = []
for output_timestep in outputs:
	y_timestep = tf.nn.elu(tf.matmul(output_timestep,W_fc) + b_fc)
	y_list.append(y_timestep)

#now combine the elements in y_list to get a tensor of shape [SEQ_MAX_LENGTH,None,OUTPUT_FEATURES]
y_tensor = tf.pack(y_list)
y = tf.mul(tf.transpose(y_tensor, [1,0,2]),bin_tensor)
#now define a loss node for training
meansq = tf.reduce_mean(tf.square(y - y_))
#define a training node
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(meansq)


def eval_in_batches(x_data,y_data,len_data,sess):
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
				predictions[begin:end, ...],l = sess.run([y,meansq],feed_dict={x : x_data[begin:end, ...], y_ : y_data[begin:end, ...], bin_tensor: len_data[begin:end, ...]})
			else:
				batch_prediction,l = sess.run([y,meansq],feed_dict = {x : x_data[-EVAL_BATCH_SIZE:, ...], y_ : y_data[-EVAL_BATCH_SIZE:, ...], bin_tensor : len_data[-EVAL_BATCH_SIZE, ...]})
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
			batch_bin_array = training_bin_array[offset:(offset + BATCH_SIZE), ...]

			feed_dict = { x: batch_x_data, y_ : batch_y_data, bin_tensor : batch_bin_array}
			#run the graph
			_, l= sess.run(
				[train_op,meansq],
				feed_dict=feed_dict)
			loss_array[step] = l

			if step % EVAL_FREQUENCY == 0:
				predictions,test_loss_array = eval_in_batches(validation_x_data,validation_y_data, validation_bin_array,sess)
				print step,l
		 
		return loss_array,predictions


def visualize_predictions(predictions,arm2):
	'''
	Takes in a numpy array of control predictions applies forward kinematics to them and visualize the output
	arm2 is training_tools arm object
	'''
	size = np.shape(predictions)[0]
	for i in range(size):
		state_sequence = np.squeeze(np.transpose(predictions[i,:,:]))
		#cut off the state sequence before the zeros
		state_sequence = state_sequence[:,:length_array[i]]
		pos_array = arm2.forward_kinematics(state_sequence)
		#get the shape index and shape name using i
		shape_index = i // len(shape_str_array)
		shape_name = shape_str_array[i % 3]
		#use this information to initialize a grid using training tools
		my_grid = tt.grid(shape_name + str(shape_index), ROOT_DIR)
		my_grid.draw_figure(pos_array)
		my_grid.save_image()

def visualize_validation_data(validation_data,arm):
	size = np.shape(validation_data)[0]
	for i in range(size):
		state_sequence = np.squeeze(np.transpose(validation_data[i,:,:]))
		#cut off the state sequence before the zeros
		state_sequence = state_sequence[:,:length_array[i]]
		pos_array = arm.forward_kinematics(state_sequence)
		#get the shape index and shape name using i
		shape_index = i // len(shape_str_array)
		shape_name = shape_str_array[i % 3]
		#use this information to initialize a grid using training tools
		my_grid = tt.grid(shape_name + str(shape_index) + "_label", ROOT_DIR)
		my_grid.draw_figure(pos_array)
		my_grid.save_image()






loss_array,predictions = train_graph()
#initialize arm2
arm2 = tt.three_link_arm(link_length_2)
visualize_predictions(predictions,arm2)
visualize_validation_data(validation_y_data,arm2)
#plt.plot(loss_array)
#plt.xlabel("batch")
#plt.ylabel("meansq loss")
#plt.show()
