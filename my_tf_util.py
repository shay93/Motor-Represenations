from __future__ import division 
import os
import numpy as np
import tensorflow as tf


class graph_construction_helper:

	def conv(self,x,weight_shape, scope, stddev = 0.1,trainable = True, reuse_variables = False):
		"""
		x should be the 4d tensor which is being convolved
		weight shape should be a list of the form [Kernel Width, Kernel Width, input channels, output channels]
		scope should be string specifying the scope of the variables in question
		"""
		with tf.variable_scope(scope) as scope:
			if not(reuse_variables):
				#initialize the weights for the convolutional layer
				W = tf.get_variable("W_conv",weight_shape,tf.float32,tf.random_normal_initializer(0.0,stddev),trainable = trainable)
				#initiaize the biases
				b = tf.get_variable("b_conv",weight_shape[-1],tf.float32,tf.constant_initializer(0.1),trainable = trainable)
			else:
				scope.reuse_variables()
				W = tf.get_variable("W_conv")
				b = tf.get_variable("b_conv")
			#calculate the output from the convolution 
			conv = tf.nn.conv2d(x,W,strides = [1,2,2,1],padding = "SAME")
			#compute the activations
			h = tf.nn.relu(tf.nn.bias_add(conv,b), name = "activations_conv")

		return h,W,b


	def fc_layer(self,x,weight_shape,scope, stddev = 0.1,trainable = True, reuse_variables = False):
		"""
		Compute the activations of the fc layer

		"""
		with tf.variable_scope(scope) as scope:
			if not(reuse_variables):
				#initialize the weights for the fc layer
				W = tf.get_variable("W_fc",weight_shape,tf.float32,tf.random_normal_initializer(0.0,stddev), trainable = trainable)
				#initiaize the biases
				b = tf.get_variable("b_fc",weight_shape[-1],tf.float32,tf.constant_initializer(0.0),trainable = trainable)
			else:
				scope.reuse_variables()
				W = tf.get_variable("W_fc")
				b = tf.get_variable("b_fc")

			h = tf.nn.relu(tf.matmul(x,W) + b, name = "activations_fc")

		return h,W,b


	def deconv(self,x,weight_shape,output_shape,scope,strides = [1,2,2,1], stddev = 0.1,trainable = True, reuse_variables = False,non_linearity = True):
		"""
		generalizable deconv function
		"""
		with tf.variable_scope(scope) as scope:
                    if not(reuse_variables):
				#initialize the weights for the deconv layer
				W = tf.get_variable("W_deconv",weight_shape,tf.float32,tf.random_normal_initializer(0,stddev), trainable = trainable)
				#initiaize the biases
				b = tf.get_variable("b_deconv",weight_shape[-2],tf.float32,tf.constant_initializer(0.1),trainable = trainable)
			else:
				scope.reuse_variables()
				W = tf.get_variable("W_deconv")
				b = tf.get_variable("b_deconv")

			#calculate the output from the deconvolution
			deconv = tf.nn.conv2d_transpose(x,W,output_shape,strides = strides)
			#calculate the activations
			if non_linearity:
				h = tf.nn.relu(tf.nn.bias_add(deconv,b), name = "activations_deconv_relu")
			else:
				h = tf.nn.bias_add(deconv,b, name = "activations_deconv")

		return h,W,b

class tensorflow_graph:
	#all graphs should be a subclass of the tensorflow graph object which has methods that can be used to train and evaluate the graph
	#so given the model the trainer or evaluator calls on the train_op in the graph to construct the graph
	def create_session(self):
		#initialize a configuration protobuf object that can be used to initialize a session
		config = tf.ConfigProto()
		#configure a sessions object such that the gpu usage grows
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)
		return sess


	def train_graph(self,sess,Epochs,batch_size,placeholder_dict,train_op,loss_op, merge_summary_op = None,log_dir = None,summary_writer_freq = 20):
		"""
		1)Epochs is the number determines the number of iterations and hence the number of times that parameter updates are made
		2)placeholder dict holds the placholder ops of the graph and the data that needs to be passed into the graph in order to compute gradients and train parameters
		3)placeholder dict is keyed by the tensorflow object and the dict holds the data set corresponding to the key
		4)init_op is used to initialize the variables in the graph based on the shape and datatypes that have been specified prior
		5)train_op is the training operation of the graph which is called upon when gradients are computed/applied
		"""
		#first get the number of samples by getting finding the shape of the first data variable
		num_samples = np.shape(placeholder_dict[placeholder_dict.keys()[0]])[0]
		#if we have a merge summary op initialize a summary writer
		if merge_summary_op is not(None):
			#initialize a summary writer and pass the graph to it
			summary_writer = tf.train.SummaryWriter(log_dir,sess.graph)


		#now loop through all the iterations compute,apply parameter updates and record values of interest
		for step in xrange(int(Epochs * num_samples) // batch_size):
			#first step is to get a random offset into the dataset
			random_offset = np.random.randint(0,num_samples - batch_size)
			#initialize an empty feed_dict for each step
			feed_dict = {}
			for op in placeholder_dict.keys():
				#construct a dictionary that stores the spliced input batch data keyed by the tensor placeholders
				feed_dict[op] = placeholder_dict[op][random_offset:random_offset + batch_size,...]
			#every 20 steps record the outputs from the summary if a merge_summary_op is provided
			if (step % summary_writer_freq == 0) and (merge_summary_op is not(None)):
				_,merged_summary,l = sess.run([train_op,merge_summary_op,loss_op], feed_dict = feed_dict)
				print l,step
				#pass the summary to the writer to record in the log file
				summary_writer.add_summary(merged_summary,step)
			else:
				#use the feed dict along with the train_op to compute, and apply gradients
				_ ,l = sess.run([train_op,loss_op],feed_dict = feed_dict)
				if (step % 20) == 0:
					print l,step

	def save_graph_vars(self,sess,save_op,save_directory):
		#save_directory = os.path.abspath(save_directory)
		save_op.save(sess,save_directory)
		print "Variables have been saved"


	def load_graph_vars(self,sess,save_op,load_path):
		load_path = os.path.abspath(load_path)
		save_op.restore(sess,load_path)
		print "Model Restored"

	def init_graph_vars(self,sess,init_op):
		#initialize the variables in the graph
		sess.run(init_op)
		print "Variables have been initialized"

	def evaluate_graph(self,sess,eval_batch_size,placeholder_dict,y_op,y_label_op, output_shape = None, loss_op = None):
		"""
		Pass in the eval set data to compute predictions, this function returns the predictions whatever those may be
		"""
		#first get the number of samples by getting finding the shape of the first data variable
		num_samples = np.shape(placeholder_dict[placeholder_dict.keys()[0]])[0]
		if num_samples < eval_batch_size:
			raise ValueError("batch size for evals larger than dataset: %d" % eval_batch_size)

		#initialize an empty array for predictions using the provided shape
		if (output_shape == None):
			predictions = np.ndarray(shape = np.shape(placeholder_dict[y_label_op]),dtype = np.float32)
		else:
			predictions = np.ndarray(shape = output_shape,dtype = np.float32)
		#furthermore initialize a list to record the test set loss
		test_loss_array = [0]*((num_samples // eval_batch_size) + 1)
		#initialize a variable to record how many eval iterations have taken place
		step = 0
		#loop through all the eval data
		for begin in xrange(0,num_samples,eval_batch_size):
			#specify the index of the end of the batch
			end = begin + eval_batch_size
			#now construct the feed dict based whether a whole batch is available or not
			feed_dict = {}
			if end <= num_samples:
				for op in placeholder_dict.keys():
					feed_dict[op] = placeholder_dict[op][begin:end, ...]
				if not(loss_op == None):
					predictions[begin:end,...],l = sess.run([y_op,loss_op],feed_dict = feed_dict)
				else:
					predictions[begin:end,...] = sess.run([y_op],feed_dict = feed_dict)
			else:
				for op in placeholder_dict.keys():
					feed_dict[op] = placeholder_dict[op][-eval_batch_size,...]
				if not(loss_op == None):
					batch_predictions,l = sess.run([y_op,loss_op], feed_dict = feed_dict)
				else:
					batch_predictions = sess.run([y_op], feed_dict = feed_dict)
				predictions[begin:, ...] = batch_predictions[-(num_samples - begin):,...]

			if not(loss_op == None):
				#append the loss
				test_loss_array[step] = l
			#increment the step variable
			step += 1
		return predictions,test_loss_array

	def add_placeholder_ops(self):
		raise NotImplementedError

	def add_model_ops(self):
		raise NotImplementedError

	def add_auxillary_ops(self):
		raise NotImplementedError

	def build_graph(self):
		#add the placeholder ops
		self.add_placeholder_ops()
		#add the model ops
		self.add_model_ops()
		#finally add all the training ops the summary ops the saver and initialization op
		self.add_auxillary_ops()
		#return the session environment in which the graph is defined and dictionary of the operations
		return self.op_dict,self.create_session()
