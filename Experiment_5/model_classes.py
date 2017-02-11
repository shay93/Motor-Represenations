from __future__ import division
import numpy as np
import tensorflow as tf
import os

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

	def build_graph(self):
		#add the placeholder ops
		self.add_placeholder_ops()
		#add the model ops
		self.add_model_ops()
		#finally add all the training ops the summary ops the saver and initialization op
		self.add_auxillary_ops()
		#return the session environment in which the graph is defined and dictionary of the operations
		return self.op_dict,self.create_session()

class physics_emulator_fixed_joint_seq(tensorflow_graph):

	def __init__(self,learning_rate = 1e-3):
		self.output_image_decoder_parameters = {"deconv_output_channels_1" : 16, "deconv_output_channels_2" : 8, "deconv_output_channels_3" : 4, "deconv_output_channels_4" : 1}
		self.lr = learning_rate
		self.gc = graph_construction_helper()
		
		###initialize dictionaries to store the model's operations, variables and resulting activations
		self.op_dict = {}
		self.var_dict = {}
		self.activation_dict = {}

	def add_placeholder_ops(self):
		#initialize placeholder for joint angle inputs
		self.op_dict["x"] = tf.placeholder(tf.float32,shape = [None,3])
		#initialize placeholder for labels
		self.op_dict["y_"] = tf.placeholder(tf.float32, shape = [None,64,64,1])
		return self.op_dict

	def add_model_ops(self,reuse_variables = False):
		h1_fc,W_fc1,b_fc1 = self.gc.fc_layer(self.op_dict["x"],[3,32*4*4],"fc_layer_1", reuse_variables = reuse_variables)
		#now reshape this to get a 4d image
		h1_fc_reshaped = tf.reshape(h1_fc,shape = [-1,4,4,32])
		#find the batch size of the input data in order to use later
		batch_size = tf.shape(h1_fc_reshaped)[0]
		#pass this to a succession of deconv layers in order to get the desired image
		h_deconv1,W_deconv1,b_deconv1 = self.gc.deconv(h1_fc_reshaped,[2,2,self.output_image_decoder_parameters['deconv_output_channels_1'],32],[batch_size,4,4,self.output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1], reuse_variables = reuse_variables)
		#calculate activations for second deconv layer
		h_deconv2,W_deconv2,b_deconv2 = self.gc.deconv(h_deconv1,[3,3,self.output_image_decoder_parameters['deconv_output_channels_2'],self.output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,self.output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2",reuse_variables = reuse_variables)
		#calculate activations for third deconv layer
		h_deconv3,W_deconv3,b_deconv3 = self.gc.deconv(h_deconv2,[3,3,self.output_image_decoder_parameters['deconv_output_channels_3'],self.output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,self.output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3",reuse_variables = reuse_variables)
		#calculate activations for fourth deconv layer
		h_deconv4,W_deconv4,b_deconv4 = self.gc.deconv(h_deconv3,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4",reuse_variables = reuse_variables)
		#calculate activations for fifth deconv layer
		h_deconv5,W_deconv5,b_deconv5 = self.gc.deconv(h_deconv4,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv5", non_linearity = False, reuse_variables = reuse_variables)
		self.op_dict["y_logits"] = h_deconv5
		self.op_dict["y"] = tf.nn.sigmoid(h_deconv5)
		var_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5,W_fc1,b_fc1]
		for var in var_list:
			self.var_dict[var.name] = var

		#specify the activation list
		activation_list = [h1_fc,h_deconv1,h_deconv2,h_deconv3,h_deconv4,h_deconv5]

		for act in activation_list:
			self.activation_dict[act.name] = act
		
		return self.op_dict
	
	def add_auxillary_ops(self):
		opt = tf.train.AdamOptimizer(self.lr)
		#define the loss op using the y before sigmoid and in the cross entropy sense
		self.op_dict["loss"] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.op_dict["y_logits"],self.op_dict["y_"]/20.))
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],self.var_dict.values())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item in self.var_dict.items()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		#add the training operation to the graph
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		#add the initialization operation
		self.op_dict["init_op"] = tf.initialize_all_variables()
		#add a saving operation
		self.op_dict["saver"] = tf.train.Saver(self.var_dict)
		return self.op_dict
