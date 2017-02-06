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
	
	def evaluate_graph(self,sess,eval_batch_size,placeholder_dict,y_op,loss_op,y_label_op):
		"""
		Pass in the eval set data to compute predictions, this function returns the predictions whatever those may be
		"""
		#first get the number of samples by getting finding the shape of the first data variable
		num_samples = np.shape(placeholder_dict[placeholder_dict.keys()[0]])[0]
		if num_samples < eval_batch_size:
			raise ValueError("batch size for evals larger than dataset: %d" % eval_batch_size)

		#initialize an empty array for predictions using the provided shape
		predictions = np.ndarray(shape = np.shape(placeholder_dict[y_label_op]),dtype = np.float32)
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
				predictions[begin:end,...],l = sess.run([y_op,loss_op],feed_dict = feed_dict) 
			else:
				for op in placeholder_dict.keys():
					feed_dict[op] = placeholder_dict[op][-eval_batch_size,...]
				batch_predictions,l = sess.run([y_op,loss_op], feed_dict = feed_dict)
				predictions[begin:, ...] = batch_predictions[-(num_samples - begin):,...]

			#append the loss
			test_loss_array[step] = l
			#increment the step variable
			step += 1
		return predictions,test_loss_array

class physics_emulator(tensorflow_graph):
	

	def __init__(self,lr = 1e-3):
		#initialize a graph constructor helper object
		self.gc = graph_construction_helper()
		#initialize parameter dictionaries which will be used to construct the graph
		#self.observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4}
		#hyper parameters required to encode the joint angle state
		self.joint_encoder_parameters = {"fc_1" : 200 , "fc_2" : 56}
		#hyper parameters required to encode the previous output image
		self.output_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 1024 - 56}
		#hyper parameters required to decode the amalgam of the previous output image and current one into a new output image
		self.output_image_decoder_parameters = {"deconv_output_channels_1" : 32, "deconv_output_channels_2" : 16, "deconv_output_channels_3" : 8, "deconv_output_channels_4" : 4, "deconv_output_channels_5" : 1}
		#specify the number of dof's for the arm for which the physics emulator is learning the forward kinematics
		self.dof = 3
		#define the learning rate
		self.lr = lr
		#initialize an empty operation dictionary to store the variables
		self.op_dict = {}
		#initialize a var dict
		self.var_dict = {}

	def add_placeholder_ops(self):
		#add a placeholder for the previous output image
		self.op_dict["x_image"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		#add a placeholder for the input joint angle state
		self.op_dict["joint_angle_state"] = tf.placeholder(tf.float32, shape = [None,self.dof])
		#now add a placholder for the observed image
		self.op_dict["y_"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		return self.op_dict


	def encode_previous_output_image(self,previous_image, reuse_variables):
		"""
		Forms a representation of the output image provided at the previous step
		"""

		#find the activations of the first conv layer
		h_conv1,W_conv1,b_conv1 = self.gc.conv(previous_image,[3,3,1,self.output_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_output", reuse_variables = reuse_variables)
		#find the activations of the second conv layer
		h_conv2,W_conv2,b_conv2 = self.gc.conv(h_conv1,[3,3,self.output_image_encoder_parameters["conv1_kernels"],self.output_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_output", reuse_variables = reuse_variables)
		#find the activations of the third conv layer
		h_conv3,W_conv3,b_conv3 = self.gc.conv(h_conv2,[3,3,self.output_image_encoder_parameters["conv2_kernels"],self.output_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_output", reuse_variables = reuse_variables)
		#find the activations of the second conv layer
		h_conv4,W_conv4,b_conv4 = self.gc.conv(h_conv3,[3,3,self.output_image_encoder_parameters["conv3_kernels"],self.output_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_output", reuse_variables = reuse_variables)
		#find the activations of the second conv layer
		h_conv5,W_conv5,b_conv5 = self.gc.conv(h_conv4,[3,3,self.output_image_encoder_parameters["conv4_kernels"],self.output_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_output", reuse_variables = reuse_variables)
		#flatten the activations in the final conv layer in order to obtain an output image
		h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*self.output_image_encoder_parameters["conv5_kernels"]])
		#pass flattened activations to a fully connected layer
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(h_conv5_reshape,[4*self.output_image_encoder_parameters["conv5_kernels"],self.output_image_encoder_parameters["fc_1"]],"fc_layer_encode_output", reuse_variables = reuse_variables)
		output_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]

		return h_fc1,output_image_encoder_variable_list 

	def encode_joints(self,x_joints, reuse_variables):
		"""
		Takes joint states and encodes them in order to generate a new point in the output image
		"""
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(x_joints,[self.dof,self.joint_encoder_parameters["fc_1"]],"fc_joint_encoder_1", reuse_variables = reuse_variables)
		#pass the activations to a second fc layer
		h_fc2,W_fc2,b_fc2 = self.gc.fc_layer(h_fc1,[self.joint_encoder_parameters["fc_1"], self.joint_encoder_parameters["fc_2"]],"fc_joint_encoder_2", reuse_variables = reuse_variables)
		joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]

		return h_fc2,joint_encoder_variable_list


	def decode_outputs(self,hidden_vector, reuse_variables):
		"""
		Combines the information provided by the joint angles and the previous output image to produce a new image
		Take in a tensor of size [None, FC_UNITS_JOINTS + FC_UNITS_IMAGE]
		and generate an image of size [None,64,64,1]
		"""	
		#find the batch size of the input data in order to use later
		batch_size = tf.shape(hidden_vector)[0]
		#reshape the hidden activation vector into a 4d image that can be deconvolved to form an image
		hidden_image = tf.reshape(hidden_vector, shape = [batch_size,4,4,64])
		#calculate activations for the first deconv layer
		h_deconv1,W_deconv1,b_deconv1 = self.gc.deconv(hidden_image,[2,2,self.output_image_decoder_parameters['deconv_output_channels_1'],64],[batch_size,4,4,self.output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1], reuse_variables = reuse_variables)
		#calculate activations for second deconv layer
		h_deconv2,W_deconv2,b_deconv2 = self.gc.deconv(h_deconv1,[3,3,self.output_image_decoder_parameters['deconv_output_channels_2'],self.output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,self.output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2", reuse_variables = reuse_variables)
		#calculate activations for third deconv layer
		h_deconv3,W_deconv3,b_deconv3 = self.gc.deconv(h_deconv2,[3,3,self.output_image_decoder_parameters['deconv_output_channels_3'],self.output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,self.output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3", reuse_variables = reuse_variables)
		#calculate activations for fourth deconv layer
		h_deconv4,W_deconv4,b_deconv4 = self.gc.deconv(h_deconv3,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4", reuse_variables = reuse_variables)
		#calculate activations for fifth deconv layer
		h_deconv5,W_deconv5,b_deconv5 = self.gc.deconv(h_deconv4,[3,3,self.output_image_decoder_parameters['deconv_output_channels_5'],self.output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,self.output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5",non_linearity = False, reuse_variables = reuse_variables)
		decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

		return h_deconv5,decoder_variable_list


	def jointangle2image(self,joint_angle,previous_image, reuse_variables):
		"""
		Calls on the respective decoder and encoders in order to map a joint angle state to an output image joint_angle and previous image are both tensors
		"""
		encoded_joint_angle,joint_encoder_variable_list = self.encode_joints(joint_angle, reuse_variables)
		previous_image_encoded,image_encode_variable_list = self.encode_previous_output_image(previous_image, reuse_variables)
		#now concatenate to obtain encoded vector
		encoded_vector = tf.concat(1,[encoded_joint_angle,previous_image_encoded])
		#pass to a decoder in order to get the output
		y_logits,decoder_variable_list = self.decode_outputs(encoded_vector,reuse_variables)
		return y_logits,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list

	def add_model_ops(self,reuse_variables = False):
		#pass the input image and joint angle tensor to jointangle2image to get y_before_sigmoid
		self.op_dict["y_logits"],joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list = self.jointangle2image(self.op_dict["joint_angle_state"],self.op_dict["x_image"], reuse_variables)
		#apply sigmoid to get y
		self.op_dict["y"] = tf.nn.sigmoid(self.op_dict["y_logits"])
		#define the loss op using the y before sigmoid and in the cross entropy sense
		var_list = joint_encoder_variable_list+image_encode_variable_list+decoder_variable_list 
		for var in var_list:
			self.var_dict[var.name] = var
		return self.op_dict


	def add_auxillary_ops(self):
		opt = tf.train.AdamOptimizer(self.lr)
		#define the loss op using the y before sigmoid and in the cross entropy sense
		self.op_dict["loss"] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.op_dict["y_logits"],self.op_dict["y_"]/255.))
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

class one_layer_fc(tensorflow_graph):

	def __init__(self,fc_units,learning_rate):
		self.fc_units = fc_units
		self.lr = learning_rate
		#initialize a graph constructor object
		self.gc = graph_construction_helper()
		#intitialize dictionaries to store the operations of the graph as well as the network activations and variables
		#graph operations may refer to any intermediate tensors in graph computations
		self.op_dict = {}
		self.activation_dict = {}
		self.var_dict = {}
	
	def add_placeholder_ops(self):
		"""
		Add input and output placeholders for data to be fed into and read from the model
		furthermore add these placeholders to the op_dict and their key will always be "x" and "y_"
		"""
		#initialize a placedholder for the output and the input
		self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None,1])
		#now a placeholder for the labels
		self.op_dict['y_'] = tf.placeholder(tf.float32, shape = [None,1])
		return self.op_dict

	def add_model_ops(self):
		"""
		This function will add the operations to the graph which will be responsible for feedforward computations
		This function will add the final output tensor to the op dict
		This will also add the intermediate activations to the activation dict
		It will finally add the variable tensors required for computations to the var dict
		"""
		#Add the first layer which is a fc layer
		h1,W1,b1 = self.gc.fc_layer(self.op_dict['x'],[1,self.fc_units],"fc_layer",stddev = 0.5)
		#add another layer of readout neurons get the output activations
		self.op_dict["y"],W2,b2 = self.gc.fc_layer(h1,[self.fc_units,1],"readout",stddev = 0.5)
		#now add the weight and bias variables to the var dict but in order to make this process easier first list all the variabls
		var_list = [W1,b1,W2,b2]
		#loop through the list and add the variables the dict keyed by their previously specified name
		for var in var_list:
			self.var_dict[var.name] = var
		#add the intermediate activations to a list
		activation_list = [h1]
		for act in activation_list:
			self.activation_dict[act.name] = act

		return self.op_dict

	def add_auxillary_ops(self):
		"""
		Add operations which are not central to the forward model these operations and tensors include
		those pertaining to training ops via back prop as well as summary ops
		add a loss op the op_dict as well as a train op a merge summary op and an init_op
		"""
		#define a means square loss between the predicted and label
		self.op_dict["loss"] = tf.reduce_mean(tf.square(self.op_dict["y"] - self.op_dict["y_"]))
		#intialize the optimizer
		opt = tf.train.AdamOptimizer(self.lr)
		#compute gradients for variables in 
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],self.var_dict.values())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item in self.var_dict.items() ]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		#add a training operation
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		#add an initialization operation
		self.op_dict["init_op"] = tf.initialize_all_variables()
		#add a save op
		self.op_dict["save_op"] = tf.train.Saver(self.var_dict)
		return self.op_dict

	def build_graph(self):
		"""
		Should call on the other functions and construct the graph which consists of the placeholder ops
		and add the model graph ops and then the auxillary ops
		"""
		#add placeholder ops
		self.add_placeholder_ops()
		#add model ops
		self.add_model_ops()
		#add training ops
		self.add_auxillary_ops()
		return self.op_dict,self.create_session()

class onetstep_observed_to_output(tensorflow_graph):

	def __init__(self,learning_rate = 1e-3, gc = graph_construction_helper()):
		self.lr = learning_rate
		self.gc = gc
		self.op_dict = {}
		self.var_dict = {}
		self.activation_dict = {}
		self.dof = 3
		self.observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 20}


	def add_placeholder_ops(self):
		#add placeholder for observed image at first timestep
		self.op_dict["x_1"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		#add another placeholder for the logits used in calculating the loss
		self.op_dict["x_1_logits"] = tf.placeholder(tf.float32, shape = [None,64,64,1])
		#add another placeholder for observed image at second timestep
		self.op_dict["x_2"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		return self.op_dict


	def add_model_ops(self, add_save_op = True, reuse_variables = False):
		#concatenate the two input channels to form x which is passed through the conv layers
		x = tf.concat(3,[self.op_dict["x_1"],self.op_dict["x_2"]])
		#pass the 2 channel observed images through a sequence of conv layers in order to encode the image and infer the joint angle
		h_conv1,W_conv1,b_conv1 = self.gc.conv(x,[3,3,2,self.observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_input", reuse_variables = reuse_variables)
		#find the activations of the second conv layer
		h_conv2,W_conv2,b_conv2 = self.gc.conv(h_conv1,[3,3,self.observed_image_encoder_parameters["conv1_kernels"],self.observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_input", reuse_variables = reuse_variables)
		#find the activations of the third conv layer
		h_conv3,W_conv3,b_conv3 = self.gc.conv(h_conv2,[3,3,self.observed_image_encoder_parameters["conv2_kernels"],self.observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_input", reuse_variables = reuse_variables)
		#find the activations of the fourth conv layer
		h_conv4,W_conv4,b_conv4 = self.gc.conv(h_conv3,[3,3,self.observed_image_encoder_parameters["conv3_kernels"],self.observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_input", reuse_variables = reuse_variables)
		#find the activations of the fifth conv layer
		h_conv5,W_conv5,b_conv5 = self.gc.conv(h_conv4,[3,3,self.observed_image_encoder_parameters["conv4_kernels"],self.observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_input", reuse_variables = reuse_variables)
		#flatten the activations in the final conv layer in order to obtain an output image
		h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*self.observed_image_encoder_parameters["conv5_kernels"]])
		#pass flattened activations to a fully connected layer
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(h_conv5_reshape,[4*self.observed_image_encoder_parameters["conv5_kernels"],self.dof],"fc_layer_encode_input_image", reuse_variables = reuse_variables)
		#now get the graph for the physics emulator
		#save the hidden layer as  joint angle state
		self.op_dict["joint_angle_state"] = h_fc1
		pe = physics_emulator_3dof()
		#add h_fc1 as the input tensor for the physics emulator
		pe.op_dict["x"] = self.op_dict["joint_angle_state"]
		#now add the model ops to the pe graph
		pe_op_dict = pe.add_model_ops()
		#designate the output from the physics emulator to be the output of the onetstep model
		self.op_dict["y_logits"] = pe_op_dict["y"]
		self.op_dict["delta_logits"] = pe_op_dict["y_logits"]
		self.op_dict["y"] = self.op_dict["delta"] + self.op_dict["x_1"]
		variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
		activation_list = [h_conv1,h_conv2,h_conv3,h_conv4,h_conv5,h_fc1]

		for var in variable_list:
			self.var_dict[var.name] = var

		for act in activation_list:
			self.activation_dict[act.name] = act

		if add_save_op:
			self.op_dict["saver"] = tf.train.Saver(pe.var_dict)

		return self.op_dict

	def add_auxillary_ops(self):
		opt = tf.train.AdamOptimizer(self.lr)
		#define the targets by renormalizing the second observed image
		targets = self.op_dict["x_2"]/255.
		#calculate the simoid sum of the logits
		sigmoid_logit_sum = tf.nn.sigmoid(self.op_dict["delta_logits"]) + tf.nn.sigmoid(self.op_dict["x_1_logits"])
		tf.histogram_summary("sigmoid sum",sigmoid_logit_sum)	
		#define the loss op using the y before sigmoid and in the cross entropy sense
		self.op_dict["loss"] = tf.reduce_mean(tf.mul(targets,-tf.log(sigmoid_logit_sum)) + tf.mul(1 - targets, -tf.log(1 - sigmoid_logit_sum)))
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],self.var_dict.values())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item in self.var_dict.items()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		#add the training operation to the graph but only apply gradients to variables from the model 
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		#add the initialization operation
		self.op_dict["init_op"] = tf.initialize_all_variables()
		return self.op_dict

class observed_to_output_seq2seq(tensorflow_graph):
	"""
	Model should create a list of objects
	"""

	def __init__(self,learning_rate,seq_max_length, gc = graph_construction_helper()):
		self.lr = learning_rate
		self.var_dict = {}
		self.op_dict = {}
		self.activation_dict = {}
		self.gc = gc
		self.seq_max = seq_max_length

	def add_placeholder_ops(self):
		#add placeholder for observed image sequence at tstep 1
		self.op_dict['x_1_sequence'] = tf.placeholder(tf.float32,shape = [None,64,64,self.seq_max],name = "x_t1_sequence_tensor")
		#now define a placeholder for the second image
		self.op_dict['x_2_sequence'] = tf.placeholder(tf.float32,shape = [None,64,64,self.seq_max], name = "x_t2_sequence_tensor")
		#define an input tensor for the logit sequence
		self.op_dict["x_1_logit_sequence"] = tf.placeholder(tf.float32, shape = [None,64,64,self.seq_max], name = "x_1_logit_tensor")
		#define an input tensor to store the binary loss, this should be the same shape as the input sequence
		self.op_dict["binary_loss_tensor"] = tf.placeholder(tf.float32, shape = [None,self.seq_max], name = "binary_loss")
		return self.op_dict

	def add_model_ops(self):
		"""
		Create a list of onetstep objects and specify x_1 to be the output from the previous tstep
		"""
		#initialize a list of onetstep graph objects
		onetstep_graph_objects = [onetstep_observed_to_output(gc = self.gc)] * self.seq_max
		#split the input tensors into a list so that we may loop through them and append to them
		x_1_list = tf.split(3,self.seq_max,self.op_dict['x_1_sequence'],name = "x_t1_list")
		x_2_list = tf.split(3,self.seq_max,self.op_dict['x_2_sequence'],name = "x_t2_list")
		x_1_logit_list = tf.split(3,self.seq_max,self.op_dict["x_1_logit_sequence"], name = "x_1_logit_list")
		#now loop through the graph objects and specify the inputs at each timestep
		for tstep in range(self.seq_max):
			onetstep_graph_objects[tstep].op_dict["x_1"] = x_1_list[tstep]
			onetstep_graph_objects[tstep].op_dict["x_2"] = x_2_list[tstep]
			onetstep_graph_objects[tstep].op_dict["x_1_logits"] = x_1_logit_list[tstep]

		#initialize a list of the op dict for each tstep
		onetstep_opdict_list = [onetstep_graph_objects[0].add_model_ops()]
		#now append objects to the onetstep_end2end_list with the op dict
		#initialize a list to record the output delta tensors at each tstep
		delta_output_list = [onetstep_opdict_list[0]["delta"]]
		#initialize a list to store the joint angle state
		joint_angle_state_list = [onetstep_opdict_list[0]["joint_angle_state"]]
		for tstep in range(1,self.seq_max):
			onetstep_opdict_list.append(onetstep_graph_objects[tstep].add_model_ops(add_save_op = False, reuse_variables = True))
			delta_output_list.append(onetstep_opdict_list[tstep]["delta"])
			joint_angle_state_list.append(onetstep_opdict_list[tstep]["joint_angle_state"])

		#before computing loss is now necessary to sum up the deltas at each timestep to obtain the output image at each timestep
		output_image_list = []
		for tstep in range(1,self.seq_max+1):
			output_image_list.append(tf.reduce_sum(delta_output_list[:tstep],0))

		#now define a loss between this output and the observed x_2_sequence define it in the meansquare sense
		#first pack the list into a tensor
		self.op_dict["y"] = tf.concat(3,output_image_list)
		self.op_dict["joint_angle_sequence"] = tf.pack(joint_angle_state_list,-1)
		self.var_dict = onetstep_graph_objects[0].var_dict
		self.op_dict["saver"] = onetstep_opdict_list[0]["saver"]
		return self.op_dict


	def add_auxillary_ops(self):
		opt = tf.train.AdamOptimizer(self.lr)
		#define the loss op using the y before sigmoid and in the cross entropy sense
		#first define the targets
		targets = self.op_dict["x_2_sequence"]/255.
		self.op_dict["loss_per_tstep"] = tf.reduce_mean(tf.mul(targets,-tf.log(self.op_dict["y"])) + tf.mul(1 - targets, -tf.log(1 - self.op_dict["y"])) ,[1,2])
		self.op_dict["loss"] = tf.reduce_mean(tf.mul(self.op_dict["loss_per_tstep"],self.op_dict["binary_loss_tensor"]))
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],self.var_dict.values())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item in self.var_dict.items()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		#add the training operation to the graph but only apply gradients to variables from the model 
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		#add the initialization operation
		self.op_dict["init_op"] = tf.initialize_all_variables()
		return self.op_dict

class onetstep_observed_joint_plus_image_to_output(tensorflow_graph):

	def __init__(self,learning_rate = 1e-3, gc = graph_construction_helper()):
		self.lr = learning_rate
		self.gc = gc
		self.op_dict = {}
		self.var_dict = {}
		self.activation_dict = {}
		self.dof = 3
		self.observed_image_encoder_parameters = {"conv1_kernels": 64, "conv2_kernels": 32, "conv3_kernels": 16, "conv4_kernels": 8, "conv5_kernels": 4, "fc_1" : 20}


	def add_placeholder_ops(self):
		#add placeholder for observed image at first timestep
		self.op_dict["x_1"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		#add another placeholder for observed image at second timestep
		self.op_dict["x_2"] = tf.placeholder(tf.float32,shape = [None,64,64,1])
		return self.op_dict


	def add_model_ops(self, add_save_op = True, reuse_variables = False):
		#concatenate the two input channels to form x which is passed through the conv layers
		x = tf.concat(3,[self.op_dict["x_1"],self.op_dict["x_2"]])
		#pass the 2 channel observed images through a sequence of conv layers in order to encode the image and infer the joint angle
		h_conv1,W_conv1,b_conv1 = self.gc.conv(x,[3,3,2,self.observed_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_input", reuse_variables = reuse_variables)
		#find the activations of the second conv layer
		h_conv2,W_conv2,b_conv2 = self.gc.conv(h_conv1,[3,3,self.observed_image_encoder_parameters["conv1_kernels"],self.observed_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_input", reuse_variables = reuse_variables)
		#find the activations of the third conv layer
		h_conv3,W_conv3,b_conv3 = self.gc.conv(h_conv2,[3,3,self.observed_image_encoder_parameters["conv2_kernels"],self.observed_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_input", reuse_variables = reuse_variables)
		#find the activations of the fourth conv layer
		h_conv4,W_conv4,b_conv4 = self.gc.conv(h_conv3,[3,3,self.observed_image_encoder_parameters["conv3_kernels"],self.observed_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_input", reuse_variables = reuse_variables)
		#find the activations of the fifth conv layer
		h_conv5,W_conv5,b_conv5 = self.gc.conv(h_conv4,[3,3,self.observed_image_encoder_parameters["conv4_kernels"],self.observed_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_input", reuse_variables = reuse_variables)
		#flatten the activations in the final conv layer in order to obtain an output image
		h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*self.observed_image_encoder_parameters["conv5_kernels"]])
		#pass flattened activations to a fully connected layer
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(h_conv5_reshape,[4*self.observed_image_encoder_parameters["conv5_kernels"],self.dof],"fc_layer_encode_input_image", reuse_variables = reuse_variables)
		#now get the graph for the physics emulator
		#save the hidden layer as  joint angle state
		self.op_dict["joint_angle_state"] = h_fc1
		pe = physics_emulator()
		#add h_fc1 as the input tensor for the physics emulator
		pe.op_dict["joint_angle_state"] = self.op_dict["joint_angle_state"]
		pe.op_dict["x_image"] = self.op_dict["x_1"]
		#now add the model ops to the pe graph
		pe_op_dict = pe.add_model_ops(reuse_variables = reuse_variables)
		#designate the output from the physics emulator to be the output of the onetstep model
		self.op_dict["y_logits"] = pe_op_dict["y_logits"]
		self.op_dict["y"] = pe_op_dict["y"]
		variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]
		activation_list = [h_conv1,h_conv2,h_conv3,h_conv4,h_conv5,h_fc1]

		for var in variable_list:
			self.var_dict[var.name] = var

		for act in activation_list:
			self.activation_dict[act.name] = act

		if add_save_op:
			self.op_dict["saver"] = tf.train.Saver(pe.var_dict)

		return self.op_dict

	def add_auxillary_ops(self):
		opt = tf.train.AdamOptimizer(self.lr)
		#define the targets by renormalizing the second observed image
		targets = self.op_dict["x_2"]/255.	
		#define the loss op using the y before sigmoid and in the cross entropy sense
		self.op_dict["loss"] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.op_dict["y_logits"],targets))
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],self.var_dict.values())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item in self.var_dict.items()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		#add the training operation to the graph but only apply gradients to variables from the model 
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		#add the initialization operation
		self.op_dict["init_op"] = tf.initialize_all_variables()
		return self.op_dict
