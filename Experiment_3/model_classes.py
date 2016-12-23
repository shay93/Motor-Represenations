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
			if reuse_variables:
				scope.reuse_variables()
			#initialize the weights for the convolutional layer
			W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_conv")
			#initiaize the biases
			b = tf.Variable(tf.constant(0.1,shape = [weight_shape[-1]]), trainable = trainable, name = "b_conv")
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
			if reuse_variables:
				scope.reuse_variables()
		
			#initialize the weights for the convolutional layer
			W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_fc")
			#initiaize the biases
			b = tf.Variable(tf.constant(0.,shape = [weight_shape[-1]]), trainable = trainable, name = "b_fc")
			#calculate biases
			h = tf.nn.relu(tf.matmul(x,W) + b, name = "activations_fc")

		return h,W,b 

	def deconv(self,x,weight_shape,output_shape,scope,strides = [1,2,2,1], stddev = 0.1,trainable = True, reuse_variables = False,non_linearity = True):
		"""
		generalizable deconv function
		"""
		with tf.variable_scope(scope) as scope:
			if reuse_variables:
				scope.reuse_variables()
			#initialize the weights for the convolutional layer
			W = tf.Variable(tf.truncated_normal(weight_shape,stddev = stddev), trainable = trainable, name = "W_deconv")
			#initiaize the biases
			b = tf.Variable(tf.constant(0.1,shape = [weight_shape[-2]]), trainable = trainable, name = "b_deconv")
			#calculate the output from the deconvolution
			deconv = tf.nn.conv2d_transpose(x,W,output_shape,strides = strides)
			#calculate the activations
			if non_linearity:
				h = tf.nn.relu(tf.nn.bias_add(deconv,b), name = "activations_deconv")
			else:
				h = tf.nn.bias_add(deconv,b, name = "activations_deconv")

		return h,W,b


class tensorflow_graph:
	#all graphs should be a subclass of the tensorflow graph object which has methods that can be used to train and evaluate the graph
	#so given the model the trainer or evaluator calls on the train_op in the graph to construct the graph
	
	def train_graph(self,Epochs,batch_size,placeholder_dict,train_op,init_op, loss_op, merge_summary_op = None,log_dir = None,summary_writer_freq = 20):
		"""
		1)Epochs is the number determines the number of iterations and hence the number of times that parameter updates are made
		2)placeholder dict holds the placholder ops of the graph and the data that needs to be passed into the graph in order to compute gradients and train parameters
		3)placeholder dict is keyed by the tensorflow object and the dict holds the data set corresponding to the key
		4)init_op is used to initialize the variables in the graph based on the shape and datatypes that have been specified prior
		5)train_op is the training operation of the graph which is called upon when gradients are computed/applied
		"""
		#first get the number of samples by getting finding the shape of the first data variable
		num_samples = np.shape(placeholder_dict[placeholder_dict.keys()[0]])[0]
		#initialize a configuration protobuf object that can be used to initialize a session
		config = tf.ConfigProto()
		#configure a sessions object such that the gpu usage grows
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)
		#if we have a merge summary op initialize a summary writer
		if merge_summary_op is not(None):
			#initialize a summary writer and pass the graph to it
			summary_writer = tf.train.SummaryWriter(log_dir,sess.graph)

		#initialize the variables in the graph
		sess.run(init_op)
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
				_ = sess.run(train_op,feed_dict = feed_dict)
		return sess


	def evaluate_graph(self,eval_batch_size,placeholder_dict,y_op,loss_op,y_label_op,sess):
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
	
	def __init__(self):
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
		self.learning_rate = 1e-3
		#initialize an empty operation dictionary to store the variables
		self.op_dict = {}

	def encode_previous_output_image(self,previous_output_image):
		"""
		Forms a representation of the output image provided at the previous step
		"""

		#expand the dimensionality of the input image
		x_image = tf.expand_dims(previous_output_image, -1)
		#find the activations of the first conv layer
		h_conv1,W_conv1,b_conv1 = self.gc.conv(x_image,[3,3,1,self.output_image_encoder_parameters["conv1_kernels"]],"Conv1_encode_output",trainable = True)
		#find the activations of the second conv layer
		h_conv2,W_conv2,b_conv2 = self.gc.conv(h_conv1,[3,3,self.output_image_encoder_parameters["conv1_kernels"],self.output_image_encoder_parameters["conv2_kernels"]],"Conv2_encode_output",trainable = True)
		#find the activations of the third conv layer
		h_conv3,W_conv3,b_conv3 = self.gc.conv(h_conv2,[3,3,self.output_image_encoder_parameters["conv2_kernels"],self.output_image_encoder_parameters["conv3_kernels"]],"Conv3_encode_output",trainable = True)
		#find the activations of the second conv layer
		h_conv4,W_conv4,b_conv4 = self.gc.conv(h_conv3,[3,3,self.output_image_encoder_parameters["conv3_kernels"],self.output_image_encoder_parameters["conv4_kernels"]],"Conv4_encode_output",trainable = True)
		#find the activations of the second conv layer
		h_conv5,W_conv5,b_conv5 = self.gc.conv(h_conv4,[3,3,self.output_image_encoder_parameters["conv4_kernels"],self.output_image_encoder_parameters["conv5_kernels"]],"Conv5_encode_output",trainable = True)
		#flatten the activations in the final conv layer in order to obtain an output image
		h_conv5_reshape = tf.reshape(h_conv5, shape = [-1,4*self.output_image_encoder_parameters["conv5_kernels"]])
		#pass flattened activations to a fully connected layer
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(h_conv5_reshape,[4*self.output_image_encoder_parameters["conv5_kernels"],self.output_image_encoder_parameters["fc_1"]],"fc_layer_encode_output",trainable = True)
		output_image_encoder_variable_list = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,W_fc1,b_fc1]

		return h_fc1,output_image_encoder_variable_list 

	def encode_joints(self,x_joints):
		"""
		Takes joint states and encodes them in order to generate a new point in the output image
		"""
		h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(x_joints,[self.dof,self.joint_encoder_parameters["fc_1"]],"fc_joint_encoder_1",trainable = True)
		#pass the activations to a second fc layer
		h_fc2,W_fc2,b_fc2 = self.gc.fc_layer(h_fc1,[self.joint_encoder_parameters["fc_1"], self.joint_encoder_parameters["fc_2"]],"fc_joint_encoder_2",trainable = True)
		joint_encoder_variable_list = [W_fc1,b_fc1,W_fc2,b_fc2]

		return h_fc2,joint_encoder_variable_list


	def decode_outputs(self,hidden_vector):
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
		h_deconv1,W_deconv1,b_deconv1 = self.gc.deconv(hidden_image,[2,2,self.output_image_decoder_parameters['deconv_output_channels_1'],64],[batch_size,4,4,self.output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1])
		#calculate activations for second deconv layer
		h_deconv2,W_deconv2,b_deconv2 = self.gc.deconv(h_deconv1,[3,3,self.output_image_decoder_parameters['deconv_output_channels_2'],self.output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,self.output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2")
		#calculate activations for third deconv layer
		h_deconv3,W_deconv3,b_deconv3 = self.gc.deconv(h_deconv2,[3,3,self.output_image_decoder_parameters['deconv_output_channels_3'],self.output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,self.output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3")
		#calculate activations for fourth deconv layer
		h_deconv4,W_deconv4,b_deconv4 = self.gc.deconv(h_deconv3,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4")
		#calculate activations for fifth deconv layer
		h_deconv5,W_deconv5,b_deconv5 = self.gc.deconv(h_deconv4,[3,3,self.output_image_decoder_parameters['deconv_output_channels_5'],self.output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,self.output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5",non_linearity = False)
		decoder_variable_list = [W_deconv1,W_deconv2,W_deconv3,W_deconv4,W_deconv5,b_deconv1,b_deconv2,b_deconv3,b_deconv4,b_deconv5]

		return tf.squeeze(h_deconv5),decoder_variable_list


	def jointangle2image(self,joint_angle,previous_image):
		"""
		Calls on the respective decoder and encoders in order to map a joint angle state to an output image joint_angle and previous image are both tensors
		"""
		encoded_joint_angle,joint_encoder_variable_list = self.encode_joints(joint_angle)
		previous_image_encoded,image_encode_variable_list = self.encode_previous_output_image(previous_image)
		#now concatenate to obtain encoded vector
		encoded_vector = tf.concat(1,[encoded_joint_angle,previous_image_encoded])
		#pass to a decoder in order to get the output
		y_before_sigmoid,decoder_variable_list = self.decode_outputs(encoded_vector)
		return y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list

	def build_graph(self):
		
		self.op_dict["x_image"] = tf.placeholder(tf.float32,shape = [None,64,64])
		self.op_dict["x_joint"] = tf.placeholder(tf.float32,shape = [None,self.dof])
		y_ = tf.placeholder(tf.float32,shape = [None,64,64])

		#pass the input image and joint angle tensor to jointangle2image to get y_before_sigmoid
		y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list = self.jointangle2image(self.op_dict["x_joint"],self.op_dict["x_image"])
		#apply sigmoid to get y
		y = tf.nn.sigmoid(y_before_sigmoid)
		#copy the output tensor to the operation dict
		self.op_dict["y"] = y
		#define the loss op using the y before sigmoid and in the cross entropy sense
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_before_sigmoid,y_))
		#add a summary node to record the loss
		tf.scalar_summary("loss summary",loss)
		#define the optimizer with the specified learning rate
		opt = tf.train.AdamOptimizer(self.learning_rate)
		grads_and_vars = opt.compute_gradients(loss, joint_encoder_variable_list + image_encode_variable_list + decoder_variable_list)
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for i,gv in enumerate(grads_and_vars)]
		var_summary_nodes = [tf.histogram_summary(gv[1].name,gv[1]) for i,gv in enumerate(grads_and_vars)]
		merged = tf.merge_all_summaries()
		train_op = opt.apply_gradients(grads_and_vars)
		#define an op to initialize variables
		self.op_dict['init_op'] = tf.initialize_all_variables()
		# Add ops to save and restore all the variables.
		self.op_dict["saver"] = tf.train.Saver(joint_encoder_variable_list+image_encode_variable_list+decoder_variable_list)
		return self.op_dict


class one_layer_fc(tensorflow_graph):

	def __init__(self,fc_units,learning_rate):
		self.fc_units = fc_units
		self.lr = learning_rate
		#initialize a placeholder dict that can be used for training the graph
		self.op_dict = {}
		self.gc = graph_construction_helper()
	

	def build_graph(self):
		#initialize a placedholder for the output and the input
		self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None,2])
		#now a placeholder for the labels
		self.op_dict['y_'] = tf.placeholder(tf.float32, shape = [None,1])
		#now add the fc layer
		h1,W1,b1 = self.gc.fc_layer(self.op_dict['x'],[2,self.fc_units],"fc_layer")
		self.op_dict["y"],W2,b2 = self.gc.fc_layer(h1,[self.fc_units,1],"readout")
		#define a means square loss between the predicted and label
		self.op_dict["loss"] = tf.reduce_mean(tf.square(self.op_dict["y"] - self.op_dict["y_"]))
		#intialize the optimizer
		opt = tf.train.AdamOptimizer(self.lr)
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],tf.all_variables())
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var.name,var) for var in tf.all_variables()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		self.op_dict["init_op"] = tf.initialize_all_variables()
		return self.op_dict



class physics_emulator_3dof(tensorflow_graph):

	def __init__(self,learning_rate):
		self.output_image_decoder_parameters = {"deconv_output_channels_1" : 16, "deconv_output_channels_2" : 8, "deconv_output_channels_3" : 4, "deconv_output_channels_4" : 1}
		self.lr = learning_rate
		self.gc = graph_construction_helper()
		self.op_dict = {}
	
	def build_graph(self):
		#initialize placeholder for joint angle inputs
		self.op_dict["x"] = tf.placeholder(tf.float32,shape = [None,3])
		#initialize placeholder for labels
		self.op_dict["y_"] = tf.placeholder(tf.float32, shape = [None,64,64,1])
		#add a fully connected layer to encode the joint angles
		h1_fc,_,_ = self.gc.fc_layer(self.op_dict["x"],[3,32*4*4],"fc_layer_1")
		#now reshape this to get a 4d image
		h1_fc_reshaped = tf.reshape(h1_fc,shape = [-1,4,4,32])
		#find the batch size of the input data in order to use later
		batch_size = tf.shape(h1_fc_reshaped)[0]
		#pass this to a succession of deconv layers in order to get the desired image
		h_deconv1,W_deconv1,b_deconv1 = self.gc.deconv(h1_fc_reshaped,[2,2,self.output_image_decoder_parameters['deconv_output_channels_1'],32],[batch_size,4,4,self.output_image_decoder_parameters['deconv_output_channels_1']],"Deconv1",strides = [1,1,1,1])
		#calculate activations for second deconv layer
		h_deconv2,W_deconv2,b_deconv2 = self.gc.deconv(h_deconv1,[3,3,self.output_image_decoder_parameters['deconv_output_channels_2'],self.output_image_decoder_parameters['deconv_output_channels_1']],[batch_size,8,8,self.output_image_decoder_parameters['deconv_output_channels_2']],"Deconv2")
		#calculate activations for third deconv layer
		h_deconv3,W_deconv3,b_deconv3 = self.gc.deconv(h_deconv2,[3,3,self.output_image_decoder_parameters['deconv_output_channels_3'],self.output_image_decoder_parameters['deconv_output_channels_2']],[batch_size,16,16,self.output_image_decoder_parameters['deconv_output_channels_3']],"Deconv3")
		#calculate activations for fourth deconv layer
		h_deconv4,W_deconv4,b_deconv4 = self.gc.deconv(h_deconv3,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_3']],[batch_size,32,32,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv4")
		#calculate activations for fifth deconv layer
		h_deconv5,W_deconv5,b_deconv5 = self.gc.deconv(h_deconv4,[3,3,self.output_image_decoder_parameters['deconv_output_channels_4'],self.output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,self.output_image_decoder_parameters['deconv_output_channels_4']],"Deconv5")
		#define a loss op between the generated output and the label
		opt = tf.train.AdamOptimizer(self.lr)
		#define the loss op using the y before sigmoid and in the cross entropy sense
		self.op_dict["loss"] = tf.reduce_mean(tf.square(h_deconv5 - self.op_dict["y_"]))
		#add a summary node to record the loss
		#tf.scalar_summary("loss summary",self.op_dict['loss'])
		#get all the variables and compute gradients
		grads_and_vars = opt.compute_gradients(self.op_dict["loss"],tf.all_variables())
		self.op_dict["y"] = h_deconv5
		#add summary nodes for the gradients
		gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) for gv in grads_and_vars]
		var_summary_nodes = [tf.histogram_summary(var.name,var) for var in tf.all_variables()]
		#add a scalar summary for the loss
		tf.scalar_summary("loss summary",self.op_dict["loss"])
		#merge the summaries
		self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
		self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
		self.op_dict["init_op"] = tf.initialize_all_variables()
		return self.op_dict
		
