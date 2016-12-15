from __future__ import division
import numpy as np
import tensorflow as tf
import os


class graph_construction_helper:
	
	def conv(x,weight_shape, scope, stddev = 0.1,trainable = True, reuse_variables = False):
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
			h = tf.nn.relu(tf.nn.bias_add(conv,b))

		return h,W,b


	def fc_layer(x,weight_shape,scope, stddev = 0.1,trainable = True, reuse_variables = False):
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
			h = tf.nn.relu(tf.matmul(x,W) + b)

		return h,W,b 

	def deconv(x,weight_shape,output_shape,scope,strides = [1,2,2,1], stddev = 0.1,trainable = True, reuse_variables = False,non_linearity = True):
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
				h = tf.nn.relu(tf.nn.bias_add(deconv,b))
			else:
				h = tf.nn.bias_add(deconv,b)

		return h,W,b


class physics_emulator:
	
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
		h_deconv5,W_deconv5,b_deconv5 = deconv(h_deconv4,[3,3,self.output_image_decoder_parameters['deconv_output_channels_5'],self.output_image_decoder_parameters['deconv_output_channels_4']],[batch_size,64,64,self.output_image_decoder_parameters['deconv_output_channels_5']],"Deconv5",non_linearity = False)
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
		self.op_dict["x_joint"] = tf.placeholder(tf.float32,shape = [None,DOF])
		y_ = tf.placeholder(tf.float32,shape = [None,64,64])

		#pass the input image and joint angle tensor to jointangle2image to get y_before_sigmoid
		y_before_sigmoid,joint_encoder_variable_list,image_encode_variable_list,decoder_variable_list = jointangle2image(self.op_dict["x_joint"],self.op_dict["x_image"])
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
		init_op = tf.initialize_all_variables()
		# Add ops to save and restore all the variables.
		self.op_dict["saver"] = tf.train.Saver(joint_encoder_variable_list+image_encode_variable_list+decoder_variable_list)
		return op_dict