import abc

import tensorflow as tf

from misc.rllab_util import get_action_dim
from predictors.state_network import StateNetwork
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from predictors.state_action_network import StateActionNetwork

class NNPolicy(StateNetwork, Policy):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        action_dim = get_action_dim(**kwargs)
        # Copy dict to not affect kwargs, which is used by Serialization
        new_kwargs = dict(**kwargs)
        if "action_dim" in new_kwargs:
            new_kwargs.pop("action_dim")
        super(NNPolicy, self).__init__(name_or_scope=name_or_scope,
                                       output_dim=action_dim,
                                       **new_kwargs)
                                       # **kwargs)

    def get_action(self, observation):
        return self.sess.run(self.output,
                             {self.observation_input: [observation]}), {}

    @abc.abstractmethod
    def _create_network(self, observation_input):
        return


class Conv_FeedForwardPolicy(NNPolicy):

    def __init__(self,name_or_scope,**kwargs):
        self.name_or_scope = name_or_scope
        self.setup_serialization(locals())
        with tf.variable_scope(name_or_scope) as scope:
            try:

				self.W_conv1 = tf.get_variable("W_conv1",[5,5,1,self.conv_kernels["kernels_1"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
				self.b_conv1 = tf.get_variable("b_conv1",[self.conv_kernels["kernels_1"]],tf.float32,tf.constant_initializer(0.1))

				self.W_conv2 = tf.get_variable("W_conv2",[5,5,self.conv_kernels["kernels_1"],self.conv_kernels["kernels_2"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
				self.b_conv2 = tf.get_variable("b_conv2",[self.conv_kernels["kernels_2"]],tf.float32,tf.constant_initializer(0.1))

				self.W_conv3 = tf.get_variable("W_conv3",[3,3,self.conv_kernels["kernels_2"],self.conv_kernels["kernels_3"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
				self.b_conv3 = tf.get_variable("b_conv3",[self.conv_kernels["kernels_3"]],tf.float32,tf.constant_initializer(0.1))

				  #now initialize the variables for the fc layers
				self.W_fc = tf.get_variable("W_fc",[9*32,2],tf.float32,tf.random_normal_initializer(0,0.1))
				self.b_fc = tf.get_variable("b_fc",[2],tf.float32,tf.constant_initializer(0.0))
            except:
				scope.reuse_variables()

				self.W_conv1 = tf.get_variable("W_conv1")
				self.b_conv1 = tf.get_variable("b_conv1")

				self.W_conv2 = tf.get_variable("W_conv2")
				self.b_conv2 = tf.get_variable("b_conv2")

				self.W_conv3 = tf.get_variable("W_conv3")
				self.b_conv3 = tf.get_variable("b_conv3")
				#now initialize the variables for the fc layers
				self.W_fc = tf.get_variable("W_fc")
				self.b_fc = tf.get_variable("b_fc")
        super(Conv_FeedForwardPolicy, self).__init__(name_or_scope=name_or_scope,
                                                    **kwargs)

    def _create_network(self,observation_input):
        """
        observation input is a tensor of shape [None,4096]
        you should output a tensor of shape [None,2]
        	"""

        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)
        conv1 = tf.nn.conv2d(x,self.W_conv1,strides = [1,3,3,1],padding = "SAME")
        h_1 = tf.nn.relu(tf.nn.bias_add(conv1,self.b_conv1))

        conv2 = tf.nn.conv2d(h_1,self.W_conv2,strides = [1,3,3,1],padding = "SAME")
        h_2 = tf.nn.relu(tf.nn.bias_add(conv2,self.b_conv2))

        conv3 = tf.nn.conv2d(h_2,self.W_conv3,strides = [1,3,3,1],padding = "SAME")
        h_3 = tf.nn.relu(tf.nn.bias_add(conv3,self.b_conv3))
  
        h_3_flattened = tf.reshape(h_3,shape = [-1,9*32])
        #finally pass through fc layer with tanh non linearity
        action = tf.nn.tanh(tf.matmul(h_3_flattened,self.W_fc) + self.b_fc)
        return action

    def get_params_internal(self):
         if "target" in self.name_or_scope:
             return [v for v in tf.all_variables() if self.name_or_scope[:-1] in v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
         else:
             return [v for v in tf.all_variables() if self.name_or_scope == v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]

class Conv_FeedForwardCritic(NNQFunction):
    """
    Pass observation through conv layers to obtain observation output of shape
    [Batch,x] the action output is of shape [None,2], so concatenate
    the tensors along dimension 1 to obtain the embedded vector
    which may be passed to a few fuly connected layers in order to output a Q value of dimension None,1
    """
    def __init__(self,name_or_scope, 
                 **kwargs):
        self.name = str(name_or_scope)
        self.conv_kernels = {"kernels_1" : 32,"kernels_2" : 32, "kernels_3" : 32}
        self.fc_units = 200
        self.setup_serialization(locals())
        
        with tf.variable_scope(name_or_scope) as scope:
            try:
                self.W_conv1 = tf.get_variable("W_conv1",[5,5,1,self.conv_kernels["kernels_1"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv1 = tf.get_variable("b_conv1",[self.conv_kernels["kernels_1"]],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv2 = tf.get_variable("W_conv2",[5,5,self.conv_kernels["kernels_1"],self.conv_kernels["kernels_2"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv2 = tf.get_variable("b_conv2",[self.conv_kernels["kernels_2"]],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv3 = tf.get_variable("W_conv3",[3,3,self.conv_kernels["kernels_2"],self.conv_kernels["kernels_3"]],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv3 = tf.get_variable("b_conv3",[self.conv_kernels["kernels_3"]],tf.float32,tf.constant_initializer(0.1))
                
                self.W_fc_obs = tf.get_variable("W_fc_obs",[9*self.conv_kernels["kernels_3"],10],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_obs = tf.get_variable("b_fc_obs",[10],tf.float32,tf.constant_initializer(0.0))
            
                self.W_fc_embed_1 = tf.get_variable("W_fc_embed_1",[12,self.fc_units],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_embed_1 = tf.get_variable("b_fc_embed_1",[self.fc_units],tf.float32,tf.constant_initializer(0.0))
                
                self.W_fc_embed_2 = tf.get_variable("W_fc_embed_2",[self.fc_units,1],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_embed_2 = tf.get_variable("b_fc_embed_2",[1],tf.float32,tf.constant_initializer(0.0))

            except:
                scope.reuse_variables()                
                self.W_conv1 = tf.get_variable("W_conv1")
                self.b_conv1 = tf.get_variable("b_conv1")
                
                self.W_conv2 = tf.get_variable("W_conv2")
                self.b_conv2 = tf.get_variable("b_conv2")
                
                self.W_conv3 = tf.get_variable("W_conv3")
                self.b_conv3 = tf.get_variable("b_conv3")
                
                #now initialize the variables for the fc layers
                self.W_fc_obs = tf.get_variable("W_fc_obs")
                self.b_fc_obs = tf.get_variable("b_fc_obs")
        
                self.W_fc_embed_1 = tf.get_variable("W_fc_embed_1")
                self.b_fc_embed_1 = tf.get_variable("b_fc_embed_1")
                
                self.W_fc_embed_2 = tf.get_variable("W_fc_embed_2")
                self.b_fc_embed_2 = tf.get_variable("b_fc_embed_2")

        super(Conv_FeedForwardCritic,self).__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self,observation_input,action_input):
        #IPython.embed()
        #the observation input is provided as a flattened tensor so reshape it
        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)
        conv1 = tf.nn.conv2d(x,self.W_conv1,strides = [1,3,3,1],padding = "SAME")
        h_1 = tf.nn.relu(tf.nn.bias_add(conv1,self.b_conv1))

        conv2 = tf.nn.conv2d(h_1,self.W_conv2,strides = [1,3,3,1],padding = "SAME")
        h_2 = tf.nn.relu(tf.nn.bias_add(conv2,self.b_conv2))

        conv3 = tf.nn.conv2d(h_2,self.W_conv3,strides = [1,3,3,1],padding = "SAME")
        h_3 = tf.nn.relu(tf.nn.bias_add(conv3,self.b_conv3))

        h_3_flattened = tf.reshape(h_3,shape = [-1,9*32])
        #finally pass through fc layer with tanh non linearity
        observation_output = tf.nn.tanh(tf.matmul(h_3_flattened,self.W_fc_obs) + self.b_fc_obs)
        #now concatenate this with the action input along the 1st dimension
        embed = tf.concat(1,[observation_output,action_input])
        #pass embed along the two defined fc layers
        h_embed_1 = tf.nn.relu(tf.matmul(embed,self.W_fc_embed_1) + self.b_fc_embed_1)
        q_value = tf.matmul(h_embed_1,self.W_fc_embed_2) + self.b_fc_embed_2
        return q_value

    def get_params_internal(self, **tags):
        #IPython.embed()
        if "target" in self.name:
             return [v for v in tf.all_variables() if self.name[:-1] in v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
        else:
             return [v for v in tf.all_variables() if self.name == v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
    




