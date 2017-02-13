import abc

import tensorflow as tf

from core.tf_util import he_uniform_initializer, mlp, linear
from misc.rllab_util import get_action_dim
from predictors.state_network import StateNetwork
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy


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


class FeedForwardPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super(FeedForwardPolicy, self).__init__(name_or_scope=name_or_scope,
                                                **kwargs)

    def _create_network(self, observation_input):
        observation_output = mlp(
            observation_input,
            self.observation_dim,
            self.observation_hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
        )
        return self.output_nonlinearity(linear(
            observation_output,
            self.observation_hidden_sizes[-1],
            self.output_dim,
            W_initializer=self.output_W_init,
            b_initializer=self.output_b_init,
        ))



class Conv_FeedForwardPolicy(NNPolicy):
    
    def __init__(self,name_or_scope):
        self.W_conv1 = tf.get_variable("W_conv1",[3,3,1,64],tf.float32,tf.random_normal_initializer(0.0,0.1))
        self.b_conv1 = tf.get_variable("b_conv1",[64],tf.float32,tf.constant_initializer(0.1))
        self.W_conv3 = tf.get_variable("W_conv2",[3,3,64,32],tf.float32,tf.random_normal_initializer(0.0,0.1))
        self.b_conv2 = tf.get_variable("b_conv2",[32],tf.float32,tf.constant_initializer(0.1))
        self.W_conv3 = tf.get_variable("W_conv3",[3,3,32,16],tf.float32,tf.random_normal_initializer(0.0,0.1))
        self.b_conv3 = tf.get_variable("b_conv3",[16],tf.float32,tf.constant_initializer(0.1))
        self.W_conv4 = tf.get_variable("W_conv4",[3,3,16,8],tf.float32,tf.random_normal_initializer(0.0,0.1))
        self.b_conv4 = tf.get_variable("b_conv4",[8],tf.float32,tf.constant_initializer(0.1))
        self.W_conv5 = tf.get_variable("W_conv5",[3,3,8,4],tf.float32,tf.random_normal_initializer(0.0,0.1))
        self.b_conv5 = tf.get_variable("b_conv5",[4],tf.float32,tf.constant_initializer(0.1))
        #now initialize the variables for the fc layers
        self.W_fc = tf.get_variable("W_fc",[16,2],tf.random_normal_initializer(0,0.1))
        self.b_fc = tf.get_variable("b_fc",[2],tf.constant_initializer(0.0))
        super(Conv_FeedForwardPolicy, self).__init__(name_or_scope=name_or_scope,
                                                **kwargs)


    def _create_network(self,observation_input):
        """
        observation input is a tensor of shape [None,4096]
        you should output a tensor of shape [None,2]
        """
        #first get all the variables
        self.W_conv1 = tf.get_variable("W_conv1")
        self.b_conv1 = tf.get_variable("b_conv1")
        self.W_conv3 = tf.get_variable("W_conv2")
        self.b_conv2 = tf.get_variable("b_conv2")
        self.W_conv3 = tf.get_variable("W_conv3")
        self.b_conv3 = tf.get_variable("b_conv3")
        self.W_conv4 = tf.get_variable("W_conv4")
        self.b_conv4 = tf.get_variable("b_conv4")
        self.W_conv5 = tf.get_variable("W_conv5")
        self.b_conv5 = tf.get_variable("b_conv5")
        #now initialize the variables for the fc layers
        self.W_fc = tf.get_variable("W_fc")
        self.b_fc = tf.get_variable("b_fc")
        #now reshape the input tensor so that it is a 4 channel image of shape [Batch,64,64,1] in order to perform convolutions
        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)
        conv1 = tf.nn.conv2d(x,self.W_conv1,strides = [1,2,2,1],padding = "SAME")
        h_1 = tf.nn.relu(tf.nn.bias_add(conv1,self.b_conv1))
        
        conv2 = tf.nn.conv2d(h_1,self.W_conv2,strides = [1,2,2,1],padding = "SAME")
        h_2 = tf.nn.relu(tf.nn.bias_add(conv2,self.b_conv2))
        
        conv3 = tf.nn.conv2d(h_2,self.W_conv3,strides = [1,2,2,1],padding = "SAME")
        h_3 = tf.nn.relu(tf.nn.bias_add(conv3,self.b_conv3))
        
        conv4 = tf.nn.conv2d(h_3,self.W_conv4,strides = [1,2,2,1],padding = "SAME")
        h_4 = tf.nn.relu(tf.nn.bias_add(conv4,self.b_conv4))
        
        conv5 = tf.nn.conv2d(h_4,self.W_conv5,strides = [1,2,2,1],padding = "SAME")
        h_5 = tf.nn.relu(tf.nn.bias_add(conv5,self.b_conv5))

        #finally pass through fc layer with tanh non linearity
        action = tf.nn.tanh(tf.matmul(h_5,self.W_fc) + self.b_fc)
        return action 

