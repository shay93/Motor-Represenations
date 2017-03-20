import abc

import tensorflow as tf

from mod_tf_util import he_uniform_initializer, mlp, linear,conv
from misc.rllab_util import get_action_dim
from predictors.state_network import StateNetwork
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
import numpy as np
import IPython

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
            observation_hidden_sizes=(200, 100),
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
        ))*0.05



class Conv_FeedForwardPolicy(NNPolicy):

    def __init__(self,
                name_or_scope, 
                **kwargs):
        self.setup_serialization(locals())
        self.hidden_W_init = he_uniform_initializer()
        self.hidden_b_init = tf.constant_initializer(0.)
        self.output_W_init = tf.random_uniform_initializer(
                -3e-4,3e-4)
        self.output_b_init = tf.random_uniform_initializer(
                -3e-4,3e-4)
        self.name_or_scope = name_or_scope
        #self.action_mlp_hidden_sizes = action_mlp_hidden_sizes       
        super(Conv_FeedForwardPolicy, self).__init__(name_or_scope=name_or_scope,
                                                    **kwargs)

    def _create_network(self,observation_input):
        """
        observation input is a tensor of shape [None,4096]
        you should output a tensor of shape [None,2]
        """
        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)
        #cast the input observation as a float and normalize before passing into layers
        x = tf.to_float(x)/255.
        with tf.variable_scope("Observation_ConvNet") as _:
          
          with tf.variable_scope("Conv_1") as _:
            h_1 = conv(
              x,
              [7,7,1,32],
              tf.nn.relu,
              strides=[1,3,3,1],
              W_initializer=self.hidden_W_init,
              b_initializer=self.hidden_b_init
              )
          
          with tf.variable_scope("Conv_2") as _:
            h_2 = conv(
              h_1,
              [5,5,32,32],
              tf.nn.relu,
              strides=[1,2,2,1],
              W_initializer=self.hidden_W_init,
              b_initializer=self.hidden_b_init
              )

          with tf.variable_scope("Conv_3") as _:
            h_3 = conv(
              h_2,
              [5,5,32,32],
              tf.nn.relu,
              strides=[1,2,2,1],
              W_initializer=self.hidden_W_init,
              b_initializer=self.hidden_b_init
              )
          print(h_3)
          h_3_flattened = tf.reshape(h_3,shape = [-1,6*6*32])

        with tf.variable_scope("Action__mlp") as _:
          observation_output = mlp(h_3_flattened,
            6*6*32,
            [200,100],
            tf.nn.relu,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init
            )

        with tf.variable_scope("Action_readout") as _:
          action = mlp(observation_output,
                 100,
                 [2],
                 tf.nn.tanh,
                 W_initializer=self.output_W_init,
                 b_initializer=self.output_b_init)
  
        return action

