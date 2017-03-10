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
            observation_hidden_sizes=(400, 300),
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

    def __init__(self,
                name_or_scope,
                **kwargs):
        self.name_or_scope = name_or_scope
        self.setup_serialization(locals())       
        super(Conv_FeedForwardPolicy, self).__init__(name_or_scope=name_or_scope,
                                                    **kwargs)

    def _create_network(self,observation_input):
        """
        observation input is a tensor of shape [None,4096]
        you should output a tensor of shape [None,2]
        """
        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)

        with tf.variable_scope("ConvNet") as _:
          
          with tf.variable_scope("Conv_1") as _:
            h_1 = conv(
              x,
              [5,5,1,32],
              tf.nn.tanh)
          
          with tf.variable_scope("Conv_2") as _:
            h_2 = conv(
              h_1,
              [5,5,32,32],
              tf.nn.tanh)

          with tf.variable_scope("Conv_3") as _:
            h_3 = conv(
              h_2,
              [3,3,32,32],
              tf.nn.tanh)

          h_3_flattened = tf.reshape(h_3,shape = [-1,9*32])

        with tf.variable_scope("MLP") as _:
          action = mlp(h_3_flattened,
            9*32,
            [2],
            tf.nn.tanh) * 3
  
        return action

    # def get_params_internal(self):
    #      if "target" in self.name_or_scope:
    #          return [v for v in tf.all_variables() if self.name_or_scope[:-1] in v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
    #      else:
    #          return [v for v in tf.all_variables() if self.name_or_scope == v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]

