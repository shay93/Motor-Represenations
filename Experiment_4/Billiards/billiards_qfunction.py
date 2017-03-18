import tensorflow as tf

from mod_tf_util import he_uniform_initializer, mlp, linear,conv
from predictors.state_action_network import StateActionNetwork
from rllab.core.serializable import Serializable
import IPython

class NNQFunction(StateActionNetwork):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, output_dim=1, **kwargs)

class FeedForwardCritic(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            action_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            embedded_hidden_sizes=(100,50,),
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.action_hidden_sizes = action_hidden_sizes
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input, action_input):
        with tf.variable_scope("observation_mlp") as _:
            observation_output = mlp(observation_input,
                                     self.observation_dim,
                                     self.observation_hidden_sizes,
                                     self.hidden_nonlinearity,
                                     W_initializer=self.hidden_W_init,
                                     b_initializer=self.hidden_b_init,
                                     )
        with tf.variable_scope("action_mlp") as _:
            action_output = mlp(action_input,
                                self.action_dim,
                                self.action_hidden_sizes,
                                self.hidden_nonlinearity,
                                W_initializer=self.hidden_W_init,
                                b_initializer=self.hidden_b_init,
                                )
        
        embedded = tf.concat(1, [observation_output, action_output])
        embedded_dim = self.action_hidden_sizes[-1] + self.observation_hidden_sizes[-1]
        
        with tf.variable_scope("fusion_mlp") as _:
            fused_output = mlp(embedded,
                               embedded_dim,
                               self.embedded_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               )

        with tf.variable_scope("output_linear") as _:
            return linear(fused_output,
                          self.embedded_hidden_sizes[-1],
                          1,
                          W_initializer=self.output_W_init,
                          b_initializer=self.output_b_init,
                          )

class Conv_FeedForwardCritic(NNQFunction):
    """
    Pass observation through conv layers to obtain observation output of shape
    [Batch,x] the action output is of shape [None,2], so concatenate
    the tensors along dimension 1 to obtain the embedded vector
    which may be passed to a few fuly connected layers in order to output a Q value of dimension None,1
    """
    def __init__(self,name_or_scope, 
                 **kwargs):
        self.setup_serialization(locals())
        self.name = str(name_or_scope)
        self.hidden_W_init=he_uniform_initializer()
        self.hidden_b_init=tf.constant_initializer(0.)
        self.output_W_init=tf.random_uniform_initializer(
            -3e-4,3e-4)
        self.output_b_init=tf.random_uniform_initializer(
            -3e-4,3e-4)
        self.action_mlp_hidden_sizes = (100,),
        self.observation_mlp_hidden_sizes = (100,),
        self.fusion_mlp_hidden_sizes = (100,50,),
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self,observation_input,action_input):
        #IPython.embed()
        #the observation input is provided as a flattened tensor so reshape it
        x = tf.expand_dims(tf.reshape(observation_input,shape = [-1,64,64]),-1)
        #cast observations to floats and normalize before performing operations on it
        x = tf.to_float(x)/255.
          
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

        
        h_3_flattened = tf.reshape(h_3,shape = [-1,1152])
          
        
        with tf.variable_scope("observation_mlp") as _:
            observation_encoded = mlp(h_3_flattened,
                               1152,
                               self.observation_mlp_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               )
       
        with tf.variable_scope("action_mlp") as _:
            action_encoded = mlp(action_input,
                               self.action_dim,
                               self.action_mlp_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               )


        #once action has been encoded concatenate it with the observation 
        embedded = tf.concat(1,[observation_encoded,action_encoded])
        embedded_dim = self.observation_mlp_hidden_sizes[-1] + self.action_mlp_hidden_sizes[-1]
        
        with tf.variable_scope("fusion_mlp") as _:
            fused_encoded = mlp(embedded,
                               embedded_dim,
                               self.fusion_mlp_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               )

        with tf.variable_scope("output_linear") as _:
            return linear(fused_encoded,
                          self.fusion_mlp_hidden_sizes[-1],
                          1,
                          W_initializer=self.output_W_init,
                          b_initializer=self.output_b_init,
                          )





