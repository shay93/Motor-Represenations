import tensorflow as tf

from core.tf_util import he_uniform_initializer, mlp, linear
from predictors.state_action_network import StateActionNetwork
from rllab.core.serializable import Serializable

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
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
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
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
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


def Conv_FeedForwardCritic(NNQFunction):
    """
    Pass observation through conv layers to obtain observation output of shape
    [Batch,x] the action output is of shape [None,2], so concatenate
    the tensors along dimension 1 to obtain the embedded vector
    which may be passed to a few fuly connected layers in order to output a Q value of dimension None,1
    """
    def __init__(self,name_or_scope,**kwargs):
        
        self.name_or_scope = name_or_scope
        self.setup_serialization(locals())
        
        with tf.variable_scope(name_or_scope) as scope:
            try:
                self.W_conv1 = tf.get_variable("W_conv1",[3,3,1,64],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv1 = tf.get_variable("b_conv1",[64],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv2 = tf.get_variable("W_conv2",[3,3,64,32],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv2 = tf.get_variable("b_conv2",[32],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv3 = tf.get_variable("W_conv3",[3,3,32,16],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv3 = tf.get_variable("b_conv3",[16],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv4 = tf.get_variable("W_conv4",[3,3,16,8],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv4 = tf.get_variable("b_conv4",[8],tf.float32,tf.constant_initializer(0.1))
                
                self.W_conv5 = tf.get_variable("W_conv5",[3,3,8,4],tf.float32,tf.random_normal_initializer(0.0,0.1))
                self.b_conv5 = tf.get_variable("b_conv5",[4],tf.float32,tf.constant_initializer(0.1))
                #now initialize the variables for the fc layers
                self.W_fc_obs = tf.get_variable("W_fc_obs",[16,10],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_obs = tf.get_variable("b_fc_obs",[2],tf.float32,tf.constant_initializer(0.0))
            
                self.W_fc_embed_1 = tf.get_variable("W_fc_embed_1",[12,200],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_embed_1 = tf.get_variable("b_fc_embed_1",[200],tf.float32,tf.constant_initializer(0.0))
                
                self.W_fc_embed_2 = tf.get_variable("W_fc_embed_2",[200,1],tf.float32,tf.random_normal_initializer(0,0.1))
                self.b_fc_embed_2 = tf.get_variable("b_fc_embed_2",[1],tf.float32,tf.constant_initializer(0.0))

            else:
                
                self.W_conv1 = tf.get_variable("W_conv1")
                self.b_conv1 = tf.get_variable("b_conv1")
                
                self.W_conv2 = tf.get_variable("W_conv2")
                self.b_conv2 = tf.get_variable("b_conv2")
                
                self.W_conv3 = tf.get_variable("W_conv3")
                self.b_conv3 = tf.get_variable("b_conv3")
                
                self.W_conv4 = tf.get_variable("W_conv4")
                self.b_conv4 = tf.get_variable("b_conv4")
                
                self.W_conv5 = tf.get_variable("W_conv5")
                self.b_conv5 = tf.get_variable("b_conv5")
                #now initialize the variables for the fc layers
                self.W_fc_obs = tf.get_variable("W_fc_obs")
                self.b_fc_obs = tf.get_variable("b_fc_obs")
        
                self.W_fc_embed_1 = tf.get_variable("W_fc_embed_1")
                self.b_fc_embed_1 = tf.get_variable("b_fc_embed_1")
                
                self.W_fc_embed_2 = tf.get_variable("W_fc_embed_2")
                self.b_fc_embed_2 = tf.get_variable("b_fc_embed_2")


        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self,observation_input,action_input):
        #the observation input is provided as a flattened tensor so reshape it
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
        h_5_flattened = tf.reshape(h_5,shape = [-1,16])
        #finally pass through fc layer with tanh non linearity
        observation_output = tf.nn.tanh(tf.matmul(h_5_flattened,self.W_fc_obs) + self.b_fc_obs)
        #now concatenate this with the action input along the 1st dimension
        embed = tf.concat(1,[observation_output,action_input])
        #pass embed along the two defined fc layers
        h_embed_1 = tf.nn.relu(tf.matmul(embed,self.W_fc_embed_1) + self.b_fc_embed_1)
        q_value = tf.matmul(h_embed_1,self.W_fc_embed_2) + self.b_fc_embed_2
        return q_value

    def get_params_internal(self):
        if "target" in self.name_or_scope:
             return [v for v in tf.global_variables() if self.name_or_scope[:-1] in v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
         else:
             return [v for v in tf.global_variables() if self.name_or_scope == v.name.split("/")[0] and not("Adam" in v.name.split("/")[-1])]
    



