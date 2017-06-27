import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
from my_tf_util import tensorflow_graph,graph_construction_helper
import tensorflow as tf


class Inference_with_Sequence(tensorflow_graph):

    def __init__(self,
                 learning_rate,
                 hidden_units = [300,50],
                 link_length = 30,
                 seq_length = 5,
                 gc = graph_construction_helper()):

        self.gc = gc
        self.lr = learning_rate
        self.seq_max_length = seq_length
        #initialize the dicts to hold tensors for the script and also for
        #the purposes of debugging
        self.op_dict = {}
        self.activation_dict = {}
        self.var_dict = {}
        #specify the number of hidden units in a lstm cell
        self.hidden_units = hidden_units
        #specify link length for the learner arm
        self.link_length = link_length

    def add_placeholder_ops(self):
        #input to graph is sequence length of 5 with arm renderings of 2DOF arm
        self.op_dict['x'] = tf.placeholder(tf.float32,shape = \
                                           [None,64,64,self.seq_max_length,2])
        #the desired 3DOF end effector position is second position
        #range of actions i -1 to 1 with -1 to 1 represention -0.05pi to 0.05pi
        self.op_dict['y_'] = tf.placeholder(tf.float32,shape = \
                                            [None,self.seq_max_length,3])
        return self.op_dict

    def encode_image(self,
                     x_input,
                     reuse_variables = True):
        """
        Encodes an image using sequence of conv layers
        """
        h_conv1,_,_ = self.gc.conv(x_input,\
                               [7,7,2,32],\
                               'Image_Encoder/Conv_1',\
                               reuse_variables = reuse_variables)

        h_conv2,_,_ = self.gc.conv(h_conv1,\
                               [5,5,32,16],
                               'Image_Encoder/Conv_2',\
                               reuse_variables = reuse_variables)

        h_conv3,_,_ = self.gc.conv(h_conv2,\
                               [5,5,16,1],\
                               'Image_Encoder/Conv_3',\
                               reuse_variables = reuse_variables)
        print(h_conv3)
        #flatten h_conv3
        h_conv3_flat = tf.reshape(h_conv3,
                                  shape = [-1,8*8])
        return h_conv3_flat

    def add_model_ops(self):
        #split the input to a list which may be passed through a series conv
        #layers followed by a lstm 
        x_list = tf.unpack(self.op_dict['x'],axis = 3) 
        #import IPython; IPython.embed()
        #initialize image feature list
        image_features_list = [self.encode_image(x_list[0],\
                                        reuse_variables = False)]
        #take each element of the list and pass it through a conv layer
        image_features_list += [self.encode_image(x) \
                                for x in x_list[1:]]
        #initialize a list to create a multilayered lstm cell
        lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(\
                    hidden_units,state_is_tuple =True) for hidden_units in
                    self.hidden_units]
        #pass this list to multirnn_cell
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list,state_is_tuple=True)
        #pass this into rnn and get hidden unit activations
        outputs,cell_states = tf.nn.rnn(lstm_cell,image_features_list,\
                                        dtype=tf.float32)
        #now intialize some variables for the decoder fc layer
        W_fc = tf.get_variable("Action_Decoder/W_fc",\
                               [self.hidden_units[-1],3],\
                               tf.float32,\
                               tf.random_normal_initializer(0.0,0.1))
        b_fc = tf.get_variable("Action_Decoder/b_fc",\
                               [3],\
                               tf.float32,
                               tf.constant_initializer(0.1))

        #take each element of the output list and pass through a fc layer

        #now compute the state and aciton list at each time step
        action_list = [tf.nn.tanh(\
                    tf.matmul(output_timestep,W_fc) + b_fc)\
                       for output_timestep in outputs]
        #at this point y_list of shape [None,3]*self.seq_max_length
        #now pack the list together in order to get a tensor
        self.op_dict["y"] = tf.pack(action_list,axis=1) 
        #now get the variables and add to a dict
        for var in tf.all_variables():
            self.var_dict[var.name] = var
        return self.op_dict
        
    def add_auxillary_ops(self):
        opt = tf.train.AdamOptimizer(self.lr)
        #define the loss op using the y before sigmoid and in the cross entropy sense
        self.op_dict["loss"] = tf.reduce_mean(tf.square(self.op_dict["y"] - self.op_dict["y_"]))
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
