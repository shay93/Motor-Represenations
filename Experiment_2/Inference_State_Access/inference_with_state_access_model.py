import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
from my_tf_util import tensorflow_graph,graph_construction_helper
import tensorflow as tf


class Inference_with_state_access(tensorflow_graph):
    
    def __init__(self,
                 learning_rate,
                 hidden_units = 100,
                 link_length = 30,
                 gc = graph_construction_helper()):
        
        self.gc = gc
        self.lr = learning_rate
        self.seq_max_length = 100
        #initialize the dicts to hold tensors for the script and also for
        #the purposes of debugging
        self.op_dict = {}
        self.activation_dict = {}
        self.var_dict = {}
        #specify the number of layers in the lstm cells
        self.layers = 5
        #specify the number of hidden units in a lstm cell
        self.hidden_units = hidden_units
        #specify link length for the learner arm
        self.link_length = link_length
    
    def add_placeholder_ops(self):
        #input to graph is sequence length of 100 with 2 input features for 2DOF arm (st,at)
        self.op_dict['x'] = tf.placeholder(tf.float32,shape = [None,2,self.seq_max_length])
        #the desired end effector position is the second input
        self.op_dict['y'] = tf.placeholder(tf.float32,shape = [None,2,self.seq_max_length])
        return self.op_dict
    
    def add_model_ops(self):
        #split the input into a list which may be fed into a multi-layer lstm cell
        x_list = tf.unpack(self.op_dict['x'],axis = 2)
        #initialize a list to create a multilayered lstm cell
        lstm_list = [tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units,state_is_tuple = True)]*self.layers
        #pass this list to multirnn_cell
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list,state_is_tuple=True)
        #pass this into rnn and get hidden unit activations
        outputs,states = tf.nn.rnn(lstm_cell,x_list,dtype=tf.float32)
        #take each element of the output list and pass through a fc layer
        #first initialize fc layer variables
        W_fc = tf.Variable(tf.truncated_normal(shape = [self.hidden_units,6],stddev = 0.1))
        b_fc = tf.Variable(tf.constant(0.,shape = [6]))
        y_list = [tf.nn.tanh(tf.matmul(output_timestep,W_fc) + b_fc) for output_timestep in outputs]
        #at this point y_list of shape [None,3]*self.seq_max_length
        self.op_dict["y"] = tf.pack(y_list,axis = 2)
        #now create get the variables
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