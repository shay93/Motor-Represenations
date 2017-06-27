import tensorflow as tf
import numpy as np
import sys
import os
#add parent dir to path in order to import my_tf_util which will be used to constuct the model
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
from my_tf_util import graph_construction_helper,tensorflow_graph

class Action_inference(tensorflow_graph):
    """
    Given two images corresponding two arm renderings
    Infer the 3DOF action required to replicate the movement
    of the end effector position
    """
    def __init__(self,
                learning_rate = 1e-3,
                gc = graph_construction_helper()):
        self.lr = learning_rate
        self.gc = gc
        self.op_dict = {}
        self.var_dict = {}

    def add_placeholder_ops(self):
        #input should be two channel image
        #pixel values should be floats between 0 and 1
        self.op_dict["x"] = tf.placeholder(tf.float32, \
                                shape=[None,7])
        #output action should be in range -1 and 1
        #furthermore action should be of form [None,3]
        self.op_dict["y_"] = tf.placeholder(tf.float32, \
                                shape=[None,3])
        self.op_dict["keep_prob"] = tf.placeholder(tf.float32)
        return self.op_dict

    def add_model_ops(self):

        h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(self.op_dict["x"], \
                                    [7,200],
                                    "fc_1",
                                     non_linearity = tf.nn.relu)

        h_fc1_dropout = tf.nn.dropout(h_fc1,self.op_dict["keep_prob"])

        h_fc2,W_fc2,b_fc2 = self.gc.fc_layer(h_fc1_dropout, \
                                    [200,300],
                                    "fc_2",
                                    non_linearity = tf.nn.relu)


        h_fc2_dropout = tf.nn.dropout(h_fc2,self.op_dict["keep_prob"])

        h_fc3,W_fc3,b_fc3  = self.gc.fc_layer(h_fc2_dropout,\
                                 [300,50],
                                "fc_3",
                                non_linearity = tf.nn.relu)
        h_fc3_dropout = tf.nn.dropout(h_fc3,self.op_dict["keep_prob"])

        self.op_dict["y"],W_fc4,b_fc4 = self.gc.fc_layer(h_fc3_dropout, \
                                    [50,3],
                                    "Readout",
                                    non_linearity = tf.nn.tanh)


        #now collect all the variables into a list
        var_list = [W_fc1,W_fc2,W_fc3,W_fc4, \
                    b_fc1,b_fc2,b_fc3,b_fc4]

        for var in var_list:
            self.var_dict[var.name] = var

        return self.op_dict

    def add_auxillary_ops(self):
        #define a mean square loss using predicted and expected
        self.op_dict["loss"] = tf.reduce_mean(tf.abs(self.op_dict["y"] - \
                                         self.op_dict["y_"]))
        #now define an optimizer
        opt = tf.train.AdamOptimizer(self.lr)
        #compute the gradients using the optimizer
        grads_and_vars = opt.compute_gradients(self.op_dict["loss"], \
                                               self.var_dict.values())
        #add summary nodes for gradients
        gradient_summary_nodes = [tf.histogram_summary(gv[1].name + "_gradients",gv[0]) \
                        for gv in grads_and_vars]
        #add summary nodes for variable values
        var_summary_nodes = [tf.histogram_summary(var_item[0],var_item[1]) for var_item
                        in self.var_dict.items()]
        #add scalar summary for loss
        tf.scalar_summary("loss summary",self.op_dict["loss"])
        #merge summaries
        self.op_dict["merge_summary_op"] = tf.merge_all_summaries()
        #add train op to the graph
        self.op_dict["train_op"] = opt.apply_gradients(grads_and_vars)
        #add initialization operations
        self.op_dict["init_op"] = tf.initialize_all_variables()
        #add save op
        self.op_dict["saver"] = tf.train.Saver(self.var_dict)
        return self.op_dict


