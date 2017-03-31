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
                                shape=[None,64,64,2])
        #output action should be in range -1 and 1
        #furthermore action should be of form [None,3]
        self.op_dict["y_"] = tf.placeholder(tf.float32, \
                                shape=[None,3])
        return self.op_dict

    def add_model_ops(self):
        #take the two channel image and add 3 conv layers
        h_conv1,W_conv1,b_conv1 = self.gc.conv(\
                            self.op_dict["x"],
                            [5,5,2,32],
                            "Conv_1",)
        h_conv2,W_conv2,b_conv2 = self.gc.conv(\
                            h_conv1,
                            [5,5,32,16],
                            "Conv_2",)
        h_conv3,W_conv3,b_conv3 = self.gc.conv(\
                            h_conv2,
                            [5,5,16,1],
                            "Conv_3",)
        print(h_conv3)
        #now flatten the activations in anticipation of fc layers
        h_conv3_flatten = tf.reshape(h_conv3, \
                            shape = [-1,64])
        #pass the flattened activations through some fc layers
        h_fc1,W_fc1,b_fc1 = self.gc.fc_layer(h_conv3_flatten, \
                                    [64,300],
                                    "fc_1")
        h_fc2,W_fc2,b_fc2 = self.gc.fc_layer(h_fc1, \
                                    [300,50],
                                    "fc_2")
        self.op_dict["y"],W_fc3,b_fc3 = self.gc.fc_layer(h_fc2, \
                                    [50,3],
                                    "Readout",
                                    non_linearity = tf.nn.tanh)
        #now collect all the variables into a list
        var_list = [W_conv1,W_conv2,W_conv3,W_fc1,W_fc2,W_fc3, \
                    b_conv1,b_conv2,b_conv3,b_fc1,b_fc2,b_fc3]

        for var in var_list:
            self.var_dict[var.name] = var

        return self.op_dict

    def add_auxillary_ops(self):
        #define a mean square loss using predicted and expected
        self.op_dict["loss"] = tf.reduce_mean(tf.square(self.op_dict["y"] - \
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


