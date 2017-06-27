from __future__ import division 
from inference_with_sequence_model import Inference_with_Sequence
import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
import pickle
import png
import results_handling as rh

eval_set_size = 100
eval_batch_size = 50
data_dir = parent_dir + "/" + "Data" + "/Experiment_5/"
log_dir = "tmp/lr_1e_3"
save_dir = "model/"
output_dir = "Output_Images/"
###model parameters
learning_rate = 1e-3
Epochs =1000
batch_size = 500
seq_length = 5

#check if the directories exist and create them if necessary
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def load_data():
    #first load the 2DOF states
    with open(data_dir + "rendered_arm_obs_seq.npy","rb") as f:
        x = pickle.load(f)
    #load the actions now
    with open(data_dir + "actions_3DOF_seq.npy","rb") as f:
        y = pickle.load(f)*20/np.pi
    assert (np.max(y) < 1 and np.min(y) > -1),\
            "3DOF actions not in correct range"
    x = np.float32(x)/255.
    return x,y

x,y = load_data()
#separate out the training test set data
x_train = x[eval_set_size:,...]
y_train = y[eval_set_size:,...]
#separate out the eval set
x_eval = x[:eval_set_size,...]
y_eval = y[:eval_set_size,...]

#now instantiate the model
model_graph = Inference_with_Sequence(learning_rate,\
                                     seq_length = seq_length)

#build the graph
op_dict,sess = model_graph.build_graph()

placeholder_train_dict = {}
placeholder_train_dict[op_dict["x"]] = x_train
placeholder_train_dict[op_dict["y_"]] = y_train
model_graph.init_graph_vars(sess,op_dict["init_op"])
#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)
model_graph.save_graph_vars(sess,op_dict["saver"],save_dir + "model.ckpt")
#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["y_"]] = y_eval
predictions,test_loss_array = model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["y_"],loss_op = op_dict["loss"])

print("Test Loss array " + str(np.mean(test_loss_array)))