from __future__ import division 
from inference_with_seq_effec_model import Inference_with_Seq_Effec
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
Epochs = 1000
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
    
    with open(data_dir + "rendered_arm_obs_seq_effec.npy","rb") as f:
        x = pickle.load(f)
    
    with open(data_dir + "effector_3DOF_seq_effec.npy","rb") as f:
        effec = pickle.load(f)
    assert (np.max(effec) < 90. and np.min(effec) > -90.),\
            "Effec pos not in correct range"
    with open(data_dir + "init_state_3DOF_seq_effec.npy","rb") as f:
        init_state = pickle.load(f)

    assert (np.max(init_state) < 2.*np.pi and np.min(init_state) > 0),\
            "3DOF states not in the correct range"
    x = np.float32(x)/255.

    return x,effec,init_state

x,effec,init_state = load_data()
#separate out the training test set data
import IPython; IPython.embed()
x_train = x[eval_set_size:,...]
effec_train = effec[eval_set_size:,...]
init_state_train = init_state[eval_set_size:,...]
#separate out the eval set
x_eval = x[:eval_set_size,...]
effec_eval = effec[:eval_set_size,...]
init_state_eval = init_state[:eval_set_size,...]

#now instantiate the model
model_graph = Inference_with_Seq_Effec(learning_rate,\
                                     seq_length = seq_length)

#build the graph
op_dict,sess = model_graph.build_graph()

placeholder_train_dict = {}
placeholder_train_dict[op_dict["x"]] = x_train
placeholder_train_dict[op_dict["effec_hat"]] = effec_train
placeholder_train_dict[op_dict["init_state"]] = init_state_train

model_graph.init_graph_vars(sess,op_dict["init_op"])
#pass the placeholder dict to the train graph function
model_graph.train_graph(sess,Epochs,batch_size,\
    placeholder_train_dict,op_dict["train_op"],op_dict["loss"],\
    op_dict["merge_summary_op"],log_dir)

model_graph.save_graph_vars(sess,op_dict["saver"],save_dir + "model.ckpt")

#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["effec_hat"]] = effec_eval
placeholder_eval_dict[op_dict["init_state"]] = init_state_eval
predictions,test_loss_array = model_graph.evaluate_graph(sess,\
                eval_batch_size,placeholder_eval_dict,op_dict["y"],\
                output_shape = [eval_set_size,seq_length,3],\
                loss_op = op_dict["loss"])

print("Test Loss array " + str(np.mean(test_loss_array)))
