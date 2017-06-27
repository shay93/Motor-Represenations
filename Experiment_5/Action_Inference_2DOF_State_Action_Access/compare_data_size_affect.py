from __future__ import division
from action_inference_model import Action_inference
import numpy as np
import matplotlib as mlp
mlp.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
import pickle
import png
import results_handling as rh
import tensorflow as tf

eval_set_size = 9984
eval_batch_size = 50
data_root_dir = parent_dir + "/Data/Experiment_5/Action_Inference_Vision/"

data_dir_list = [data_root_dir + "samples_100000/", data_root_dir +
                 "samples_50000/", data_root_dir + "samples_20000/"]

sample_size_list = [100000,50000,20000]
Epoch_list = [200,(90000 // (50000 - eval_set_size))*200,\
              (90000 // (20000 -eval_set_size))*200]
log_dir = "tmp/1e-3"
save_dir = "model/"
output_dir = "Output_Images/"
graph_dir = "Graphs/"
###model parameters
learning_rate = 1e-3
Epochs = 2
batch_size = 64


#check if the directories exist and create them if necessary
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)



def load_data(data_dir):
    #first load the 3DOF actions and rescale to range -1 to 1
    with open(data_dir + "actions_3DOF.npy","rb") as f:
        y = pickle.load(f)*20/np.pi
    #next load the rendered arm observations
    with open(data_dir + "stacked_states_2DOF.npy","rb") as f:
        #states between -pi and pi
        states = pickle.load(f)
    #load the 2DOF actions 
    with open(data_dir + "actions_2DOF.npy","rb") as f:
        #actions in range -pi to pi
        actions_2DOF = pickle.load(f)
    #now rescale the 2DOF actions and concatenate with 2DOF initial
    #states
    initial_states = states[:,:2]/np.pi
    scaled_actions_2DOF = actions_2DOF*(1./np.std(actions_2DOF))
    x = np.concatenate((initial_states,scaled_actions_2DOF),axis =1)
    return x,y

#initialize the figure to save
fig = plt.figure()

for j,data_dir in enumerate(data_dir_list):
    #import IPython; IPython.embed()
    x,y = load_data(data_dir)
    #separate out the training test set data
    x_train = x[eval_set_size:,...]
    y_train = y[eval_set_size:,...]
    #separate out the eval set
    x_eval = x[:eval_set_size,...]
    y_eval = y[:eval_set_size,...]

    #now instantiate the model
    model_graph = Action_inference(learning_rate)

    #build the graph
    op_dict,sess = model_graph.build_graph()

    placeholder_train_dict = {}
    placeholder_train_dict[op_dict["x"]] = x_train
    placeholder_train_dict[op_dict["y_"]] = y_train
    placeholder_train_dict[op_dict["keep_prob"]] = 0.5

    #form the placeholder eval dict
    placeholder_eval_dict = {}
    placeholder_eval_dict[op_dict["x"]] = x_eval
    placeholder_eval_dict[op_dict["y_"]] = y_eval
    placeholder_eval_dict[op_dict["keep_prob"]] = 1.
    model_graph.init_graph_vars(sess,op_dict["init_op"])
    #pass the placeholder dict to the train graph function
    train_loss_array,test_loss_array = model_graph.train_graph(sess,Epoch_list[j],\
                batch_size,placeholder_train_dict,op_dict["train_op"],\
                op_dict["loss"],op_dict["merge_summary_op"],log_dir,
                eval_placeholder_dict = placeholder_eval_dict,
                eval_freq = 100)


    predictions,_= model_graph.evaluate_graph(sess,batch_size,placeholder_eval_dict,op_dict["y"],op_dict["y_"],loss_op = op_dict["loss"])
    tf.reset_default_graph()
    #plot the test and training loss and save as graph
    plt.plot(train_loss_array,label = "Train Loss %d" % sample_size_list[j])
    plt.plot(test_loss_array,label = "Test Loss %d" % sample_size_list[j])
    
plt.ylabel("L1 Loss")
plt.xlabel("Train Step")
plt.title("Loss Plot")
plt.legend()
fig.savefig("Graphs/data_size_loss_comparison.png")
