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

eval_set_size = 20000
eval_batch_size = 64
data_dir = parent_dir +\
        "/Data/Experiment_5/Action_Inference_Vision/samples_100000/"
log_dir = "tmp/1e_4_momentum_model10_500Epochs_dropout.8"
save_dir = os.getcwd() + "/model/"
output_dir = "Output_Images/"
graph_dir = "Graphs/"
###model parameters
learning_rate = 1e-4
Epochs = 500
batch_size = 64


#check if the directories exist and create them if necessary
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

def load_data():
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
    with open(data_dir + "states_3DOF.npy","rb") as f:
        #range -pi to pi
        states_3DOF = pickle.load(f)
    #now rescale the 2DOF actions and concatenate with 2DOF initial
    #states
    initial_states = states[:,:2]/np.pi
    scaled_actions_2DOF = actions_2DOF*(1./np.std(actions_2DOF))
    x = np.concatenate((initial_states,\
                        scaled_actions_2DOF,\
                       states_3DOF),axis =1)
    return x,y


x,y = load_data()
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
placeholder_train_dict[op_dict["keep_prob"]] = .8

#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["y_"]] = y_eval
placeholder_eval_dict[op_dict["keep_prob"]] = 1.
model_graph.init_graph_vars(sess,op_dict["init_op"])
#pass the placeholder dict to the train graph function
train_loss_array,test_loss_array = model_graph.train_graph(sess,
                                                Epochs,
                                                batch_size,
                                                placeholder_train_dict,
                                                op_dict["train_op"],
                                                op_dict["loss"],
                                                op_dict["merge_summary_op"],
                                                log_dir,
                                                eval_placeholder_dict=placeholder_eval_dict,
                                                eval_freq = 1000,
                                                save_directory = save_dir + "model.cpkt",
                                                save_op = op_dict["saver"]
                                                )

model_graph.save_graph_vars(sess,op_dict["saver"],save_dir + "model.ckpt")

predictions,_= model_graph.evaluate_graph(sess,eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["y_"],loss_op = op_dict["loss"])

#plot the test and training loss and save as graph
fig = plt.figure()
plt.plot(train_loss_array,label = "Train Loss")
plt.plot(test_loss_array,label = "Test Loss")
plt.ylabel("L1 Loss")
plt.xlabel("Train Step")
plt.title("Loss Plot")
plt.legend()
fig.savefig("Graphs/Loss_1e_4_500Epochs_8e1dropout.png")

