from __future__ import print_function
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from action_inference_model import Action_inference
import numpy as np
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
sys.path.append("/home/shay93/Motor-Represenations/Data_Generation_Scripts/Experiment_5/")
import render_arm_util as r

eval_set_size = 20000
#specify the 2DOF and 3DOF link length
link_length_2DOF = 40.
link_length_3DOF = 30.

data_dir = parent_dir +\
        "/Data/Experiment_5/Action_Inference_Vision/samples_100000/"

#specify the directory in which model saved
load_dir = "model/"

#specify path to the output directory to save these images
output_dir = "High_Low_Error_Visualizations/"

#create the directory if it does not exist
if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

#load the action inference model
model_graph = Action_inference()

#build the graph
op_dict,sess = model_graph.build_graph()

#initialize graph variables and load the correct versions
model_graph.init_graph_vars(sess,op_dict["init_op"])
model_graph.load_graph_vars(sess,op_dict["saver"],load_dir + "model.ckpt")

#now load the dataset

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
    #now rescale the 2DOF actions and concatenate with 2DOF initial
    #states
    initial_states = states[:,:2]/np.pi
    scaled_actions_2DOF = actions_2DOF*(1./np.std(actions_2DOF))
    x = np.concatenate((initial_states,scaled_actions_2DOF),axis =1)
    return x,y


x,y = load_data()


#separate out the eval set
x_eval = x[:eval_set_size,...]
y_eval = y[:eval_set_size,...]


#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["x"]] = x_eval
placeholder_eval_dict[op_dict["y_"]] = y_eval
placeholder_eval_dict[op_dict["keep_prob"]] = 1.

#now evaluate the graph and find the test loss array for each sample

predictions,test_loss_array = model_graph.evaluate_graph(sess,
                                          1,
                                          placeholder_eval_dict,
                                          op_dict["y"],
                                          op_dict["y_"],
                                          loss_op = op_dict["loss"])

#now sort the test loss so as to visualize high and low error preds
sorted_idx = np.argsort(test_loss_array)
#get the indices of the 50 highest and lowest error predictions
subset_idx = np.concatenate((sorted_idx[:50],\
                             sorted_idx[-50:]))

#now load the relevant data in order to visualize

with open(data_dir + "states_3DOF.npy","rb") as f:
    states_3DOF = pickle.load(f)

with open(data_dir + "actions_2DOF.npy","rb") as f:
    actions_2DOF = pickle.load(f)

with open(data_dir + "actions_3DOF.npy","rb") as f:
    actions_3DOF = pickle.load(f)

with open(data_dir + "stacked_states_2DOF.npy","rb") as f:
    stacked_states_2DOF = pickle.load(f)

#also subset the predicted actions and rescale them
pred_actions = predictions[subset_idx,...]*(np.pi/20)


#shift 3DOF states range to 0 to 2pi
states_3DOF[states_3DOF < 0] = states_3DOF[states_3DOF < 0] + 2*np.pi

#now get the next 3DOF states 
next_states_3DOF = np.mod(states_3DOF + actions_3DOF,\
                          np.pi*2)

pred_next_states_3DOF = np.mod(states_3DOF + predictions*(np.pi/20),\
                               np.pi*2)
#save the predicted next states
with open("pred_next_states_3DOF.npy","wb") as f:
    pickle.dump(pred_next_states_3DOF,f)

avg_error = np.mean(np.divide(\
        np.abs(pred_next_states_3DOF - next_states_3DOF),\
        np.abs(next_states_3DOF - states_3DOF)))

median = np.median(np.divide(\
        np.abs(pred_next_states_3DOF - next_states_3DOF),\
        np.abs(next_states_3DOF - states_3DOF)))

print("Avg relative error is %f" % avg_error)
print("Median relative error is %f" % median)

#now subset all the above so that we only render the edge cases
states_3DOF = states_3DOF[subset_idx,...]
actions_2DOF = actions_2DOF[subset_idx,...]
actions_3DOF = actions_3DOF[subset_idx,...]
stacked_states_2DOF = stacked_states_2DOF[subset_idx,...]
pred_next_states_3DOF = pred_next_states_3DOF[subset_idx,...]


#separate out the 2DOF states and next 2DOF states
states_2DOF = stacked_states_2DOF[:,:2]
next_states_2DOF = stacked_states_2DOF[:,2:]

#now render using the above
states_2DOF_rendered = np.squeeze(r.Render_2DOF_arm(\
                            states_2DOF[:,np.newaxis,:],
                            link_length_2DOF))

next_states_2DOF_rendered = np.squeeze(r.Render_2DOF_arm(\
                                next_states_2DOF[:,np.newaxis,:],
                                link_length_2DOF))

#both of the above should return an array of shape [100,64,64,1]

#do the same for the 3DOF states
states_3DOF_rendered = np.squeeze(r.Render_3DOF_arm(\
                            states_3DOF[:,np.newaxis,:],
                            link_length_3DOF))


next_states_3DOF_rendered = np.squeeze(r.Render_3DOF_arm(\
                            next_states_3DOF[:,np.newaxis,:],
                            link_length_3DOF))

pred_next_states_3DOF_rendered = np.squeeze(r.Render_3DOF_arm(\
                                    pred_next_states_3DOF[:,np.newaxis,:],
                                    link_length_3DOF))

#now loop through the number of examples and save the images
for i in range(100):
    #concatenate the renderings of each arm along columns
    rendering_2DOF = np.concatenate((states_2DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     next_states_2DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     next_states_2DOF_rendered[i,...]),\
                                    axis = 1)

    rendering_3DOF = np.concatenate((states_3DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     next_states_3DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     pred_next_states_3DOF_rendered[i,...]),
                                    axis = 1)

    #now concatenate both of the above over the rows
    output_image = np.float64(np.concatenate((rendering_2DOF,\
                                   np.ones((2,196))*255,\
                                   rendering_3DOF),\
                                  axis = 0))
    #now save the output image
    file_name = output_dir + "output_image_" + str(i) + ".png"
    #import IPython; IPython.embed()
    plt.imsave(file_name,output_image,cmap = "Greys_r")
