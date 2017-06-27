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
import physics_util as p

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
    with open(data_dir + "actions_2DOF.npy","rb") as f:
        #2DOF actions in range -pi to pi and normalize to -1 to 1
        y = pickle.load(f)/np.pi
    #next load the 3DOF states
    with open(data_dir + "states_3DOF.npy","rb") as f:
        #states between -pi and pi and normalize to -1 to 1
        states = pickle.load(f)/np.pi
    #load the 3DOF actions and normalize to 
    with open(data_dir + "actions_3DOF.npy","rb") as f:
        #actions in range -pi to pi
        actions_3DOF = pickle.load(f)*(20./np.pi)

    x = np.concatenate((states,\
                        actions_3DOF),axis=1)
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

print("The average validation set loss is %f" % np.mean(test_loss_array))
#now sort the test loss so as to visualize high and low error preds
sorted_idx = np.argsort(test_loss_array)
#get the indices of the 50 highest and lowest error predictions
subset_idx = np.concatenate((sorted_idx[:50],\
                             sorted_idx[-50:]))

#now load the relevant data in order to visualize

with open(data_dir + "states_3DOF.npy","rb") as f:
    states_3DOF = pickle.load(f)[:eval_set_size,...]

with open(data_dir + "actions_2DOF.npy","rb") as f:
    actions_2DOF = pickle.load(f)[:eval_set_size,...]

with open(data_dir + "actions_3DOF.npy","rb") as f:
    actions_3DOF = pickle.load(f)[:eval_set_size,...]

with open(data_dir + "stacked_states_2DOF.npy","rb") as f:
    stacked_states_2DOF = pickle.load(f)[:eval_set_size,...]

#rescale the predicted actions
pred_actions = predictions*(np.pi)

#get states for the 2DOF arm
states_2DOF = stacked_states_2DOF[:,:2]

#shift 2DOF states range to 0 to 2pi
states_2DOF[states_2DOF < 0] = states_2DOF[states_2DOF < 0] + 2*np.pi

#now get the next 2DOF states 
next_states_2DOF = stacked_states_2DOF[:,2:]

#using the predicted actions compute the next set of 2DOF states
pred_next_states_2DOF = np.mod(states_2DOF + pred_actions,\
                               np.pi*2)

#similarly compute the next 3DOF states based on the above
states_3DOF[states_3DOF < 0] = states_3DOF[states_3DOF < 0] + 2*np.pi

next_states_3DOF = np.mod(states_3DOF + actions_3DOF,\
                                np.pi*2)

#save the predicted 2DOF states
with open("pred_next_states_2DOF.npy","wb") as f:
    #predicted 2DOF states in range 0 to 2pi
    pickle.dump(pred_next_states_2DOF,f)

#find the relative error as the ratio between the actual
#change in the end effector pos and the 
rel_error = np.divide(\
        np.abs(p.Forward_Kinematics(pred_next_states_2DOF,link_length_2DOF)\
               - p.Forward_Kinematics(next_states_2DOF,link_length_2DOF)),\
        np.abs(p.Forward_Kinematics(next_states_2DOF,link_length_2DOF)\
               - p.Forward_Kinematics(states_2DOF,link_length_2DOF)))

#now find the average and median of this relative error
avg_rel_error = np.mean(rel_error)
median_rel_error = np.median(rel_error)

print("Avg relative error is %f" % avg_rel_error)
print("Median relative error is %f" % median_rel_error)

#now subset all the above so that we only render the edge cases
states_3DOF = states_3DOF[subset_idx,...]
states_2DOF = states_2DOF[subset_idx,...]
next_states_2DOF = next_states_2DOF[subset_idx,...]
next_states_3DOF = next_states_3DOF[subset_idx,...]

pred_next_states_2DOF = pred_next_states_2DOF[subset_idx,...]


#now render using the above
states_2DOF_rendered = np.squeeze(r.Render_2DOF_arm(\
                            states_2DOF[:,np.newaxis,:],\
                            link_length_2DOF,\
                            coloured = True))

next_states_2DOF_rendered = np.squeeze(r.Render_2DOF_arm(\
                                next_states_2DOF[:,np.newaxis,:],
                                link_length_2DOF,
                                coloured = True))

pred_next_states_2DOF_rendered = np.squeeze(r.Render_2DOF_arm(\
                            pred_next_states_2DOF[:,np.newaxis,:],
                            link_length_2DOF,
                            coloured = True))
#both of the above should return an array of shape [100,64,64,3]

#do the same for the 3DOF states
states_3DOF_rendered = np.squeeze(r.Render_3DOF_arm(\
                            states_3DOF[:,np.newaxis,:],
                            link_length_3DOF,
                            coloured = True))


next_states_3DOF_rendered = np.squeeze(r.Render_3DOF_arm(\
                            next_states_3DOF[:,np.newaxis,:],
                            link_length_3DOF,
                            coloured = True))


#now loop through the number of examples and save the images
for i in range(100):
    #concatenate the renderings of each arm along columns
    rendering_2DOF = np.concatenate((states_2DOF_rendered[i,...],\
                                     np.ones((64,2,3))*255,\
                                     next_states_2DOF_rendered[i,...],\
                                     np.ones((64,2,3))*255,\
                                     pred_next_states_2DOF_rendered[i,...]),\
                                    axis = 1)

    rendering_3DOF = np.concatenate((states_3DOF_rendered[i,...],\
                                     np.ones((64,2,3))*255,\
                                     next_states_3DOF_rendered[i,...],\
                                     np.ones((64,2,3))*255,\
                                     next_states_3DOF_rendered[i,...]),
                                    axis = 1)

    #now concatenate both of the above over the rows
    output_image = np.float64(np.concatenate((rendering_2DOF,\
                                   np.ones((2,196,3))*255,\
                                   rendering_3DOF),\
                                  axis = 0))
    #now save the output image
    file_name = output_dir + "output_image_" + str(i) + ".png"
    #import IPython; IPython.embed()
    plt.imsave(file_name,output_image)
