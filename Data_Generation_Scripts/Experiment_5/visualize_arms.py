import numpy as np
import matplotlib.pyplot as plt
import render_arm_util as r
import os
import pickle
import matplotlib.pyplot as plt

#specify link length for 2DOF and 3DOF arm
link_length_2DOF = 40
link_length_3DOF = 30

#also specify the number of examples to consider
num_examples = 100
#specify root directory
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
#first thing is to load data
data_dir = root_dir + \
    "/Data/Experiment_5/Action_Inference_Vision/samples_100000/"
#specify the output dir
output_dir = data_dir + "Visualizations/"

#create dir if it does not exists
if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

with open(data_dir + "states_3DOF.npy","rb") as f:
    states_3DOF = pickle.load(f)[:100,...]

with open(data_dir + "actions_2DOF.npy","rb") as f:
    actions_2DOF = pickle.load(f)[:100,...]

with open(data_dir + "actions_3DOF.npy","rb") as f:
    actions_3DOF = pickle.load(f)[:100,...]

with open(data_dir + "stacked_states_2DOF.npy","rb") as f:
    stacked_states_2DOF = pickle.load(f)[:100,...]

#first shift the 3DOF states
states_3DOF = states_3DOF[states_3DOF < 0] + 2*np.pi
#now get the next 3DOF states 
next_states_3DOF = np.mod(states_3DOF + actions_3DOF,\
                          np.pi*2)

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

#now loop through the number of examples and save the images
for i in range(num_examples):
    #concatenate the renderings of each arm along columns
    rendering_2DOF = np.concatenate((states_2DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     next_states_2DOF_rendered[i,...]),\
                                    axis = 1)

    rendering_3DOF = np.concatenate((states_3DOF_rendered[i,...],\
                                     np.ones((64,2))*255,\
                                     next_states_3DOF_rendered[i,...]),\
                                    axis = 1)

    #now concatenate both of the above over the rows
    output_image = np.float64(np.concatenate((rendering_2DOF,\
                                   np.ones((2,130))*255,\
                                   rendering_3DOF),\
                                  axis = 0))
    #now save the output image
    file_name = output_dir + "output_image_" + str(i) + ".png"
    #import IPython; IPython.embed()
    plt.imsave(file_name,output_image,cmap = "Greys_r")
