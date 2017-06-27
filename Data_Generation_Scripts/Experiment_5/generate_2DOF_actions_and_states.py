import numpy as np
import os
import sys
from render_arm_util import Render_2DOF_arm
import pickle

#get the root repo directory
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
#data directory
data_dir = root_dir + "/Data/Experiment_5/Action_Inference_State_Access_2DOF/"
#create the data dir if it does not exist
if not(os.path.exists(data_dir)):
    os.makedirs(data_dir)

#specify the link length for the 2DOF and 3DOF arms
link_length_2DOF = 40
#specify the num_sequences for which to generate data for
num_sequences = 20000
#specify the delta_range, delta range is in radians
delta_range = 0.05*np.pi
#generate some random 2DOF states and 2DOF actions
#states in range 0 to 2pi
states_2DOF = np.random.rand(num_sequences,2)*2*np.pi
#generate some actions in the delta range
actions_2DOF = (np.random.rand(num_sequences,2) - 0.5)*delta_range
#use actions and states to get next states
next_states_2DOF = np.mod(states_2DOF + actions_2DOF,np.pi*2)
#now concatenate the states along a new dimension
stacked_states_2DOF = np.concatenate((states_2DOF,\
                                next_states_2DOF),axis = 1)

with open(data_dir + "stacked_states_2DOF.npy","wb") as f:
    pickle.dump(stacked_states_2DOF,f)

with open(data_dir + "actions_2DOF.npy","wb") as f:
    pickle.dump(actions_2DOF,f)

