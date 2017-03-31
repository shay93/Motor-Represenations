from rllab.sampler.utils import rollout
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import uuid
import os
import pickle
import imageio
import IPython
import sys
#it is necessary to import the planar arm environment
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
env_dir = root_dir + "/Experiment_4/low_dim_2DOF_arm/"
#add this to the python path
sys.path.append(env_dir)
import planar_arm_2DOF_lowdim
from render_arm_util import Render_2DOF_arm

#specify a data directory in which to store the sampled states
data_dir = root_dir + "/Data/Experiment_5/"
#specify the number of sequences to use
num_sequences = 100
#specify the learnt policy file from which to load
file_name=r'/home/shay93/rllab/data/local/ddpg-planararm/ddpg-planararm_2017_03_21_01_13_24_0001/params.pkl'
#sample learnt 2DOF policy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    data = joblib.load(file_name)
    policy = data['policy']
    env = data['env']
    #initialize an array to hold the rendered arm
    rendered_states = np.ndarray(shape=[100,64,64,num_sequences], \
                                dtype = 'uint8')
    #initialize an array to hold the initial state of the
    #2DOF arm 
    init_state_array = np.ndarray(shape=[num_sequences,2])
    #initialize an array to hold target loc
    target_loc_array = np.ndarray(shape=[num_sequences,2])
    idx = 0
    while idx < num_sequences:
        rollout_dict = rollout(env, policy,\
                        max_path_length = 100,
                        animated = False)
        observations = rollout_dict["observations"]
        #slice the states from the observations and append
        rendered_states[...,idx] = np.squeeze(Render_2DOF_arm(\
                                    np.pi*observations[:,:2],40))
        init_state_array[idx,...] = observations[0,:2]
        target_loc_array[idx,...] = observations[0,2:4]
        idx += 1

#reshape the rendered arm states
rendered_states = np.transpose(rendered_states, \
                            axes = [3,1,2,0])
#rendered shapes now of dimension [num_sequences,64,64,episode_length]
#now save the arrays in the data directory
with open(data_dir + "reaching_2DOF_render.npy","wb") as f:
    pickle.dump(rendered_states,f)

with open(data_dir + "initial_state_2DOF.npy","wb") as f:
    pickle.dump(init_state_array,f)

with open(data_dir + "target_loc.npy","wb") as f:
    pickle.dump(target_loc_array,f)

