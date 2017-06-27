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
env_dir = root_dir + "/Experiment_4/low_dim_3DOF_arm/"
#add this to the python path
sys.path.append(env_dir)
import planar_arm_3DOF_lowdim
from render_arm_util import Render_3DOF_arm

#specify a data directory in which to store the sampled states
data_dir = root_dir + "/Data/Experiment_5/Learned_3DOF_Policy/"
if not(os.path.exists(data_dir)):
    os.makedirs(data_dir)
#specify the number of sequences to use
num_sequences = 5
#specify the episode length
episode_length = 100
#specify the batch size for each evaluated sequence
batch_size = 100
#link length for learned policy arm
link_length_3DOF = 30.
#specify the learnt policy file from which to load
file_name=r'/home/shay93/rllab/data/local/ddpg-planararm-3DOF/ddpg-planararm-3DOF_2017_03_21_22_30_26_0001/params.pkl'
#sample learnt 2DOF policy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config = config) as sess:
    data = joblib.load(file_name)
    policy = data['policy']
    env = data['env']
    #initialize an array to hold the rendered arm
    rendered_states = np.ndarray(\
        shape=[batch_size,64,64,episode_length],\
        dtype = 'uint8')
    #initialize an array to hold the initial state of the
    #2DOF arm 
    init_state_array = np.ndarray(\
                    shape=[batch_size,3])
    #initialize an array to hold target loc for each seq
    target_loc_array = np.ndarray(\
                    shape=[batch_size,2])
    #initialize array to hold the 3DOF observations,rewards,actions
    obs_3DOF = np.ndarray(\
                    shape=[batch_size,episode_length,5])
    
    rewards_3DOF = np.ndarray(\
                    shape=[batch_size,episode_length])
    
    actions_3DOF = np.ndarray(\
                    shape=[batch_size,episode_length,3])
    idx = 0
    #initialize an integer to keep track of the batch number
    batch_num = 0
    while batch_num < num_sequences:
        #get the mod index wrt to batch size
        mod_idx = idx % (batch_size)
        rollout_dict = rollout(env, policy,\
                        max_path_length = episode_length,
                        animated = False)
        observations = rollout_dict["observations"]
        obs_3DOF[mod_idx,...] = rollout_dict["observations"]
        rewards_3DOF[mod_idx,...] = rollout_dict["rewards"]
        actions_3DOF[mod_idx,...] = rollout_dict["actions"]
        #slice the states from the observations and append
        rendered_states[mod_idx,...] = np.squeeze(Render_3DOF_arm(\
                            np.pi*observations[np.newaxis,:,:3],\
                            link_length_3DOF))
        init_state_array[mod_idx,...] = np.pi*observations[0,:3]
        target_loc_array[mod_idx,...] = observations[0,3:]
        idx += 1
        #if a batch is full save the array
        if mod_idx == (batch_size - 1):
            #specify the file names for the saved states etc.
            render_file_name = "reaching_3DOF_render"\
                    + str(batch_num) + ".npy"

            init_file_name =  "initial_state_3DOF"\
                    + str(batch_num) + ".npy"

            target_file_name ="target_loc"\
                    + str(batch_num) + ".npy"

            obs_file_name = "obs_3DOF" \
                    + str(batch_num) + ".npy"

            reward_file_name = "rewards_3DOF" \
                    + str(batch_num) + ".npy"

            action_file_name = "actions_3DOF" \
                    + str(batch_num) + ".npy"

            with open(data_dir + obs_file_name,"wb") as f:
                pickle.dump(obs_3DOF,f)

            with open(data_dir + reward_file_name,"wb") as f:
                pickle.dump(rewards_3DOF,f)

            with open(data_dir + action_file_name,"wb") as f:
                pickle.dump(actions_3DOF,f)

            with open(data_dir + render_file_name,"wb") as f:
                pickle.dump(rendered_states,f)

            with open(data_dir + init_file_name,"wb") as f:
                pickle.dump(init_state_array,f)

            with open(data_dir + target_file_name,"wb") as f:
                pickle.dump(target_loc_array,f)

            #now increment the batch number
            batch_num += 1
            print(batch_num)


