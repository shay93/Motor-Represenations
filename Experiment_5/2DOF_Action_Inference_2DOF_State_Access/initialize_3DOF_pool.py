from misc.simple_replay_pool import SimpleReplayPool
import pickle
import numpy as np
import os
root_dir = os.path.dirname(os.path.dirname(\
                os.getcwd()))
#initialize a simple replay pool and populate it with the
#inferred 3DOF states and actions

#specify data directory which is where we will store these states
data_dir = root_dir + "/" + "Data/Experiment_5/Pool_Data/"
#num of sequences which will be used to initialize the pool
num_sequences = int(1e4)
#the number of sequences that are saved in a given batch
saved_batch_size = 100
#using this find the number of batches
num_saved_batches = num_sequences // saved_batch_size
def shift_theta_range(angle_array):
    """
    Shifts range of angle given between to 0 to 2pi to -pi and pi
    """
    array_length = np.shape(angle_array)[0]
    shifted_angles = np.zeros(array_length)
    for i,angle in enumerate(angle_array):
        if angle > 1.:
            shifted_angles[i] = angle - 2.
        else:
            shifted_angles[i] = angle
    return shifted_angles


#load inferred 3DOF states and actions
pool_3DOF = SimpleReplayPool(10**6,5,3)


#loop through the saved batches and add samples

with open(data_dir + "infer_obs.npy","rb") as f:
    inferred_obs = pickle.load(f)

with open(data_dir + "infer_actions.npy","rb") as f:
    inferred_actions = pickle.load(f)

with open(data_dir + "infer_reward.npy","rb") as f:
    infer_reward = pickle.load(f)

#get the number of sequences in the loaded dataset
num_sequences = np.shape(inferred_obs)[0]
episode_length = np.shape(inferred_obs)[1]
#loop over the sequences and episodes and add samples
#states have range -1 to 1 representing -pi to pi
#actions have range -1 to 1 representing -0.05pi to 0.05pi
inferred_actions = (inferred_actions*20)/np.pi
#reward will be as is
for seq in range(num_sequences):
    for tstep in range(episode_length - 1):
        state_obs = shift_theta_range(inferred_obs[seq,tstep,:3]/np.pi)
        target_obs  = (inferred_obs[seq,tstep,3:5] - 95.)/22.
        observation = np.concatenate([state_obs,target_obs],axis = 0)
        action = inferred_actions[seq,tstep,...]
        reward = infer_reward[seq,tstep,...]
        final_state = False
        terminal = False
        pool_3DOF.add_sample(observation,action,reward,\
                             terminal,final_state)

#just save using pickle in data directory
with open(data_dir + "preinitialized_pool.npy","wb") as f:
    pickle.dump(pool_3DOF,f)

