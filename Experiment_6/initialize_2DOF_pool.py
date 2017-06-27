from misc.simple_replay_pool import SimpleReplayPool
import pickle
import numpy as np
import os
root_dir = (os.path.dirname(\
                os.getcwd()))
#initialize a simple replay pool and populate it with the
#inferred 3DOF states and actions

#specify data directory which is where we will store these states
data_save_dir = root_dir + "/" + "Data/Experiment_5/Pool_Data/"
#specify the directory from which to load the saved data
data_dir = root_dir + "/Data/Experiment_5/Learned_2DOF_Policy/"
#num of sequences which will be used to initialize the pool
num_sequences = int(1e3)
#the number of sequences that are saved in a given batch
saved_batch_size = 100
#using this find the number of batches
num_saved_batches = num_sequences // saved_batch_size


#add empty pool for 2DOF imitator
pool_2DOF = SimpleReplayPool(10**6,4,2)

#loop through the saved batches and add samples
for batch_num in range(num_saved_batches):
    with open(data_dir + "obs_2DOF%d.npy" % batch_num,"rb") as f:
        obs_2DOF = pickle.load(f)

    with open(data_dir + "actions_2DOF%d.npy" % batch_num,"rb") as f:
        actions_2DOF = pickle.load(f)

    with open(data_dir + "rewards_2DOF%d.npy" % batch_num,"rb") as f:
        rewards_2DOF = pickle.load(f)

    #get the number of sequences in the loaded dataset
    num_sequences = np.shape(obs_2DOF)[0]
    episode_length = np.shape(obs_2DOF)[1]
    for seq_num in range(num_sequences):
        for tstep in range(episode_length):

            if tstep == 99:
                final_state = True
            else:
                final_state = False

            terminal = False
            pool_2DOF.add_sample(obs_2DOF[seq_num,tstep,...],\
                                 actions_2DOF[seq_num,tstep,...],\
                                 rewards_2DOF[seq_num,tstep,...],\
                                terminal,final_state)

#print the pool size
print("The pool size is %d" % pool_2DOF._size)
#just save using pickle in data directory
with open(data_save_dir + "preinitialized_2DOF_pool.npy","wb") as f:
    pickle.dump(pool_2DOF,f)

