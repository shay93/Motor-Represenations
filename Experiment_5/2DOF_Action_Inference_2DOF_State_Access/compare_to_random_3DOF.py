from __future__ import print_function,division 
import numpy as np
import pickle
import matplotlib as mlp
mlp.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
root_dir = os.path.dirname(os.path.dirname(\
                os.getcwd()))
sys.path.append(root_dir)
#we also need the render arm util, so it is necessary
#to add that to the path as well
arm_util_path = root_dir + "/Data_Generation_Scripts/Experiment_5/"
sys.path.append(arm_util_path)
import pickle
import physics_util as phys

#create a directory for the graphs
if not(os.path.exists("Graphs")):
    os.makedirs("Graphs")

num_sequences = 100
episode_length = 100
delta_range = np.pi*0.05
#specify 3DOF link length
link_length_3DOF = 30
#dir from which to load data
pool_data_dir = root_dir + "/Data/Experiment_5/Pool_Data/"
learned_data_dir = root_dir + "/Data/Experiment_5/Learned_2DOF_Policy/"

#load 3DOF trajectory from this
with open(pool_data_dir + "infer_end_effector_3DOF0.npy","rb") as f:
    #shape should be [num_seq,100,2]
    infer_end_effector = pickle.load(f)

#also load the target position for each of these episodes
with open(Learned_2DOF_dir + "target_loc0.npy","rb") as f:
    #shape should be [100,2]
    target_loc_array = np.round(pickle.load(f)*22. + 95)
#now generate a random set of actions and an initial state to
#get the random 3DOF trajectories and then find the end effector

rand_state = (np.random.rand(num_sequences,1,3))*np.pi*2
rand_actions = (np.random.rand(num_sequences,episode_length - 1,3)\
                - 0.5)*2*delta_range

#use the rand_state and actions to get the rand_seq of states
rand_states_3DOF = np.mod(np.cumsum(np.concatenate(\
                    [rand_state,rand_actions],axis = 1),\
                    axis = 1),2.*np.pi)

#use the rand_states to get rand_end effector positions
rand_end_effector = phys.Forward_Kinematics(rand_states_3DOF,\
                                            link_length_3DOF,\
                                            bias = 63)
#tile the target loc array to make computing distance easier
target_loc_repeat = np.repeat(np.expand_dims(target_loc_array,axis = 1),\
                              episode_length,axis = 1)
#now use target loc with end effector to compute some distance metrics
distance_rand = np.linalg.norm(rand_end_effector - target_loc_repeat,\
                               axis = -1)

distance_infer = np.linalg.norm(infer_end_effector - target_loc_repeat,\
                                axis = -1)

#now compute rewards using the distances
reward_rand = (1./((distance_rand/90.) + 1.) - 0.5)*2
reward_infer = (1./((distance_infer/90.) + 1.) - 0.5)*2

#now make two plots one will be average reward per timestep
#across episodes
reward_rand_tstep = np.mean(reward_rand,axis = 0) #[episode]
reward_infer_tstep = np.mean(reward_infer,axis = 0) #[episode]
#now compute reward average per episode
reward_rand_episode = np.mean(reward_rand,axis = 1) #[num_sequences]
reward_infer_episode = np.mean(reward_infer,axis = 1) #[num_sequences]
#print(np.arange(1,episode_length).shape)
#now make plots of the above
fig = plt.figure()
plt.plot(np.arange(1,episode_length+1),\
         reward_rand_tstep,\
         label = "Random Policy")
plt.plot(np.arange(1,episode_length+1),\
         reward_infer_tstep,\
         label = "Infer Policy")
plt.xlabel("Timestep")
plt.ylabel("Avg Reward")
plt.legend()
plt.title("Average Reward per tstep")
fig.savefig("Graphs/Reward_Tstep.png")


fig = plt.figure()
plt.plot(np.arange(1,num_sequences+1),\
         reward_rand_episode,\
         label = "Random Policy")
plt.plot(np.arange(1,num_sequences+1),\
         reward_infer_episode,\
         label = "Infer Policy")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.legend()
plt.title("Average Reward per Episode")
fig.savefig("Graphs/Reward_Episode.png")



