import numpy as np
import imageio
import sys
import os
import matplotlib as mlp

mlp.use("Agg")

import matplotlib.pyplot as plt

#get the path for most of the data lies
root_dir = os.path.dirname(os.path.dirname(\
                            os.getcwd()))

sys.path.append(root_dir)
import render_arm_util as r
import physics_util as p
import imageio
import pickle
#append the model dir to the path
model_root_dir = root_dir + \
"/Experiment_5/2DOF_Action_Inference_3DOF_State_Action_Access/"

#add the model dir to the path
sys.path.append(model_root_dir)

from action_inference_model import Action_inference

#specify the link length of the arm 
link_length_2DOF = 40.
link_length_3DOF = 30.

#specify an output directory for the images
output_dir = "Trajectories/"
#specify the directory in which model parameters are saved
model_load_dir = "model/"
#specify directory for output graphs
graph_dir = "Graphs/"

#create this directory if it does not exist
if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

#specify the data directory using the root directory
data_dir = root_dir + "/Data/Experiment_5/Learned_3DOF_Policy/"

#load a sequence of observations, actions and initial states
#from the 3DOF ddpg policy
#fn stands for filename
action_3DOF_fn = "actions_3DOF0.npy" #[-1,1]
initial_state_3DOF_fn = "initial_state_3DOF0.npy" #[-1,1]
obs_3DOF_fn = "obs_3DOF0.npy" # [-1,1] for the states
initial_state_3DOF_fn = "initial_state_3DOF0.npy"#[-pi,pi]
target_loc_fn = "target_loc0.npy"
reward_3DOF_fn = "rewards_3DOF0.npy"
#sampled actions and states are already in correct range
#so just load them as is

with open(data_dir + action_3DOF_fn,"rb") as f:
    #shape is [20,episode_length,3]
    actions_3DOF = pickle.load(f)

with open(data_dir + obs_3DOF_fn,"rb") as f:
    #shape is [20,episode_length,3]
    obs_3DOF = pickle.load(f)

with open(data_dir + initial_state_3DOF_fn,"rb") as f:
    #shape is [20,3]
    initial_state_3DOF = pickle.load(f)

with open(data_dir + target_loc_fn,"rb") as f:
    target_loc = np.round(\
                pickle.load(f)*22. + 95.).astype("int")

with open(data_dir + reward_3DOF_fn,"rb") as f:
    rewards_3DOF = pickle.load(f)

#get the 3DOF states from the loaded 3DOF observations
states_3DOF = obs_3DOF[:,:,:3]
#get the number of sequences and episode length
num_sequences = np.shape(actions_3DOF)[0]
episode_length = np.shape(actions_3DOF)[1]

#get x by concatenating states and actions along last dim
x = np.concatenate((states_3DOF[:,:-1,:],\
                    actions_3DOF[:,:-1,:]),\
                   axis = 2)

#intialize a action_2DOF array to hold the predictions
actions_2DOF = np.ndarray((num_sequences,\
                          episode_length - 1,\
                          2))
#now get a list of sequences for x to sequentially pass
#into the model
x_list = np.split(x,num_sequences,axis = 0)
#each element of the list should be of shape [1,99,6]

#instantiate a dict to hold the data
placeholder_eval_dict = {}

#now load the model
model_graph = Action_inference()

#build the graph
op_dict,sess = model_graph.build_graph()

#initialize graph variables and load the correct versions
model_graph.init_graph_vars(sess,op_dict["init_op"])
model_graph.load_graph_vars(sess,op_dict["saver"],model_load_dir + "model.ckpt")

#now loop through the each sequence and get the predicted 2DOF actions at each
#timestep
for i,x in enumerate(x_list):
    placeholder_eval_dict[op_dict["x"]] = np.squeeze(x)
    placeholder_eval_dict[op_dict["keep_prob"]] = 1.
    #get the predictions for this sequence and assign to 2DOF actions
    actions_2DOF[i,:,:],_= model_graph.evaluate_graph(sess,
                                              episode_length - 1,
                                              placeholder_eval_dict,
                                              op_dict["y"],
                                              output_shape = [episode_length -
                                                              1, 2])

#once we have the actions we should rescale them to their actual values
actions_2DOF = actions_2DOF*np.pi

#now use the rescaled actions and add them to the intitial state in order to
#get the state at every timestep 

#but first get the initial end effector position for each state
init_end_effec = p.Forward_Kinematics(initial_state_3DOF,\
                                link_length_3DOF)


#use the end effector to get the initial 2DOF state
initial_state_2DOF = p.Inverse_Kinematics_2DOF(init_end_effec,\
                                        link_length_2DOF)
#find the sequences which start of at points not reachable by
#the 2DOF arm
reachable_states = np.max(np.logical_not(\
                            np.isnan(initial_state_2DOF)),axis = 1)
#now subset initial states based on the reachability criteria
initial_state_2DOF = initial_state_2DOF[reachable_states,...]
#similary subset actions 2DOF and 3DOF states
actions_2DOF = actions_2DOF[reachable_states,...]
states_3DOF = states_3DOF[reachable_states,...]
#subset the target loc as well
target_loc = target_loc[reachable_states,...]
num_reachable_states = np.shape(initial_state_2DOF)[0]
#find sequences which have initial states that are not reachable
#increase the rank of the 2DOF states
initial_state_2DOF = np.expand_dims(initial_state_2DOF,\
                                    axis = 1)

#shape should now be [20,1,2]


#get the 2DOF states into the correct domain
initial_state_2DOF[initial_state_2DOF < 0] = \
    initial_state_2DOF[initial_state_2DOF < 0] + 2*np.pi
#now get the 2DOF states in the range 0 to 2pi by acting on this
#information
states_2DOF = np.mod(np.cumsum(np.concatenate(\
                (initial_state_2DOF, actions_2DOF),axis = 1),\
                axis = 1),2.*np.pi) #shape [20,100,2]

#get end effector positions for the 2DOF arm 
infer_end_effector = p.Forward_Kinematics(states_2DOF,\
                                    link_length_2DOF,\
                                    bias = 63) #shape [20,100,2]
#now tile the target loc such that it matches this shape
target_loc_tiled = np.repeat(target_loc[:,np.newaxis,:],
                             episode_length,axis = 1)

#now find the distance between end effector and target
distance_infer = np.linalg.norm(infer_end_effector - target_loc_tiled,\
                                axis = -1)

#now compute reward
reward_infer = (1./((distance_infer/90.) + 1.) - 0.5)*2

#get the average reward per tstep
reward_infer_tstep = np.mean(reward_infer,axis = 0) #[episode]

#similarly compute get the average reward for the 3DOF arm
rewards_3DOF_tstep = np.mean(rewards_3DOF,axis = 0) #[episode]

#now create a plot of the two
fig = plt.figure()

plt.plot(np.arange(1,episode_length + 1),\
         rewards_3DOF_tstep,\
         label = "3DOF Rewards")

plt.plot(np.arange(1,episode_length + 1),\
         reward_infer_tstep,\
         label = "Imitated 2DOF Rewards")

plt.xlabel("Timestep")
plt.ylabel("Avg Reward per Timestep (20 seqs)")
plt.legend()
plt.title("Average Reward per tstep")
fig.savefig(graph_dir + "Reward_Tstep.png")

#generate some random 2DOF actions to compare against
rand_actions_2DOF = (np.random.rand(num_reachable_states,\
                              episode_length - 1,2) - 0.5)*2*np.pi

#use rand actions to get rand states
rand_states_2DOF = np.mod(np.cumsum(np.concatenate(\
                (initial_state_2DOF, rand_actions_2DOF),axis = 1),\
                axis = 1),2.*np.pi) #shape [20,100,2]


rand_end_effector = p.Forward_Kinematics(rand_states_2DOF,\
                                    link_length_2DOF,\
                                    bias = 63) #shape [20,100,2]

#now find the distance between end effector and target
distance_rand = np.linalg.norm(rand_end_effector - target_loc_tiled,\
                                axis = -1)

#now compute reward
reward_rand = (1./((distance_rand/90.) + 1.) - 0.5)*2
#get the average reward per tstep
reward_rand_tstep = np.mean(reward_rand,axis = 0) #[episode]


#now create a plot of the two
fig = plt.figure()

plt.plot(np.arange(1,episode_length + 1),\
         reward_infer_tstep,\
         label = "2DOF Imitation Rewards")

plt.plot(np.arange(1,episode_length + 1),\
         reward_rand_tstep,\
         label = "Random 2DOF Rewards")

plt.xlabel("Timestep")
plt.ylabel("Avg Reward per Timestep (20 seqs)")
plt.legend()
plt.title("Average Reward per tstep")
fig.savefig(graph_dir + "Reward_Tstep_rand.png")
#target loc is of shape [20,2] so repeat it such that it is of shape
#we have 2DOF states and 3DOF actions let's render both of them
rendered_3DOF = r.Render_3DOF_arm(states_3DOF*np.pi,\
                  link_length_3DOF,
                  coloured = True,
                  target_loc = target_loc)

#shape is [20,64,64,100,3] for both

rendered_2DOF = r.Render_2DOF_arm(states_2DOF,\
                                  link_length_2DOF,
                                  coloured = True,
                                  target_loc = target_loc)

#now concatenate the two arrays to produce a final image array
final_image_array = np.concatenate((rendered_2DOF,
                                    np.ones((num_reachable_states,64,3,100,3))*255,
                                    rendered_3DOF), axis = 2).astype("uint8")

#loop through the number of sequences and images to get the output
for i in range(num_reachable_states):
     #initialize a writer object
    with imageio.get_writer(output_dir + "output_%d.gif" % i) as writer:
        #now loop through the episode and append each image
        for j in range(episode_length):
            writer.append_data(final_image_array[i,:,:,j,:])






