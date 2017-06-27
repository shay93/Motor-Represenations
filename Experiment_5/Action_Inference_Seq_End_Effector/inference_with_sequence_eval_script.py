import tensorflow as tf
import numpy as np
import sys
import os
import imageio
import matplotlib.pyplot as plt
root_dir = os.path.dirname(os.path.dirname(\
                os.getcwd()))
sys.path.append(root_dir)
#we also need the render arm util, so it is necessary
#to add that to the path as well
arm_util_path = root_dir + "/Data_Generation_Scripts/Experiment_5/"
sys.path.append(arm_util_path)
import pickle
from action_inference_model import Action_inference
import physics_util as phys
from render_arm_util import Render_3DOF_arm
import training_tools as tt

#specify the link length for 2DOF and 3DOF arm
link_length_2DOF = 40
link_length_3DOF = 30
#initialize a shape maker object to help draw
sp = tt.shape_maker()
eval_seq_dir = "Eval_Sequences/"
eval_movie_dir = "Eval_Movies/"
#create the output directories if they 
#dont already exist

if not(os.path.exists(eval_seq_dir)):
    os.makedirs(eval_seq_dir)

if not(os.path.exists(eval_movie_dir)):
    os.makedirs(eval_movie_dir)

#specify the batch size for evaluation 
num_sequences = 100
#specify the episode length used to rollout
episode_length = 100
#specify the load directory for the model 
load_dir = "model/"

#specify the data directory for the data to
#be evaluatated
data_dir = root_dir + "/" + "Data/Experiment_5/"

#load arm renderings for 2DOF arm
with open(data_dir + "reaching_2DOF_render.npy","rb") as f:
    #state_sequences of dimension [num_seq,64,64,100]
    state_sequences = np.float32(pickle.load(f))/255.

with open(data_dir + "initial_state_2DOF.npy","rb") as f:
    #shape of [num_seq,2]
    #range of initial states is [-pi,pi] 
    initial_state_2DOF = pickle.load(f)

with open(data_dir + "target_loc.npy","rb") as f:
    target_loc_array = np.round(pickle.load(f)*22. + 95.)

#once we have paired states it is necessary to get them into
#the form [[99,64,64,2]]*num_sequences i.e. 99 state pairings

#split along first dimension to get a list of sequences
paired_states_list = [np.transpose(np.squeeze(t),axes = [2,0,1,3])\
                      for t in np.split(paired_states,\
                            num_sequences,axis = 0)]

#now build the graph in order to evaluate the inference
model_graph = Action_inference()
#build the graph
op_dict,sess = model_graph.build_graph()
#initialize and load graph parameters
model_graph.init_graph_vars(sess,op_dict["init_op"])
model_graph.load_graph_vars(sess,op_dict["saver"],load_dir + "model.ckpt")

#we have to loop through each element of list and infer
#sequence of actions

#initialize an array to record actions
inferred_actions = np.ndarray(shape = [num_sequences,\
                                       episode_length - 1,\
                                       3])
#import IPython;IPython.embed()
for i,x in enumerate(paired_states_list):
    #import IPython;IPython.embed()
    predictions,test_loss_array = model_graph.evaluate_graph(sess,\
                            episode_length - 1,\
                            {op_dict["x"]: x},\
                            op_dict["y"],\
                            op_dict["y_"],\
                            output_shape = [99,3])
    #predictions of shape [None,3]
    #append predictions to inferred actions
    #predictions in range -1 to 1 but really represent
    #-0.05pi to 0.05pi
    inferred_actions[i,:,:] = (predictions/20.)*np.pi

#Take initial 2DOF state and get end effector position
end_effector_array = phys.Forward_Kinematics(\
                            initial_state_2DOF,link_length_2DOF)
#initialize an array for initial 3DOF state
initial_state_3DOF = np.ndarray(shape = [num_sequences,3])
#shape of end_effector array is [num_seq,2]
for i,end_effector in enumerate(end_effector_array):
    #what is the range of this 
    initial_state_3DOF[i,:] = phys.Inverse_Kinematics_3DOF(\
                            end_effector,link_length_3DOF)

#increase Rank of 3DOF state to make it compatible with actions
initial_state_3DOF = np.expand_dims(initial_state_3DOF,
                                    axis = 1)
print(np.round(end_effector_array[:5,:]))
print(np.round(phys.Forward_Kinematics(initial_state_3DOF,link_length_3DOF)[:5,:]))

#assert \
#np.all(np.round(phys.Forward_Kinematics(initial_state_3DOF,link_length_3DOF))\
        #== np.round(end_effector_array)),"Jacobian Transpose not working"

#now use the initial 3DOF state and 3DOF actions to get 3DOF states
#they should be of shape [num_seq,episode_length,3]
states_3DOF = np.mod(np.cumsum(np.concatenate(\
                    [initial_state_3DOF,inferred_actions],axis = 1),\
                    axis = 1),2.*np.pi)
#3DOF states of shape [num_seq,episode_length,3]
#now visualize 3DOF states

for i in range(num_sequences):
    target_loc = (int(target_loc_array[i,0]),\
                  int(target_loc_array[i,1]))
    sliced_3DOF_states = states_3DOF[i,...]
    rendered_images_3DOF = Render_3DOF_arm(\
                        sliced_3DOF_states,
                        link_length_3DOF,
                        target_loc = target_loc)
    seq_directory = eval_seq_dir + "sequence" + str(i) + "/"
    #create a save directory to store the images
    if not(os.path.exists(seq_directory)):
        os.makedirs(seq_directory)
    #now loop through these rendered images and save in eval_seq
    for j in range(episode_length):
            #IPython.embed()
            #print(np.shape(rendered_images))    
            #get the rendered 2DOF images as well
            rendered_image_2DOF = state_sequences[i,:,:,j]*255.
            #add target to rendered image
            target_points = sp.get_points_to_increase_line_thickness(\
                                                [target_loc],width=5)
            for pt in target_points:
                rendered_image_2DOF[round(pt[0]/2.),\
                                    round(pt[1]/2.)] = 125
            #concatenate the rendered 3DOF and 2DOF images along
            #column axis
            image = np.concatenate((rendered_images_3DOF[j,:,:,0],\
                            np.ones((64,3))*255,\
                            rendered_image_2DOF),axis=1)
            plt.imsave(seq_directory + "timestep" + str(j) + ".png",
                       image,cmap = "Greys_r")

    with imageio.get_writer(eval_movie_dir + "sequence_" + str(i) +
                                ".gif", mode='I') as writer:
        for k in range(episode_length):
            image = imageio.imread(seq_directory + "timestep" + str(k) + ".png")
            writer.append_data(image, meta={'fps' : 5})

# 3DOF end effector position for these states
#this will be of shape [num_seq,episode_length,3]
infer_end_effector = phys.Forward_Kinematics(states_3DOF,
                                             link_length_3DOF,
                                            bias = 63)
#now save this in data directory
with open(data_dir + "infer_end_effector_3DOF.npy","wb") as f:
    pickle.dump(infer_end_effector,f)

#similarly compute the rewards based on these end effectors
#this involves repeating the target loc array
#also
target_loc_repeat = np.repeat(np.expand_dims(target_loc_array,axis = 1),\
                              episode_length,axis = 1)
#now compute reward based of this 
distance_normalized = np.linalg.norm(infer_end_effector - target_loc_repeat,\
                                     axis = -1)/90.
#should have dimension [None,2]
infer_reward = (1./(distance_normalized + 1) - 0.5)*2
#but we also have to add bonus reward for overlap
overlap_boolean = np.all(infer_end_effector == target_loc_repeat,axis = -1)
#now add bonus reward if true
#overlap_boolean_repeat = np.repeat(overlap_boolean[...,np.newaxis],e,axis = 2)
infer_reward[overlap_boolean] += 0.5
#now concatenate the states and target loc in order to get the observations
obs = np.concatenate((states_3DOF,target_loc_repeat),axis = -1)
#import IPython; IPython.embed()
#okay so now save the actions obs and rewards to data dir
with open(data_dir + "infer_obs.npy","wb") as f:
    pickle.dump(obs,f)

with open(data_dir + "infer_actions.npy","wb") as f:
    pickle.dump(inferred_actions,f)

with open(data_dir + "infer_reward.npy","wb") as f:
    pickle.dump(infer_reward,f)