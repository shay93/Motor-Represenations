import numpy as np
import os
import sys
from render_arm_util import Render_2DOF_arm
import pickle

#get the root repo directory
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
#data directory
data_dir = root_dir + "/Data" + "/Experiment_5"
#create the data dir if it does not exist
if not(os.path.exists(data_dir)):
    os.makedirs(data_dir)

#specify the link length for the 2DOF and 3DOF arms
link_length_2DOF = 40
link_length_3DOF = 30
#specify the num_sequences for which to generate data for
NUM_SEQUENCES = 20000
#specify the delta_range, delta range is in radians
DELTA_RANGE = 0.05*np.pi
#specify the sequence length
SEQUENCE_LENGTH = 5

def inverse_kinematics_2DOF(end_effector,
                            link_length,
                            check = True):
    """ Args: Effector_Position - An ndarray consisting of the desired
                end_effector positions of an arm
        Returns: State with the same dimensions as theta_2
    """
    #separate out the x and y positions
    end_effector_x = end_effector[...,0]
    end_effector_y = end_effector[...,1]
    #find the angle betweeen the first and second links
    #define a variable c to help you 
    c = np.divide(end_effector_x**2 + end_effector_y**2\
                  - 2*(link_length**2),2*link_length**2)
    theta_2 = np.arctan2((1 - c**2)**0.5,c)
    k1 = link_length*(1 + np.cos(theta_2))
    k2 = link_length*np.sin(theta_2)
    gamma = np.arctan2(k2,k1)
    theta_1 = np.arctan2(end_effector_y,end_effector_x) - gamma
    states_2DOF = np.concatenate((theta_1[...,np.newaxis], \
                             theta_2[...,np.newaxis]),axis = -1)
    #print(end_effector[:5,0])
    #print(forward_kinematics(states_2DOF,link_length)[:5,0])
    if check:
        assert (np.all(np.round(forward_kinematics(states_2DOF,link_length)) == \
            np.round(end_effector))),\
        "Inverse and Forward Kinematics dont line up"
    return states_2DOF

def forward_kinematics(joint_angle_state,link_length):
	"""
	use the joint angle information to get an end effector position
	"""
	#find the number of degrees of freedom of the arm
	n_dof = joint_angle_state.shape[-1]
	#initialize the xpos and ypos
	xpos = np.zeros(shape = joint_angle_state.shape[:-1] + (1,))
	ypos = np.zeros(shape = joint_angle_state.shape[:-1] + (1,))
	for i in range(1,n_dof+1):
		xpos += link_length*np.cos(np.expand_dims(np.sum(joint_angle_state[...,:i],axis = -1),-1))
		ypos += link_length*np.sin(np.expand_dims(np.sum(joint_angle_state[...,:i],axis = -1),-1))
	return np.concatenate((xpos,ypos),axis = -1)

def get_2DOF_states(actions_3DOF,
                    init_states_3DOF):
    """
    Get 2DOF states such that end effector position
    matches the 3DOF end effector over the sequence
    """

    #next states for the 3DOF arm between 0 and 2pi
    #shape is [num_sequences,seq_length + 1,3]
    states_3DOF = np.mod(np.cumsum(np.concatenate(\
                            [init_states_3DOF,actions_3DOF],axis = 1),\
                            axis = 1),2.*np.pi)
    #index out the next states from states_3DOF
    next_states_3DOF = states_3DOF[:,1:,:]
    states_3DOF = states_3DOF[:,:-1,:]
    #now compute the end effector position of the 3DOF arm
    end_effector = forward_kinematics(states_3DOF,link_length_3DOF)
    #use the end effector position to get the state for the 2DOF arm
    states_2DOF = inverse_kinematics_2DOF(end_effector,\
                                          link_length_2DOF,\
                                          check = False)
    #get the next end_effector position at each timestep
    next_end_effector = forward_kinematics(next_states_3DOF,\
                                           link_length_3DOF)
    #get the next states for 2DOF
    next_states_2DOF = inverse_kinematics_2DOF(next_end_effector,\
                                               link_length_2DOF,\
                                              check = False)

    return states_2DOF,next_states_2DOF

def random_3DOF_action_states(num_sequences = NUM_SEQUENCES,
                             delta_range = DELTA_RANGE,
                             seq_length = SEQUENCE_LENGTH):
    #generate random 3DOF actions between -DELTA_RANGE and DELTA_RANGE
    actions_3DOF = (np.random.rand(\
                    num_sequences,seq_length,3) - 0.5)*2*delta_range
    #now randomly select the initial states between 0 and 2pi
    init_states_3DOF = (np.random.rand(num_sequences,1,3))*2*np.pi
    #get the corresponding 2DOF states using the above
    states_2DOF,next_states_2DOF = get_2DOF_states(actions_3DOF,\
                                                   init_states_3DOF)
    #2DOF states should be of shape [num_sequences,seq_length,2]
    #check for nan's i.e. the 3DOF pos not reachable by 2DOF
    states_isnan = np.isnan(states_2DOF)
    next_states_isnan = np.isnan(next_states_2DOF)
    #now continuously tune the chosen 3DOF states and actions
    while np.any(states_isnan) or np.any(next_states_isnan):
        #use boolean arrays to get logical index for 3DOF states
        states_logic = np.repeat(np.max(\
                    states_isnan,axis = -1)[...,np.newaxis],\
                        3, axis = 2)
        next_states_logic = np.repeat(np.max(\
                    next_states_isnan,axis = -1)[...,np.newaxis],\
                        3, axis = 2)
        #regenerate the problematic samples
        actions_3DOF[next_states_logic] = \
                (np.random.rand() - 0.5)*2*delta_range
        #import IPython; IPython.embed()
        init_states_3DOF[np.expand_dims(states_logic[:,0,:],axis = 1)] = \
                np.random.rand()*2*np.pi
        #now get the new 2DOF states
        states_2DOF,next_states_2DOF = get_2DOF_states(actions_3DOF,\
                                                   init_states_3DOF)
        #check for nan's i.e. the 3DOF pos not reachable by 2DOF
        states_isnan = np.isnan(states_2DOF)
        next_states_isnan = np.isnan(next_states_2DOF)
    #now return the 2DOF states and 3DOF actions which satisfy criteria
    assert not((np.any(np.isnan(states_2DOF)) or \
            np.any(np.isnan(next_states_2DOF)))),\
            "There are still nan's in the 2DOF states"
    return actions_3DOF,states_2DOF,next_states_2DOF

actions_3DOF,states_2DOF,next_states_2DOF = random_3DOF_action_states()
states_2DOF_rendered = Render_2DOF_arm(states_2DOF,link_length_2DOF)
next_states_2DOF_rendered = Render_2DOF_arm(next_states_2DOF,link_length_2DOF)
#concatenate the two images along the second dimension
rendered_arm_obs = np.concatenate((states_2DOF_rendered[...,np.newaxis], \
                            next_states_2DOF_rendered[...,np.newaxis]),\
                                  axis = -1)
#now create the directory to save these images
with open(data_dir + "/" + "rendered_arm_obs_seq.npy","wb") as f:
    #shape is [num_sequences,64,64,seq_length,2]
    pickle.dump(rendered_arm_obs,f)

with open(data_dir + "/" + "actions_3DOF_seq.npy","wb") as f:
    #shape is [num_sequences,seq_length,2]
    pickle.dump(actions_3DOF,f)

