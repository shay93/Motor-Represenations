
import numpy as np
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import training_tools as tt
import pickle
import IPython

def inverse_kinematics_2DOF(end_effector,link_length):
    """ Args: Effector_Position - An ndarray consisting of the desired
                end_effector positions of an arm
        Returns: State with the same dimensions as the 
    """
    #separate out the x and y positions
    end_effector_x = end_effector[...,0]
    end_effector_y = end_effector[...,1]
    #find the angle betweeen the first and second links
    #define a variable c to help you 
    c = np.divide(end_effector_x**2 + end_effector_y**2 - 2*(link_length**2),2*link_length**2)
    theta_2 = np.arctan2((1 - c**2)**0.5,c)
    k1 = link_length*(1 + np.cos(theta_2))
    k2 = link_length*np.sin(theta_2)
    gamma = np.arctan2(k2,k1)
    theta_1 = np.arctan2(end_effector_y,end_effector_x) - gamma
    
    states_2DOF = np.concatenate((theta_1[...,np.newaxis], \
                             theta_2[...,np.newaxis]),axis = -1)

    print(end_effector[0,:5,:])
    print(forward_kinematics(states_2DOF,link_length)[0,:5,:])
    IPython.embed()
    assert (np.all(np.round(forward_kinematics(states_2DOF,link_length)) == np.round(end_effector))),\
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
		print(i)
		#IPython.embed()
		xpos += link_length*np.cos(np.expand_dims(np.sum(joint_angle_state[...,:i],axis = -1),-1))
		ypos += link_length*np.sin(np.expand_dims(np.sum(joint_angle_state[...,:i],axis = -1),-1))
	return np.concatenate((xpos,ypos),axis = -1)

#specify the num_sequences for which to generate data for
num_sequences = 5
#specify the sequence length, we could make this variable but let's stick
#to fixed length sequences for now
sequence_length = 100
#specify the delta_range, delta range is in radians normalized to range -1 to 1
delta_range = 0.05
#generate a random sequence of 3DOF joint ACTIONS between -delta_range and delta_range
actions_3DOF = (np.random.rand(num_sequences,sequence_length,3) - 0.5)*2*delta_range
#now randomly select the initial states between 0 and 2
initial_states_3DOF = (np.random.rand(num_sequences,1,3))*2
#concatenate the states with the actions and perform a cumulative sum to obtain
#states for the 3DOF arm between 0 and 2
states_3DOF = np.mod(np.cumsum(np.concatenate(
    [initial_states_3DOF,actions_3DOF[:,:-1,:]],axis = 1),axis=1),2.)
#get the next states at each timestep by adding the actions to the states
next_states_3DOF = np.mod(states_3DOF + actions_3DOF,2.)
    
#as a sanity check the dimensions of states_3DOF
#print(states_3DOF.shape)
#check what the min and max of the 3DOF states are
print(np.max(states_3DOF))
print(np.min(states_3DOF))
#now compute the end effector position of the 3DOF arm
end_effector = forward_kinematics(states_3DOF,30)
#use the end effector position to get the state for the 2DOF arm
states_2DOF = inverse_kinematics_2DOF(end_effector,50)
#get the next end_effector position at each timestep
next_end_effector = forward_kinematics(next_states_3DOF,30)
#get the next states for 2DOF
next_states_2DOF = inverse_kinematics_2DOF(next_end_effector,50)
#use the next states and current states for the 2DOF to get the 2DOF actions
actions_2DOF = next_states_2DOF - states_2DOF


#now save all of the above in a data directory 
with open("states_2DOF.npy","wb") as f:
    pickle.dump(states_2DOF,f)
    
with open("states_3DOF.npy","wb") as f:
    pickle.dump(states_3DOF,f)
    
