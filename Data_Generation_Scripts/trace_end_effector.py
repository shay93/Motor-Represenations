import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
import training_tools as tt

#specify the number of sequences for which to trace the end effector
num_sequences =  100
image_dir = "Arm_Trajectory_Images/"
movie_dir = "Arm_Trajectory_Movies/"

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
		#print(i)
		#IPython.embed()
		xpos += link_length*np.cos(np.expand_dims(np.sum(joint_angle_state[...,:i]*np.pi,axis = -1),-1))
		ypos += link_length*np.sin(np.expand_dims(np.sum(joint_angle_state[...,:i]*np.pi,axis = -1),-1))
	return np.concatenate((xpos,ypos),axis = -1)
	


##first load states and actions for the 2DOF and and 3DOF arms
with open("states_2DOF.npy","rb") as f:
    states_2DOF = pickle.load(f) #dim [5000,100,2]
    
with open("actions_2DOF.npy","rb") as f:
    actions_2DOF = pickle.load(f) #dim [5000,100,2]
    
with open("states_3DOF.npy","rb") as f:
    states_3DOF = pickle.load(f) #dim [5000,100,3]
    
with open("actions_3DOF.npy","rb") as f:
    actions_3DOF = pickle.load(f) #dim [5000,100,3]
    
#once we have the states and sequences slice out the number of sequences to trace the
#end effector for

states_2DOF = states_2DOF[:num_sequences,...]
actions_2DOF = actions_2DOF[:num_sequnces,...]
states_3DOF = states_3DOF[:num_sequences,...]
actions_3DOF = actions_3DOF[:num_sequences,...]

#once we have the set of actions and states lets find the current state of the arm
cur_state_2DOF = np.mod(np.cumsum(np.concatenate((states_2DOF[:,0,:],\
                            actions_2DOF[:,:-1,:]),axis = 1),axis = 1),2)
                            
cur_state_3DOF = np.mod(np.cumsum(np.concatenate((states_3DOF[:,0,:],\
                            actions_3DOF[:,:-1,:]),axis = 1),axis = 1),2)
                            
#next step is getting the end effector position for this make use of
#forward kinematics function for this purpose
end_effector_2DOF = forward_kinematics(cur_state_2DOF,50) #dim [num_sequences,100,2]
end_effector_3DOF = forward_kinematics(cur_state_3DOF,30) #dim [num_sequences,100,2]

assert (np.all(np.round(end_effector_2DOF) == np.round(end_effector_3DOF))), "End effectors for the arms do not line up"

#next step is to generate images for these end effector positions and visualize
for j in range(num_sequences):
        seq_directory = image_dir + "sequence_" + str(j) + "/"
        #first create a directory in which to store the sequence for this image
        if not(os.path.exists(seq_directory)):
            os.makedirs(seq_directory)
        
        for timestep in range(99):
            #first get the end effector sequence up until this timestep by slicing
            end_effector_2DOF_snapshot = np.squeeze(end_effector_2DOF[j,:timestep + 1,:])
            end_effector_2DOF_snapshot_list = [np.squeeze(a) \
                                        for a in np.vsplit(end_effector_2DOF_snapshot,timestep)]
            grid_2DOF_obj = tt.grid(grid_size = (128,128))
            #draw the trajectory up until this timestep
            grid_2DOF = grid_2DOF_obj.draw_figure(end_effector_2DOF_snapshot_list)
            #perform the same for the 3DOF end effector as well
            end_effector_3DOF_snapshot = np.squeeze(end_effector_3DOF[j,:timestep + 1,:])
            end_effector_3DOF_snapshot_list = [np.squeeze(a) \
                                        for a in np.vsplit(end_effector_3DOF_snapshot,timestep)]
            grid_3DOF_obj = tt.grid(grid_size = (128,128))
            #draw the trajectory up until this timestep
            grid_3DOF = grid_2DOF_obj.draw_figure(end_effector_3DOF_snapshot_list)
            #now concatenate the two arrays alogn the columns with some white space
            final_image_array = np.concatenate((grid_2DOF,\
                                np.zeros(shape = (128,5)),grid_3DOF),axis = 1)
            
            #save the final image
            plt.imsave(seq_directory + "timestep" + str(timestep),
                final_image_array,
                cmap = "Greys_r")

        with imageio.get_writer(movie_dir + "/" + "sequence_" + str(j) + ".gif", mode='I') as writer:
            for i in range(99):
                image = imageio.imread(seq_directory + "timestep" + str(i))
                writer.append_data(image, meta={'fps' : 5})

