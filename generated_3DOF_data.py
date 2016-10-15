import numpy as np
import matplotlib.pyplot as plt
import training_tools as tt
import pickle


#this script should load the data for the two dof arm and generate the
#the joint sequence for the 3DOF arm

output_path = "/home/shayaan/Research/Redwood/Motor-Represenations/Training_Data_3DOF_Arm/"
link_length_1 = 50
link_length_3 = 60
arm1 = tt.two_link_arm(link_length_1)
arm3 = tt.three_link_arm(link_length_3)
#initialize a dict to hold the data
joint_seq_dict = {}
shape_str_array = ['Rectangle', 'Square', 'Triangle']
#load the joint sequence list for the first arm for each shape
for shape_name in shape_str_array:
	with open("Training_Data_First_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_1) + '.npy',"rb") as f:
		key = "arm1_" + shape_name
		joint_seq_dict[key] = pickle.load(f)


#now for each list and each shape run the forward dynamics for arm 1 to get the position array and use the inverse dynamics of arm3 to get the control list
for shape_name in shape_str_array:
	key = "arm1_" + shape_name
	joint_seq_list = joint_seq_dict[key]
	output_seq_list = [0] * len(joint_seq_list)
	for i,joint_seq in enumerate(joint_seq_list):
		#pass the joint seq to forward kinematics to get pos_list
		end_effector_list = arm1.forward_kinematics(joint_seq)
		#now get the output seq
		output_seq = arm3.inverse_kinematics(end_effector_list)
		output_seq_list[i] = output_seq
	#now save the list
	with open(output_path + shape_name + '.npy','wb') as f:
		pickle.dump(output_seq_list,f)

