import numpy as np
import matplotlib.pyplot as plt
import training_tools as tt
import pickle

arm1_path = "/home/shayaan/Research/Redwood/Motor-Represenations/Training_Data_First_Arm/"
arm2_path = "/home/shayaan/Research/Redwood/Motor-Represenations/Training_Data_Second_Arm/"

shape_name_array = [ 'Square', 'Triangle']
link_length_1 = 50
link_length_2 = 80



for shape_name in shape_name_array:
	myfile = open("Training_Data_First_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_1) + '.npy',"rb") 

	state_list_first_arm = pickle.load(myfile)

	
	#initialize two arms that will be used  
	first_arm = tt.two_link_arm(link_length_1)
	second_arm = tt.two_link_arm(link_length_2)
	#initialize two lists that may be used to capture the states of the arms once they are calculated
	state_list_second_arm = [0] * (num)
	
	for i,state_array in enumerate(state_list_first_arm):
		pos_array = first_arm.forward_kinematics(state_array)
		state_list_second_arm[i] = second_arm.inverse_kinematics(pos_array)



	with open("Training_Data_Second_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_2) + '.npy',"wb") as f:
		pickle.dump(state_list_second_arm,f)
		f.close()
