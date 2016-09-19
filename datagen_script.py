import training_tools as tt
import pickle



def generate_training_data(num,link_length_1,link_length_2):
	"""
	This function generates the shapes images and saves them to the Training Images folder
	Furthermore, it computes the state of the arm i.e. thetas and saves them as a npy file
	args: num - the number of imagess of each shape to generate, an int
		  link_length - A float or integer specifying the length of each link in the robot arm
		  				in units of pixel width
	returns: -
	"""
	shape_name_array = ['Rectangle', 'Square', 'Triangle']
	for shape_name in shape_name_array:
		my_shape_maker = tt.shape_maker("Training_Images/")
		pos_list = my_shape_maker.gen_shapes(shape_name, num)
		
		#initialize two arms that will be used  
		first_arm = tt.two_link_arm(link_length_1)
		second_arm = tt.two_link_arm(link_length_2)
		#initialize two lists that may be used to capture the states of the arms once they are calculated
		state_list_first_arm = [0] * (num)
		state_list_second_arm = [0] * (num)
		
		for i,pos_array in enumerate(pos_list):
			state_list_first_arm[i] = first_arm.inverse_kinematics(pos_array)
			state_list_second_arm[i] = second_arm.inverse_kinematics(pos_array)
	
		with open("Training_Data_First_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_1) + '.npy','w') as f:
			pickle.dump(state_list_first_arm,f)
			f.close()

		with open("Training_Data_Second_Arm/" + 'saved_state' + '_' + shape_name + '_' + str(link_length_2) + '.npy','w') as f:
			pickle.dump(state_list_second_arm,f)
			f.close()




#saved_array = pickle.load(open('saved_state', 'rb')
generate_training_data(1000,50,80)
