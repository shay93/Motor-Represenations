from model_classes import physics_emulator_3dof

eval_set_size = 200
Epochs = 10
batch_size = 500
eval_batch_size =  20
root_dir = "joint2image/"
log_dir = root_dir + "tmp/summary/"

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

output_dir = root_dir + "Output_Images/"

if not os.path.exists(log_dir):
	os.makedirs(output_dir)


#load the data first
def load_data(num):
	with open(ROOT_DIR + "joint_state_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		joint_state_array = pickle.load(f)[:num,...]

	with open(ROOT_DIR + "target_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		target_image_array = pickle.load(f)[:num,...]

	with open(ROOT_DIR + "input_image_array_" + str(DOF) + "DOF" + ".npy","rb") as f:
		input_image_array = pickle.load(f)[:num,...]


	return joint_state_array,target_image_array,input_image_array



joint_state_array,target_image_array,input_image_array = load_data(5000)

#form get the delta image
deta_image_array = target_image_array - input_image_array

#now separate the arrays into the training and eval sets
joint_state_array_train = joint_state_array[eval_set_size:,...]
delta_image_array_train = delta_image_array[eval_set_size:,...]
#now specify the eval set
joint_state_array_eval = joint_state_array[:eval_set_size,...]
delta_image_array_eval = delta_image_array[:eval_set_size,...]
#instantiate physics emulator graph
pe = physics_emulator_3dof(1e-3)

#build the graph
op_dict = pe.build_graph()


#use the opt_dict to construct the placeholder dict
placeholder_train_dict = {}
placeholder_train_dict[op_dict["y_"]] = delta_image_array_train
placeholder_train_dict[op_dict["x"]] = joint_state_array_train

#pass the placeholder dict to the train graph function
pe.train_graph(Epochs,batch_size,placeholder_train_dict,op_dict["train_op"],op_dict["init_op"],op_dict["loss"],op_dict["merge_summary_op"],log_dir)

#form the placeholder eval dict
placeholder_eval_dict = {}
placeholder_eval_dict[op_dict["y_"]] = delta_image_array_eval
placeholder_eval_dict[op_dict["x"]] = joint_state_array_eval

predictions,test_loss_array = pe.evaluate_graph(eval_batch_size,placeholder_eval_dict,op_dict["y"],op_dict["loss"])


def calculate_IOU(predictions,target,directory):
	threshold_list = np.arange(0,0.9,step = 0.025)
	IoU_list = []
	for i,threshold in enumerate(threshold_list):
		good_mapping_count = 0
		bad_mapping_count = 0
		for i in range(EVAL_SIZE):
			arr_pred = np.nonzero(np.round(predictions[i,...]))
			pos_list_pred = zip(arr_pred[0],arr_pred[1])
			arr_input = np.nonzero(target[i,...])
			pos_list_input = zip(arr_input[0],arr_input[1])
			intersection = set(pos_list_pred) & set(pos_list_input)
			union = set(pos_list_input + pos_list_pred)
			if (len(intersection) / len(union)) > threshold:
				good_mapping_count += 1
			else:
				bad_mapping_count += 1

		IoU_list.append(good_mapping_count / TRAIN_SIZE)


	with open(directory + "percentage_correct.npy","wb") as f:
		pickle.dump(IoU_list,f)


def save_images(predictions,target,directory):
	for i in range(eval_set_size):
		plt.imsave(directory + "output_image" + str(i) + ".png", predictions[i,...], cmap = "Greys_r")
		plt.imsave(directory + "target_image" + str(i) + ".png", target[i,...], cmap = "Greys_r")


calculate_IOU(predictions,delta_image_array_eval,root_dir)

save_images(predictions,delta_image_array_eval,output_dir)