from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#add Experiment_3 directory to the python path
parent_dir = os.path.dirname(os.getcwd())
experiment_dir = parent_dir + "/" + "Experiment_3"
sys.path.append(experiment_dir)

from model_classes import physics_emulator_3dof


#now specify the saved model directory
phys_model_dir = experiment_dir + "/" + "joint2image/" + "model/"

#initialize the model
model = physics_emulator_3dof()

#build the graph
op_dict,sess = model.build_graph()
#initialize the variables
model.init_graph_vars(sess,op_dict["init_op"])
#load the saved variables
model.load_graph_vars(sess,op_dict["saver"],phys_model_dir + "model.ckpt")

#define a function that takes in an input of theta [Batch,3] and outputs delta images of [Batch,64,64,1]
def physics_model(theta):
	"""
	theta - Joint angle state for 3DOF arm of shape [Batch,3]
	outputs - Delta images resulting from the input theta of shape [Batch,64,64,1]
	"""
	eval_batch_size = np.shape(theta)[0]
	delta_output,_ = model.evaluate_graph(sess,eval_batch_size,{op_dict["x"] : theta},op_dict["y"],op_dict["y_"], output_shape = [eval_batch_size,64,64,1])
	return delta_output


####test whether this works
#generate a random joint angle

joint_state =  np.array([[0.,0.,np.pi/1.2]])
#pass this through the model and plot the image
output_image = physics_model(joint_state)
plt.imshow(output_image[0,:,:,0],cmap = "Greys_r")
plt.show()




