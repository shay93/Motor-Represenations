
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle

parent_dir = os.path.dirname(os.getcwd())
experiment_3_dir  = parent_dir + "/" + "Experiment_3"
sys.path.append(parent_dir)

import training_tools as tt
from planar2d import env_2DOF_arm

root_dir = os.getcwd() + "/" + "Reaching_Task_2DOF_Random"

#create the root_dir as
if not(os.path.exists(root_dir)):
	os.makedirs(root_dir)

"""
Generate some random trajectories and determine number of instances
in which the agent achieved the objective and what percentage of the
1000 or so trajectories were successful. Generate a random trajectory
save
"""
num_trajectories = 1000

#initialize an environment
env = env_2DOF_arm()

def gen_delta_trajectory(num_steps = 100):
	return [np.squeeze(delta) for delta in np.split((np.random.rand(2,num_steps) - 0.5)*(np.pi/2),num_steps,1)]

#initialize a list for the rewards
avg_episode_returns = []
avg_episode_done = []
for i in xrange(num_trajectories):
	episode_done = []
	episode_returns = []
	delta_traj = gen_delta_trajectory()
	#now create a directory in which to store the output images
	traj_dir = root_dir + "/" + "trajectory" + str(i)
	if i < 100:
		if not(os.path.exists(traj_dir)):
			os.makedirs(traj_dir)
	#now loop through the deltas and calculate the next image at each step
	for j,delta in enumerate(delta_traj):
		cur_image,reward,done,info = env.step(delta)
		episode_returns.append(reward)
		episode_done.append(done)
		#save the image corresponding to the current image
		if i < 100:
			plt.imsave(traj_dir + "/" + "timestep" + str(j), np.reshape(cur_image,(64,64)), cmap = "Greys_r")
		if done:
			env.reset()
			break
	env.reset()

	avg_episode_returns.append(np.average(reward))
	avg_episode_done.append(np.amax(episode_done))

print(np.sum(avg_episode_done)/num_trajectories)
print(np.sum(np.greater(avg_episode_returns,0.0))/num_trajectories)

with open("terminal_list_random.npy",wb) as f:
	pickle.dump(avg_episode_done)