from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from billiards_2D import Billiards_2D

def get_action(pos,env):
	return np.subtract([pos],env.actor)

#generate the X and Y coordinates of the 64 by 64 grid
X = np.arange(0,64,1)
Y = np.arange(0,64,1)

#get the 2D versions of the arrays using mesh grid
X_2D,Y_2D = np.meshgrid(X,Y)

#initialize the environment
env = Billiards_2D()

#get the target location using the above
target = env.target
actor = env.actor

#initialize the rewards array so that we may populate it
rewards_2D = np.ndarray(shape = (64,64))

#loop through the X and Y arrays and generate an action and get corresponding
#reward with which to populate rewards array

for pos in list(zip(X,Y)):
	action = get_action(pos,env)
	_,reward,_,_ = env.step(action)
	rewards_2D[pos[0],pos[1]] = reward


#now plot the reward in 3D!
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X_2D,Y_2D,rewards_2D,cmap = cm.coolwarm)

fig.savefig("reward_plot.png")

