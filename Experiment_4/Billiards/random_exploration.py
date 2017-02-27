import numpy as np
import matplotlib.pyplot as plt
import os
from billiards_2D import Billiards_2D
import imageio
movie_dir = "Random_Exploration_Movies/"

if not(os.path.exists(movie_dir)):
	os.makedirs(movie_dir)

num_movies = 5

#initialize the environment
env = Billiards_2D()

#generate random actions
actions = (np.random.rand(num_movies,2,100) - 0.5) * 6


for i in range(num_movies):
	#index out the actions for the movie
	trajectory_actions = actions[i,...]

	with imageio.get_writer(movie_dir + "sequence_" + str(i) + ".gif",mode = 'I') as writer:
		#now enumerate through the actions
		for j in range(100):
			displacement = np.expand_dims(trajectory_actions[:,j],0)
			#now compute the environment step
			cur_image,reward,done,_ = env.step(displacement)
			writer.append_data(image,meta = {'fps':5})




