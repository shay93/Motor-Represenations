from __future__ import division 
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt
import png
import pickle


def plot_npy(i):
	"""
	Takes an input of weights and saves them as images so that training may be observed
	inputs: A sessions object to evaluate the weights
	"""
	root_output_directory = "Image_Autoencoder_Ver%d_Outputs/" %(i)
	W_conv1 = pickle.load(open(root_output_directory + "W_conv1.npy",'rb'))
	W_conv2 = pickle.load(open(root_output_directory + "W_conv2.npy",'rb'))

	#initialize a figure to store the images
	conv1_fig = plt.figure(1,(20.,20.))
	#initialize an image grid
	conv1_grid = ImageGrid(conv1_fig, 111,nrows_ncols=(32 // 8,8),axes_pad = 0.1) 
	for i in range(32):
		kernel = W_conv1[:,:,0,i]
		kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
		conv1_grid[i].imshow(kernel_normed, cmap = "Greys_r")
	
	conv1_fig.savefig(root_output_directory + "Conv1_Kernels.png")
	plt.close(conv1_fig)
	#perform the above for second conv layer as well
	conv2_fig = plt.figure(1,(20.,20.))
	conv2_grid = ImageGrid(conv2_fig, 111,nrows_ncols=(64 // 8,8),axes_pad = 0.1)
	for j in range(64):
		kernel = W_conv2[:,:,0,j]
		kernel_normed = np.divide(kernel,np.mean(kernel)) * 255
		conv2_grid[j].imshow(kernel_normed,cmap = "Greys_r")

	conv2_fig.savefig(root_output_directory + "Conv2_Kernels.png")
	plt.close(conv2_fig)



	training_loss = pickle.load(open(root_output_directory + "training_loss.npy",'rb'))
	training_loss_fig = plt.figure()
	plt.plot(training_loss)
	plt.title('Training Loss')
	training_loss_fig.savefig(root_output_directory + "training_loss.png")
	plt.close(training_loss_fig)

	testing_loss = pickle.load(open(root_output_directory + "testing_loss.npy",'rb'))
	testing_loss_fig = plt.figure()
	plt.plot(testing_loss)
	plt.title('Testing Loss')
	testing_loss_fig.savefig(root_output_directory + "testing_loss.png")
	plt.close(testing_loss_fig)



plot_npy(1)
plot_npy(2)