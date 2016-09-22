from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import png
import csv
from scipy.integrate import ode
import pickle


class grid:

	def __init__(self,file_name,directory_name):
		self.grid_size = 64
		self.grid = np.zeros((self.grid_size,self.grid_size))
		self.save_name = directory_name + file_name + ".png"	

	def draw_figure(self,pos_array):
	    #this function should take the end effector position and draw on a 64 by 64 grid
	    for pos in pos_array:
	        self.grid[pos[0],pos[1]] = 255
	    return self.grid

	def save_image(self):
	    png.from_array(self.grid.tolist(),'L').save(self.save_name)
	    
	def show_image(self):
	    img = plt.imread(self.save_name)
	    return plt.imshow(img,cmap ='Greys_r')


class shape_maker:

	def __init__(self,directory_name):
		self.std = 6
		self.mu = 32
		self.directory_name = directory_name

	def draw_line(self,start_point,end_point,step_size,tol):
		"""
		Args: A tuple for start_point and end_point,tuple of the form (row,column)
		Returns: Array of tuples each entry corresponding to the position of the line
		"""
		#first unpack the tuples 
		end_point_x,end_point_y = end_point
		start_point_x,start_point_y = start_point
		#initialize an output array
		output_array = [start_point]
		#initialize current point
		current_point_x,current_point_y = start_point
		#first deal with situation where startpoint and endpoint lie in the same row i.e. line is horizontal
		if end_point_x == start_point_x:
			#if end point is to the right of the start point
			if end_point_y > start_point_y:
			    output_array_y = range(start_point_y,end_point_y + 1)
			    output_array_x = [start_point_x] * len(output_array_y)
			else:
			    output_array_y = range(end_point_y,start_point_y + 1)
			    output_array_x = [start_point_x] * len(output_array_y)
			return zip(output_array_x,output_array_y)

		#if on the other hand this special case does not come into consideration determine the gradient of the line
		m = (end_point_y - start_point_y)/(end_point_x - start_point_x)
		#find the y intercept of the line
		c = start_point_y - m*start_point_x
		while abs(current_point_x - end_point_x) > tol:
			#increment the row position of the point and calculate the relevant column position up until the end point is reached
			current_point_y = np.round((m*current_point_x + c))
			if start_point_x < end_point_x:
			    current_point_x += step_size
			else:
			    current_point_x -= step_size
			#since we only want whole numbers we round the current point to get the position of the marker but only
			#include the tuple in our output array if the new point is different than the previous one. 
			if np.round(current_point_x) != output_array[-1][0] or np.round(current_point_y) != output_array[-1][1]:
			    output_array.append((np.round(current_point_x), np.round(current_point_y)))
		return output_array


	def get_vertices(self,shape):

		"""
		we require a function that can take the abstract specification of a geometric shape and translate
		the above to a lower level abstraction in the case of geometric shapes this is just the start point
		and end point of each line segment
		Args: shape - a string specifying the shape to be drawn
		Returns: A list of tuples corresponding to the vertices of the shapes
		"""
		if shape == "Square":
			d = np.floor(self.sample_truncated_normal(9,10,30,1))[0]
			vertices = [(-d,0),(0,d),(d,0),(0,-d)]
		if shape == "Triangle":
			del_1 = np.floor(self.sample_truncated_normal(10,10,40,2))
			del_2 = np.floor(self.sample_truncated_normal(10,10,40,2))
			del_3 = -(del_1 + del_2)
			vertices = [del_1,del_2,del_3]
		if shape == "Rectangle":
			d = np.floor(self.sample_truncated_normal(9,10,30,2))
			vertices = [(-d[0],0),(0,d[1]),(d[0],0),(0,-d[1])]

		return vertices


	def sample_truncated_normal(self,mu,std,threshold,dim):
		"""
		This is utilized by the function that generates vertices given the shape of needed 
		"""
		sample = std*np.random.randn(dim) + mu
		if np.sum(np.logical_or(abs(sample) > threshold , abs(sample) < 5)) > 0:
			return self.sample_truncated_normal(mu,std,threshold,dim)
		else:
			return sample


	def get_points(self,shape):
		"""
		This is the main function and is responsible for calling on the others to generate
		an array of tuples with each tuple denoting a pixel to be filled in the grid
		"""
		pos_array = []
		#generate a random start point for the grid
		start_point = np.round(self.std*np.random.randn(2) + self.mu)
		#then get the vertices of the shape required
		vertices = self.get_vertices(shape)
		if self.check(vertices,start_point):
			for point in vertices:
				end_point = (int(start_point[0] + point[0]),int(start_point[1] + point[1]))
				pos_array.extend(self.draw_line((start_point[0],start_point[1]),end_point,0.001,0.001))
				start_point = end_point
			return pos_array
		else:
			return self.get_points(shape)


	def check(self,control_vector,start_point):
		prev_point = start_point
		vector_list = []
		angle = 90
		j = 0
		for control in control_vector:
			current_point = (int(start_point[0] + control[0]),int(start_point[1] + control[1]))
			vector = np.subtract(current_point,start_point)
			vector_list.append(vector)
			if len(vector_list) > 1:
				angle = np.arccos(np.dot(vector_list[j],vector_list[j+1])/(np.linalg.norm(vector_list[j+1])*np.linalg.norm(vector_list[j])))* 360/(2*np.pi)
				j += 1
			if abs(current_point[0]) > 62 or abs(current_point[1]) > 62 or current_point[0] < 1 or current_point[1] < 1 or abs(angle) < 25 or abs(angle) > 155 :
				return False
			start_point = current_point
		return True


	def gen_shapes(self,shape_str,num):
		#initialize a list to hold the point arrays
		pos_list = [0] * num
		for i in range(num):
			#initialize a grid object
			my_grid = grid(shape_str + str(i),self.directory_name)
			#generate the point
			points_array = self.get_points(shape_str)
			#add points to list
			pos_list[i] = points_array
			#pass the points to the grid
			my_grid.draw_figure(points_array)
			#now save the figure
			my_grid.save_image()
		return pos_list

class two_link_arm:

	def __init__(self,link_length):
		self.ic = [0,0,0,0]
		self.t_end = 5
		self.link_length = link_length


	def forward_dynamics(self,Torques):
		""" Args: Torques - a 2d array of forces and torques of size [num of forces * length of time]
		Returns: Accel - a 2d array of accelerations of size [num of degrees of motion * length of time]
		"""
		#using Newtonian dyanmics the Torque at each link is related to the angle at that link by T = I*theta''
		#use the torques to find the thetas generated over a period of time
		#there are four systems of equations that need to be numerically integrated in order to obtain the correct thetas
		#initialize a states 
		I = 1
		def f(t,y,arg1,arg2):
			y_prime = [y[1],arg1/I,y[3],arg2/I]
			return y_prime
		#set initial displacements and angular velocities to be zero
		r = ode(f).set_integrator('lsoda')
		i = 0
		t0 = 0
		#get the number of timesteps that are used to generate the torques
		_,timesteps = np.shape(Torques)
		#initialize a states array
		states = np.zeros((4,timesteps))
		dt = self.t_end / timesteps
		r.set_initial_value(self.ic, t0).set_f_params(Torques[0,i],Torques[1,i])
		while r.successful() and r.t < (self.t_end - dt):
			r.set_f_params(Torques[0,i],Torques[1,i])
			temp =  r.integrate(r.t + dt)
			#use the above to integrate the
			states[:,i] = temp
			i += 1
		return states


	def forward_kinematics(self,state):
		""" Args: State -  a 2d array of positions of size [num of degrees of motion * length of time]
		    Returns: Effector_Position - a list of tuples consisting of (x,y) position of arm effector and with number of entries 
		                             equal to the length of time being considered.
		"""
		#use the general two link arm with theta1 and theta2
		theta_1 = state[0,:]
		theta_2 = state[1,:]
		x = self.link_length*(np.cos(theta_1) + np.cos(theta_1 + theta_2))
		y = self.link_length*(np.sin(theta_1) + np.sin(theta_1 + theta_2))
		Effector_position = zip(np.round(x),np.round(y))
		return Effector_position

	def inverse_kinematics(self,Effector_position):
	    """ Args: Effector_Position - a list of tuples consisting of (x,y) position of arm effector and with number of entries 
	                                 equal to the length of time being considered.
	        Returns: State -  a 2d array of positions of size [num of degrees of motion * length of time]
	    """
	    #given x-y cartesian coordinates is it possible to obtain the values of theta and phi that correspond to the end effector position
	    theta_1 = [0] * len(Effector_position)
	    theta_2 = [0] * len(Effector_position)
	    for i,pos in enumerate(Effector_position):
	        x = pos[0]
	        y = pos[1]
	        theta_2[i] = np.arccos((x**2 + y**2 - 2*(self.link_length**2))/(2*(self.link_length**2)))
	        k1 = self.link_length*(1 + np.cos(theta_2[i]))
	        k2 = self.link_length*np.sin(theta_2[i])
	        theta_1[i] = np.arctan(y/x) - np.arctan(k2/k1)
	    #get the value of the
	    states = np.vstack((theta_1,theta_2))
	    return states

	def inverse_dynamics(self,states):
	    """ Args : Accel -  a 2d array of accelerations of size [num of degrees of motion * length of time]
	        Returns: Forces - a 2d array of forces and torques of size [num of forces * length of time]
	    """
	    
	    #in case of 2 link arm there are only two forces to worry about call them T_phi and T_theta then using newton's laws
	    #one may obtain the inverse dynamics
	    # Assuming that the moment of inertia for the two bars is the same and that we are computing the acceleration at the bottom
	    #of each bar we may obtain the forces as follows
	    #one has to use the states signal and use it to get the torques
	    torques_1 = self.forward_difference(states[1,:])
	    torques_2 = self.forward_difference(states[3,:])
	    Torques = np.vstack((torques_1,torques_2))
	    return Torques

	def forward_difference(self,ndarray):
		dt = self.t_end /len(ndarray)
		#zero pad the ndarray
		ndarray = np.concatenate(([0],ndarray))
		derivative = np.diff(ndarray) / dt
		return derivative


