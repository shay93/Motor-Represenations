from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import png
import csv
import pickle


class grid:

	def __init__(self,file_name = 'None',directory_name = 'None', grid_size = (64,64)):
		self.grid_size = grid_size
		self.grid = np.zeros((self.grid_size[0],self.grid_size[1]))
		self.save_name = directory_name + file_name + ".png"	

	def draw_figure(self,pos_array, wrap = False, pixel_value = 255.):
	    #this function should take the end effector position and draw on a 64 by 64 grid

		for pos in pos_array:
			if pos[0] < self.grid_size[0] and pos[0] > 0 and pos[1] < self.grid_size[1] and pos[1] > 0:
				self.grid[int(pos[0]),int(pos[1])] = pixel_value

		return self.grid

	def save_image(self):
	    png.from_array(self.grid.tolist(),'L').save(self.save_name)
	    
	def show_image(self):
	    img = plt.imread(self.save_name)
	    return plt.imshow(img,cmap ='Greys_r')


class shape_maker:

	def __init__(self, mu = 32, std = 6, dir_name = None):
		self.std = std
		self.mu = mu
		self.directory_name = dir_name
		self.pixel_width = 3

	def draw_line(self,pos_1,pos_2):
		"""
		Implement Bresenham's algorithm 
		"""
		x0 = pos_1[0]
		y0 = pos_1[1]
		#now unpack second position
		x1 = pos_2[0]
		y1 = pos_2[1]
		dx = abs(x1 - x0)
		dy = abs(y1 - y0)
		x,y = x0,y0
		sx = -1 if x0 > x1 else 1
		sy = -1 if y0 > y1 else 1
		if dx > dy:
		    itr = 0
		    pos_list = [0] * (dx + 1)
		    err = dx/2.0
		    while not(x == x1):
		        pos_list[itr] = (x,y)
		        err -= dy
		        if err < 0:
		            y += sy
		            err += dx
		        x += sx
		        itr += 1
		else:
		    itr = 0
		    pos_list = [0] * (dy + 1)
		    err = dy/2.0
		    while not(y == y1):
		        pos_list[itr] = (x,y)
		        err -= dx
		        if err < 0:
		            x += sx
		            err += dy
		        y += sy
		        itr += 1
		pos_list[-1] = (x,y)
		return pos_list

	def increase_point_thickness(self,pt,grid_size):
		"""
		Returns a list of tuple with each element in a list corresponding to the (x,y) position of a point in the grid
		"""

		if grid_size % 2 == 0:
			return ValueError("Grid Size must be odd")

		grid_width = int((grid_size -1) / 2)
		pt_list = [0]*(2*grid_width + 1)
		for j in range(-grid_width,grid_width + 1):
			left_list = [(pt[0] - j,pt[1] - i) for i in range(1,grid_width + 1)]
			right_list = [(pt[0] - j,pt[1] + i) for i in range(1,grid_width + 1)]
			middle_list = left_list + [(pt[0] - j,pt[1])] + right_list
			pt_list[j] = middle_list

		return pt_list


	def get_points_to_increase_line_thickness(self,pos_list,width = 3):
		#initialize a list to store the information needed
		return [pt for sublist in [self.increase_point_thickness(pos,width) for pos in pos_list] for subsublist in sublist for pt in subsublist]


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
		if shape == "Rhombus":
			l = 5. + 20.*np.random.rand()
			l_x = np.round(np.sin(45/180 * np.pi)*l)
			l_y = np.round(np.cos(45/180 * np.pi)*l)
			vertices = [(-l_x,l_y),(l_x,l_y),(l_x,-l_y),(-l_x,-l_y)]
		if shape == "Hexagon":
			l = 5. + 20.*np.random.rand()
			l_x = np.round(np.sin(60/180 * np.pi)*l)
			l_y = np.round(np.cos(60/180 * np.pi)*l)
			vertices = [(-l_x,l_y),(0,l),(l_x,l_y),(l_x,-l_y),(0,-l),(-l_x,-l_y)]

		
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
		line_segment_list = []
		#generate a random start point for the grid
		start_point = np.round(self.std*np.random.randn(2) + self.mu)
		#then get the vertices of the shape required
		vertices = self.get_vertices(shape)
		if self.check(vertices,start_point):
			for point in vertices:
				end_point = (int(start_point[0] + point[0]),int(start_point[1] + point[1]))
				line_segment = self.draw_line((start_point[0],start_point[1]),end_point)
				line_segment_list.append(line_segment)
				pos_array.extend(line_segment)
				start_point = end_point
			return pos_array,None
		else:
			return self.get_points(shape)

	def onSegment(self,p,q,r):
		"""
		given three points determines whether points are collinear
		"""
		if q[0] <= max(p[0],r[0]) and q[0] >= min(p[0],r[0]) and q[1] <=max(p[1],r[1]) and q[1] >= min(p[1],r[1]):
			return True

		return False

	def orientation(self,p,q,r):
		"""
		determines the orientation of three given points
		"""
		val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0])*(r[1] - q[1])
		if val == 0:
			return 0
		if val > 0:
			return 1
		if val < 0:
			return 2

	def check_intersect(self,p1,q1,p2,q2):
		"""
		Determines if two lines intersect given their end points
		"""
		o1 = self.orientation(p1,q1,p2)
		o2 = self.orientation(p1,q1,q2)
		o3 = self.orientation(p2,q2,p1)
		o4 = self.orientation(p2,q2,q1)

		if (not(o1 == o2) and not(o3 == o4)):
			return True

		if (o1 == 0 and self.onSegment(p1,p2,q1)):
			return True

		if (o2 == 0 and self.onSegment(p1,q2,q1)):
			return True
		
		if (o3 == 0 and self.onSegment(p2,p1,q2)):
			return True

		if (o4 == 0 and self.onSegment(p2,q1,q2)):
			return True

		return False
	
	def check_intersect_all_segments(self,delta_vector,start_point):
		"""
		Given the delta vector and start points determine if any of the lines segments intersect
		"""
		segment_list = [[(start_point[0],start_point[1]),(start_point[0] + delta_vector[0][0],start_point[1] + delta_vector[0][1])]]
		#now loop through the delta vector and append to the segment_list
		for i in xrange(1,len(delta_vector)):
			previous_segment_end_point = segment_list[i - 1][1]
			cur_delta = delta_vector[i]
			current_segment = [previous_segment_end_point,(previous_segment_end_point[0] + cur_delta[0], previous_segment_end_point[1] + cur_delta[1])]
			#now append to the segment list
			segment_list.append(current_segment)
		
		#now loop through segments 
		for i,segment in enumerate(segment_list):
			segment_start = segment[0]
			segment_end = segment[1]
			for j,second_segment in enumerate(segment_list):
				second_segment_start = second_segment[0]
				second_segment_end = second_segment[1]
				if i == j:
					continue
				if self.check_intersect(segment_start,segment_end,second_segment_start,second_segment_end):
					return True

		return False


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
			if abs(current_point[0]) > 62 or abs(current_point[1]) > 62 or current_point[0] < 1 or current_point[1] < 1 or abs(angle) < 25 or abs(angle) > 155 or not(self.check_intersect_all_segments(control_vector,start_point)):
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
			points_array,thicken_points_array = self.get_points(shape_str)
			#add points to list
			pos_list[i] = points_array
			#pass the points to the grid
			my_grid.draw_figure(points_array)
			my_grid.draw_figure(thicken_points_array)
			#now save the figure
			my_grid.save_image()
		return pos_list

class two_link_arm:

	def __init__(self,link_length):
		self.ic = [0,0,0,0]
		self.t_end = 5
		self.link_length = link_length


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

class three_link_arm:

		def __init__(self,link_length):
			self.link_length = link_length
			self.phi_e = np.pi / 2

		def forward_kinematics(self,state):
			""" Args: State -  a 2d array of positions of size [num of degrees of motion * length of time]
    			Returns: Effector_Position - a list of tuples consisting of (x,y) position of arm effector and with number of entrie
    			equal to the length of time being considered.
			"""
			theta_1 = state[0,:]
			theta_2 = state[1,:]
			theta_3 = state[2,:]
			x_e = self.link_length*(np.cos(theta_1) + np.cos(theta_1 + theta_2) + np.cos(theta_1 + theta_2 + theta_3))
			y_e = self.link_length*(np.sin(theta_1) + np.sin(theta_1 + theta_2) + np.sin(theta_1 + theta_2 + theta_3))
			effector_position = zip(np.round(x_e),np.round(y_e))
			return effector_position
