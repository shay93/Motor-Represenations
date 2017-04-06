import numpy as np
from scipy.misc import imresize
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
import training_tools as tt

def Render_2DOF_arm(state,
                   link_length,
                   sp = tt.shape_maker(),
                   target_loc = None):
    """
    state of shape [None,2]
    return [None,64,64,1]
    """
    #get the number of samples in the state
    num_samples = np.shape(state)[0]
    #initialize an ndarray with the right dimensions
    rendered_images = np.ndarray(shape = [num_samples,64,64,1],dtype='uint8')
    #first separate out into theta_1 and theta_1
    theta_1 = state[...,0] #at this point dim is [None]
    theta_2 = state[...,1]
    #specify the anchor point
    start_x = 63
    start_y = 63
    start_point = (start_x,start_y)
    #get positions of the end of each link by applying forward kinematics
    x_link_1 = np.round(np.cos(theta_1)*link_length + start_x)
    y_link_1 = np.round(np.sin(theta_1)*link_length + start_y)
    x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*link_length)
    y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*link_length)
    #concatentate the x and y positions to get the tuple position
    link1_end_point = np.stack((x_link_1,y_link_1),axis=1)
    link2_end_point = np.stack((x_link_2,y_link_2),axis=1)
    #both link end effector arrays should be of shape [None,2] at this point
    #now loop through the end effectors
    for i in range(num_samples):
        link1_tuple = (int(link1_end_point[i,0]),int(link1_end_point[i,1]))
        link2_tuple = (int(link2_end_point[i,0]),int(link2_end_point[i,1]))
        #get a list of positions that draw the intermediary points between the end effector positions
        pos_list = sp.draw_line(start_point,link1_tuple)\
                + sp.draw_line(link1_tuple,link2_tuple)
        #now get the extended point list in order to thicken the lines
        arm_points = sp.get_points_to_increase_line_thickness(pos_list,width = 7)
        #now initialize a grid in order to save the correct images
        temp_grid = tt.grid(grid_size = (128,128))
        #draw the the points
        cur_image = temp_grid.draw_figure(arm_points)
        #add target to image if provided
        if target_loc is not None:
            temp_grid.draw_figure(sp.get_points_to_increase_line_thickness(\
                                [target_loc],width=9),pixel_value = 125)
        resize_im = imresize(cur_image,[64,64])
        #add the image to the output
        rendered_images[i,:,:,0] = resize_im
    return rendered_images



def Render_3DOF_arm(state,
                    link_length,
                    sp = tt.shape_maker(),
                    target_loc = None):
    """
    state of shape [None,3]
    return[None,64,64,1]
    """

    #get the number of samples in the state
    num_samples = np.shape(state)[0]
    #initialize an ndarray with the right dimensions
    rendered_images = np.ndarray(shape = [num_samples,64,64,1],dtype='uint8')
    #first separate out into theta_1 and theta_1
    theta_1 = state[...,0] #at this point dim is [None]
    theta_2 = state[...,1]
    theta_3 = state[...,2]
    #specify the anchor point
    start_x = 63
    start_y = 63
    start_point = (start_x,start_y)
    #get positions of the end of each link by applying forward kinematics
    x_link_1 = np.round(np.cos(theta_1)*link_length + start_x)
    y_link_1 = np.round(np.sin(theta_1)*link_length + start_y)
    x_link_2 = np.round(x_link_1 + np.cos(theta_1 + theta_2)*link_length)
    y_link_2 = np.round(y_link_1 + np.sin(theta_1 + theta_2)*link_length)
    x_link_3 = np.round(x_link_2 + np.cos(theta_1 + theta_2 +\
                                          theta_3)*link_length)
    y_link_3 = np.round(y_link_2 + np.sin(theta_1 + theta_2 +\
                                          theta_3)*link_length)
    #concatentate the x and y positions to get the tuple position
    link1_end_point = np.stack((x_link_1,y_link_1),axis=1)
    link2_end_point = np.stack((x_link_2,y_link_2),axis=1)
    link3_end_point = np.stack((x_link_3,y_link_3),axis=1)
    #both link end effector arrays should be of shape [None,2] at this point
    #now loop through the end effectors
    for i in range(num_samples):
        link1_tuple = (int(link1_end_point[i,0]),int(link1_end_point[i,1]))
        link2_tuple = (int(link2_end_point[i,0]),int(link2_end_point[i,1]))
        link3_tuple = (int(link3_end_point[i,0]),int(link3_end_point[i,1]))
        #get a list of positions that draw the intermediary points between the end effector positions
        pos_list = sp.draw_line(start_point,link1_tuple)\
                + sp.draw_line(link1_tuple,link2_tuple)\
                + sp.draw_line(link2_tuple,link3_tuple)
        #now get the extended point list in order to thicken the lines
        arm_points = sp.get_points_to_increase_line_thickness(pos_list,width = 7)
        #now initialize a grid in order to save the correct images
        temp_grid = tt.grid(grid_size = (128,128))
        #draw the the points
        cur_image = temp_grid.draw_figure(arm_points) 
        if target_loc is not None:
            temp_grid.draw_figure(sp.get_points_to_increase_line_thickness(\
                                [target_loc],width=9),pixel_value = 125)
        resize_im = imresize(cur_image,[64,64])
        #add the image to the output
        rendered_images[i,:,:,0] = resize_im
    return rendered_images
=======
>>>>>>> 86a93ed378a1e9ff28ccdfe7b2f439bb0966bee4
