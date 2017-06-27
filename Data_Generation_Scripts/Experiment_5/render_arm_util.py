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
                   target_loc = None,
                   coloured = False):
    """
    state of shape [seq_length,num_tsteps,2]
    return [None,64,64,num_tsteps]
    """
    #get the number of samples in the state
    num_samples = np.shape(state)[0]
    num_tsteps = np.shape(state)[1]
    #initialize an ndarray with the right dimensions
    if not(coloured):
        rendered_images = np.ndarray(shape = [num_samples,64,64,num_tsteps],\
                                 dtype='uint8')
    else:
        rendered_images = np.ndarray(shape = [num_samples,64,64,num_tsteps,3],\
                                 dtype='uint8')
    #first separate out into theta_1 and theta_1
    theta_1 = state[...,0] #at this point dim is [num_samples,num_tsteps]
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
    link1_end_point = np.stack((x_link_1,y_link_1),axis=-1)
    link2_end_point = np.stack((x_link_2,y_link_2),axis=-1)
    #both link end effector arrays should be of shape [None,2] at this point
    #now loop through the end effectors
    if not(coloured):
        for j in range(num_tsteps):
            for i in range(num_samples):
                link1_tuple = (int(link1_end_point[i,j,0]),int(link1_end_point[i,j,1]))
                link2_tuple = (int(link2_end_point[i,j,0]),int(link2_end_point[i,j,1]))
                #get a list of positions that draw the intermediary points
                #between the end effector positions
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
                rendered_images[i,:,:,j] = resize_im
    else: #i.e if coloured 
        for j in range(num_tsteps):
            for i in range(num_samples):
                link1_tuple = (int(link1_end_point[i,j,0]),int(link1_end_point[i,j,1]))
                link2_tuple = (int(link2_end_point[i,j,0]),int(link2_end_point[i,j,1]))
                #get a list of positions that draw the intermediary points
                #between the end effector positions

                link_1_points = sp.get_points_to_increase_line_thickness(\
                                    sp.draw_line(start_point,link1_tuple),\
                                    width = 7)

                link_2_points = sp.get_points_to_increase_line_thickness(\
                                    sp.draw_line(link1_tuple,link2_tuple),\
                                    width = 7)

                #now initialize three grids to save for the rgb values
                r_grid = tt.grid(grid_size = (128,128))
                g_grid = tt.grid(grid_size = (128,128))
                b_grid = tt.grid(grid_size = (128,128))
                #draw the the points for link 1 in red
                r_grid.draw_figure(link_1_points)
                #draw the points for link 2 in blue
                b_grid.draw_figure(link_2_points)
                #add target to image if provided
                if target_loc is not None:
                    #draw the poitns for the target in white
                    target_points = sp.get_points_to_increase_line_thickness(\
                                [target_loc],width=9)
                    r_grid.draw_figure(target_points)
                    b_grid.draw_figure(target_points)
                    g_grid.draw_figure(target_points)
                #now stack the rgb grids to get the image
                cur_image = np.stack((r_grid.grid,\
                                      g_grid.grid,\
                                      b_grid.grid),
                                     axis = -1)

                resize_im = imresize(cur_image,[64,64,3])
                #add the image to the output
                rendered_images[i,:,:,j,:] = resize_im
    return rendered_images



def Render_3DOF_arm(state,
                    link_length,
                    sp = tt.shape_maker(),
                    target_loc = None,
                    coloured = False):
    """
    state of shape [seq_length,num_tsteps,3]
    return[seq_length,64,64,num_tsteps]
    """

    #get the number of samples in the state
    num_samples = np.shape(state)[0]
    num_tsteps = np.shape(state)[1]
    #initialize an ndarray with the right dimensions
    if not(coloured):
        rendered_images = np.ndarray(shape = [num_samples,64,64,num_tsteps],\
                                 dtype='uint8')
    else:
        rendered_images = np.ndarray(shape = [num_samples,64,64,num_tsteps,3],\
                                 dtype='uint8')
    #first separate out into theta_1 and theta_1
    theta_1 = state[...,0] #at this point dim is [seq_length,num_tsteps]
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
    link1_end_point = np.stack((x_link_1,y_link_1),axis=2)
    link2_end_point = np.stack((x_link_2,y_link_2),axis=2)
    link3_end_point = np.stack((x_link_3,y_link_3),axis=2)
    #both link end effector arrays should be of shape [None,num_tsteps,2] at this point
    #now loop through the end effectors
    if not(coloured):
        for j in range(num_tsteps):
            for i in range(num_samples):
                link1_tuple = (int(link1_end_point[i,j,0]),int(link1_end_point[i,j,1]))
                link2_tuple = (int(link2_end_point[i,j,0]),int(link2_end_point[i,j,1]))
                link3_tuple = (int(link3_end_point[i,j,0]),int(link3_end_point[i,j,1]))
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
                rendered_images[i,:,:,j] = resize_im
    else: #i.e we want to colour the image 
        for j in range(num_tsteps):
            for i in range(num_samples):
                link1_tuple = (int(link1_end_point[i,j,0]),int(link1_end_point[i,j,1]))
                link2_tuple = (int(link2_end_point[i,j,0]),int(link2_end_point[i,j,1])) 
                link3_tuple = (int(link3_end_point[i,j,0]),int(link3_end_point[i,j,1]))
                #get a list of positions that draw the intermediary points
                #between the end effector positions

                link_1_points = sp.get_points_to_increase_line_thickness(\
                                    sp.draw_line(start_point,link1_tuple),\
                                    width = 7)

                link_2_points = sp.get_points_to_increase_line_thickness(\
                                    sp.draw_line(link1_tuple,link2_tuple),\
                                    width = 7)

                link_3_points = sp.get_points_to_increase_line_thickness(\
                                    sp.draw_line(link2_tuple,link3_tuple),\
                                    width = 7)
                #now initialize three grids to save for the rgb values
                r_grid = tt.grid(grid_size = (128,128))
                g_grid = tt.grid(grid_size = (128,128))
                b_grid = tt.grid(grid_size = (128,128))
                #draw the the points for link 1 in red
                r_grid.draw_figure(link_1_points)
                #draw the points for link 2 in blue
                b_grid.draw_figure(link_2_points)
                #draw points for link3 in lime
                g_grid.draw_figure(link_3_points)
                #add target to image if provided
                if target_loc is not None:
                    #draw the poitns for the target in white
                    target_points = sp.get_points_to_increase_line_thickness(\
                                [target_loc],width=9)
                    r_grid.draw_figure(target_points)
                    b_grid.draw_figure(target_points)
                    g_grid.draw_figure(target_points)
                #now stack the rgb grids to get the image
                cur_image = np.stack((r_grid.grid,\
                                      g_grid.grid,\
                                      b_grid.grid),
                                     axis = -1)

                resize_im = imresize(cur_image,[64,64,3])
                #add the image to the output
                rendered_images[i,:,:,j,:] = resize_im
    return rendered_images

