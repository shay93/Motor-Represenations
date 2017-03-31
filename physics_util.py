import numpy as np
import IPython

def Inverse_Kinematics_2DOF(end_effector,\
                            link_length):
    """ Args: Effector_Position - An ndarray consisting of the desired
                end_effector positions of an arm
        Returns: State 
    """
    #separate out the x and y positions
    end_effector_x = end_effector[...,0]
    end_effector_y = end_effector[...,1]
    #find the angle betweeen the first and second links
    #define a variable c to help you 
    c = np.divide(end_effector_x**2 + end_effector_y**2 - 2*(link_length**2),2*link_length**2)
    theta_2 = np.arctan2((1 - c**2)**0.5,c)
    k1 = link_length*(1 + np.cos(theta_2))
    k2 = link_length*np.sin(theta_2)
    gamma = np.arctan2(k2,k1)
    theta_1 = np.arctan2(end_effector_y,end_effector_x) - gamma
    states_2DOF = np.concatenate((theta_1[...,np.newaxis], \
                             theta_2[...,np.newaxis]),axis = -1)
    #print(end_effector[:5,0])
    #print(forward_kinematics(states_2DOF,link_length)[:5,0])
    assert (np.all(np.round(Forward_Kinematics(states_2DOF,link_length)) == np.round(end_effector))),\
    "Inverse and Forward Kinematics dont line up"
    return states_2DOF

def Inverse_Kinematics_3DOF(loc,
                           link_length):
    """
    Use the Jacobian transpose to computethe joint angle state
    required to get the end effector of a 3DOF arm to the
    target location

    Args : loc which is a 1D numpy array,
            link_length the length of each link
    Returns : numpy 1d array joint angle state
    """
    #specify alpha which the the learning rate
    alpha = 0.001
    #first randomly initialize a joint angle
    state = (np.random.rand(3)+1)*np.pi
    #get the end effector pos using this state
    end_effector = Forward_Kinematics(state,
                            link_length)
    #specify the distance from end effector to target
    distance = np.linalg.norm(\
                        np.subtract(loc,end_effector))
    #compute the delta in end effector space
    delta_end_effector = np.expand_dims(\
                    np.subtract(loc,end_effector),axis = -1)
    J = Construct_Jacobian_3DOF(state)
    #now apply the update equations up until the L2 norm
    #is greater than 1
    step = 0
    while distance > .3:
        #compute the amount arm should change angles by
        delta_theta = alpha*np.matmul(np.transpose(J),\
                delta_end_effector)
        #IPython.embed()
        #update the state by adding the new delta to it
        state = np.mod(np.sum((np.squeeze(delta_theta)\
                    ,state),axis = 0),2*np.pi)
        #compute the end_effector using this new state
        end_effector = Forward_Kinematics(state,
                                          link_length)
        #get the new distance using this
        distance = np.linalg.norm(\
                        np.subtract(loc,end_effector))
        #print(distance)
        #also get the new delta_end_effector
        delta_end_effector = np.expand_dims(\
                    np.subtract(loc,end_effector),axis = -1)
        #update the Jacobian using the new state
        J = Construct_Jacobian_3DOF(state)
        step += 1
        if step > 30000:
            break
    return state

def Forward_Kinematics(joint_angle_state,
                       link_length,
                       bias = 0):
	"""
	Use the joint angle information to get an end effector position
    for arm with equal length links but with any number of them
	"""
	#find the number of degrees of freedom of the arm
	n_dof = joint_angle_state.shape[-1]
	#initialize the xpos and ypos
	xpos = np.zeros(shape = joint_angle_state.shape[:-1] + (1,)) + bias
	ypos = np.zeros(shape = joint_angle_state.shape[:-1] + (1,)) + bias
	for i in range(1,n_dof+1):
		xpos += link_length*np.cos(np.expand_dims(np.sum(\
                            joint_angle_state[...,:i],axis = -1),-1))
		ypos += link_length*np.sin(np.expand_dims(np.sum(\
                            joint_angle_state[...,:i],axis = -1),-1))
	return np.concatenate((xpos,ypos),axis = -1)

def Construct_Jacobian_3DOF(joint_angle_state):
    """
    Helper function to construct the Jacobian using the provided
    joint angle state
    """
    #get the values of theta_1, theta_2
    #and theta_3 by splitting state
    theta_1,theta_2,theta_3 = np.split(\
            joint_angle_state,3)
    #now construct the Jacobian element by
    #element
    J11 = -np.sin(theta_1) - np.sin(theta_1 + theta_2) - \
            np.sin(theta_1 + theta_2 + theta_2)
    J12 = -np.sin(theta_1 + theta_2) - np.sin(theta_1 + \
                                    theta_2 + theta_3)
    J13 = -np.sin(theta_1 + theta_2 + theta_3)
    #now define the second row
    J21 = np.cos(theta_1) + np.cos(theta_1 + theta_2) + \
            np.cos(theta_1 + theta_2 + theta_3)
    J22 = np.cos(theta_2 + theta_3) + np.cos(theta_1 + \
                                    theta_2 + theta_3)
    J23 = np.cos(theta_1 + theta_2 + theta_3)
    #use these elements to completely specify the Jacobian
    J = np.array([[J11,J12,J13],[J21,J22,J23]])
    return np.squeeze(J)
