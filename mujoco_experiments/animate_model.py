'''
Simple script to move two spheres in a world
Inputs -
Outputs -
Mayur Mudigonda, Feb 2016
'''


from mujoco_py import MjModel, MjViewer
import numpy as np
import argparse
import sys
import IPython
from PIL import Image

def convert_string_ascii(data,idx,fname='test.png'):
    #im = Image.fromstring("RGB",(500,500),data)
    im = Image.frombytes("RGB",(500,500),data)
    im = im.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    if idx is not None:
        im.save(fname+str(idx)+'.png')
    else:
        im.save(fname)
    return
'''
This sets the Initial positions of the joints
and object
'''
def set_init_pos(MW,table_pos,DEBUG=False):
    if DEBUG:
        pos = np.zeros(6)
    else:
        #The values generated must be within the limitations specified in the xml file
        #If not, the joints act like springs and recoil sharply creating very abnormal
        #behavior
        pos = np.zeros(6)
        pos[0] = np.random.uniform(-.75,.75) #proximal rot x
        pos[1] = np.random.uniform(-.65,0.3) #proximal rot z(-.1,0.4),0.45 is the magical contact number
        pos[2] = np.random.uniform(-.7,0.0) #distal rot z
        pos[3] = np.random.uniform(-0.5,0.5) #obj x
        pos[4] = np.random.uniform(0.25,0.6) #obj y
        #pos[5] = -0.5 + table_pos #where 0.1 is the object's z size
        pos[5] = table_pos  #where 0.1 is the object's z size
    MW.data.qpos = pos

    return MW,pos

'''
Sets initial torques/controls for the joints
'''
def set_init_ctrl(MW,DEBUG=False):
    if DEBUG:
        ctrl = np.zeros(13)
    else:
        #ctrl = np.random.rand(13)
        ctrl = np.ones(13)
        ctrl[5] = 0.5
        ctrl[0] = - 2*ctrl[0] # to induce right side roll
        ctrl[1:3] = 0.
        ctrl[7] = 0.
        ctrl[11] = 0.
        print ctrl
    MW.data.ctrl = ctrl

    return MW,ctrl

'''
Sets initial velocities for the joints 
'''
def set_init_vel(MW,DEBUG=False):
    if DEBUG:
        vel = np.zeros(6)
    else:
        vel = np.random.randn(6)
    MW.data.qvel = vel

    return MW,vel

def execute_grasp(MW,DEBUG=False):

    return MW

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser to create arguments for the multi sphere land')
    parser.add_argument('--fname',type=str, default='../models/MPL/MPL_Basic.xml',help ='The path to the model file name from which to generate data')
    args = parser.parse_args()
    print ("Parsing arguments")
    fname = args.fname
    print ("Loading Model")
    MW = MjModel(fname)
    print ("Setting Viewer")
    viewer = MjViewer()
    print ("Setting model to viewer")
    viewer.set_model(MW)
    print ("start viewer")
    viewer.start()
    print ("Setting azimuth so we can see the arm")
    viewer.cam.azimuth = 90
    viewer.cam.distance = 1. 
    viewer.cam.elevation = -45
    #MW,pos = set_init_pos(MW,new_table_height,False)
    #MW,vel = set_init_vel(MW,True)
    MW,ctrl_init = set_init_ctrl(MW,False)
    for ii in range(500):
        print ("Step once")
        MW.step()
        print ("Loop once")
        viewer.loop_once()
        data, width, height = viewer.get_image()
        convert_string_ascii(data,ii,'test')
    IPython.embed()
