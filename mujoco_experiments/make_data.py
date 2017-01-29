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
import os
import pickle

def convert_string_ascii(data,idx,run,DEBUG=True,path=None):
    #im = Image.fromstring("RGB",(500,500),data)
    im = Image.frombytes("RGB",(500,500),data)
    im = im.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    if DEBUG is False:
       im.save(path+'/run'+str(run)+'/test'+str(idx)+'.png')
    return
'''
This sets the Initial positions of the joints
and object
'''
def set_init_pos(MW,DEBUG=False):
    if DEBUG:
        pos = np.zeros(6)
    else:
        #The values generated must be within the limitations specified in the xml file
        #If not, the joints act like springs and recoil sharply creating very abnormal
        #behavior
        pos = np.zeros(6)
        tmp = 0.1*np.random.rand(3)
        pos[:3] = tmp
        pos[3:] = tmp
        print("Position is {}",pos)
    MW.data.qpos = pos

    return MW,pos

'''
Sets initial torques/controls for the joints
'''
def set_init_ctrl(MW,DEBUG=False):
    if DEBUG:
        ctrl = np.zeros(6)
    else:
        ctrl = np.zeros(6)
        tmp = np.random.randn(3)
        ctrl[:3] = tmp
        ctrl[3:] = -tmp
        ctrl[4] = -1*ctrl[4]
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
    parser.add_argument('--fname',type=str, default='../mujoco_model/arm_3link_push.xml',help ='The path to the model file name from which to generate data')
    parser.add_argument('--output',type=str, default='/media/Gondor/Data/motor-learning/simple/',help ='The path to the  output where the data is going to be stored')
    parser.add_argument('--nsamples',type=int, default=1,help='Number of samples')
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
    viewer.cam.distance = 2.5
    #viewer.cam.elevation = -45
    viewer.cam.elevation = 0
    viewer.cam.trackbodyid = -1
    viewer.cam.lookat[0] = 0.
    qpos = []
    ctrl = []
    for jj in range(args.nsamples):
        #Make folder
        try:
            os.mkdir(args.output+'run'+str(jj))
        except:
            pass
        MW,pos = set_init_pos(MW,False)
        MW,vel = set_init_vel(MW,True)
        MW,ctrl_init = set_init_ctrl(MW,False)
        qpos.append(pos)
        ctrl.append(ctrl_init)
        for ii in range(200):
            # print ("Step once")
            MW.step()
            #print ("Loop once")
            viewer.loop_once()
            data, width, height = viewer.get_image()
            convert_string_ascii(data,ii,jj,False,args.output)
    #Writing out things
    pickle.dump(qpos,open(args.output+'qpos.pkl','wb'))
    pickle.dump(ctrl,open(args.output+'ctrl.pkl','wb'))
