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

def convert_string_ascii(data,fname='test.png'):
    #im = Image.fromstring("RGB",(500,500),data)
    im = Image.frombytes("RGB",(500,500),data)
    im = im.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    im.save(fname)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser to create arguments for the multi sphere land')
    parser.add_argument('--fname',type=str, default='../models/hand_object.xml',help ='The path to the model file name from which to generate data')
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
    print ("Step once")
    MW.step()
    print ("Loop once")
    viewer.loop_once()
    data, width, height = viewer.get_image()
    convert_string_ascii(data)
    IPython.embed()
