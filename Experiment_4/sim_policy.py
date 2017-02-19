from rllab.sampler.utils import rollout
import argparse
import numpy as np
import joblib
import uuid
import tensorflow as tf
from planar2d import  env_2DOF_arm
import json
import matplotlib.pyplot as plt
import os

filename = str(uuid.uuid4())
root_dir = "Eval_Sequences/"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    
    parser.add_argument('--num_sequences',type=int,default=1000,
                        help='Number of sequences to evaluate')

    args = parser.parse_args()

    policy = None
    env = None

    with tf.Session() as sess:
        #with sess.as_default():
        #    policy.get_action(obs)
        data = joblib.load(args.file)
        if 'policy' in data:
            print("Policty in data")
            policy = data['policy']
        else:
            print("Optimizable policy")
            qf = data['optimizable_qfunction']
            policy = qf.implicit_policy
        env = data['env']
        rollout_lst_dict = []
        idx = 0
        while idx < args.num_sequences:
            print("Inside the loop")
            try:
                print("Trying to rollout")
                rollout_dict = rollout(env, policy, max_path_length=args.max_path_length,
                               animated=False, speedup=args.speedup)
                rollout_lst_dict.append(rollout_dict)
            # Hack for now. Not sure why rollout assumes that close is an
            # keyword argument
            except TypeError as e:
                if (str(e) != "render() got an unexpected keyword "
                              "argument 'close'"):
                    raise e
            idx +=1


    #now run through all the sequences and plot them
    for j in range(args.num_sequences):
        seq_directory = root_dir + "sequence_" + str(j) + "/"
        #first create a directory
        if not(os.path.exists(seq_directory)):
            os.makedirs(seq_directory)
        seq_length = np.shape(rollout_lst_dict[j]['observations'])[0]
        for i in range(seq_length):
            plt.imsave(seq_directory + "timestep" + str(i),
                rollout_lst_dict[j][i,:].reshape((64,64)),
                cmap = "Greys_r")


