

from algos.ddpg import DDPG
from base_nn_policy import FeedForwardPolicy
from mod_nn_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from planar_arm_2DOF_lowdim import  Planar_arm_2DOF_lowdim
from planarspace import PlanarSpace
from sandbox.rocky.tf.envs.base import TfEnv
import IPython
#from model_classes import Conv_FeedForwardCritic,Conv_FeedForwardPolicy
import sys
import argparse
import os

def run_task(*_):
    env = Planar_arm_2DOF_lowdim()
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope = "critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        summary_dir = args.summary_dir
    )
    #IPython.embed()
    algorithm.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_dir',type=str,default='/home/shay93/Motor-Represenations/Experiment_4/tensorflow_summaries')
    parser.add_argument('--qf_learning_rate',type=float,default=1e-3)
    parser.add_argument('--scale_reward',type=float,default=1.)
    args = parser.parse_args()
    if not(os.path.exists(args.summary_dir)):
       os.makedirs(args.summary_dir)
    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-planararm",
        seed=2,
  ) 
