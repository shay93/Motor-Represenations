

from algos.ddpg import DDPG
from planar_arm_nn_policy import FeedForwardPolicy
from planar_arm_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from planar_arm_2DOF_lowdim import  Planar_arm_2DOF_lowdim
import IPython
import sys
import argparse
import os

def run_task(*_):
    env = Planar_arm_2DOF_lowdim(num_steps=args.max_path_length)
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
        summary_dir = args.summary_dir,
        soft_target_tau = args.soft_target_tau,
        policy_learning_rate = args.policy_lr,
        qf_learning_rate = args.qf_lr,
        max_path_length=args.max_path_length,
    )
    #IPython.embed()
    algorithm.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_dir',type=str,default='/home/shay93/Motor-Represenations/Experiment_4/tensorflow_summaries')
    parser.add_argument('--qf_lr',type=float,default=1e-3)
    parser.add_argument('--policy_lr',type=float,default=1e-4)
    parser.add_argument('--scale_reward',type=float,default=1.)
    parser.add_argument('--soft_target_tau',type=float,default=1e-2)
    parser.add_argument('--max_path_length',type=int,default=1000)
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
