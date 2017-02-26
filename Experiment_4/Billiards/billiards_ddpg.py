

from algos.ddpg import DDPG
from billiards_nn_policy import Conv_FeedForwardPolicy
from billiards_qfunction import Conv_FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from billiards_2D import  Billiards_2D
from planarspace import PlanarSpace
from sandbox.rocky.tf.envs.base import TfEnv
import IPython
#from model_classes import Conv_FeedForwardCritic,Conv_FeedForwardPolicy

def run_task(*_):
    env = Billiards_2D()
    es = OUStrategy(env_spec=env.spec)
    qf = Conv_FeedForwardCritic(
        name_or_scope = "critic",
        env_spec=env.spec,
    )
    policy = Conv_FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        eval_samples = 1000,
        min_pool_size = 1000,
        epoch_length = 1000,
	n_epochs = 200,
        n_updates_per_time_step = 1,
        max_path_length = 100,
	batch_size = 32,
        replay_pool_size = 100000,
        discount = 0.99,
        summary_dir = "/home/shay93/Motor-Represenations/Experiment_4/Billiards/tensorflow_summaries",
        
    )
    #IPython.embed()
    algorithm.train()



if __name__ == "__main__":
    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-billiards",
        seed=2,
  ) 
