:x

from algos.ddpg import DDPG
from base_nn_policy import Conv_FeedForwardPolicy
from qfunctions.nn_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from planar2d import  env_2DOF_arm
from planarspace import PlanarSpace
from sandbox.rocky.tf.envs.base import TfEnv
import IPython

def run_task(*_):
    env = env_2DOF_arm()
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
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
        batch_size = 500,
        n_epochs = 1000,
        min_pool_size = 1000,
        replay_pool_size = 200000,
    )
    algorithm.train()



if __name__ == "__main__":
    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-planar2D",
        seed=2,
  ) 
