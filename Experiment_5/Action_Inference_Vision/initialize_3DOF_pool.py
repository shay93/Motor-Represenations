from misc.simple_replay_pool import SimpleReplayPool
import pickle

#initialize a simple replay pool and populate it with the
#inferred 3DOF states and actions

#load inferred 3DOF states and actions
pool_3DOF = SimpleReplayPool()

with open(data_dir + "infer_obs.npy","wb") as f:
    inferred_obs = pickle.load(f)

with open(data_dir + "infer_actions.npy","wb") as f:
    inferred_actions = pickle.load(f)

with open(data_dir + "infer_reward.npy","rb") as f:
    infer_reward = pickle.load(f)

#loop over the sequences and episodes and add samples
#TODO make sure the states and actions are in the correct form
for seq in range(num_sequences):
    for tstep in range(episode_length):
        observation = inferred_obs[seq,tstep,...]
        action = inferred_actions[seq,tstep,...]
        reward = infer_reward[seq,tstep,...]
        final_state = False
        pool_3DOF.add_sample(observation,action,reward,final_state)

#TODO how to save pool and pass to online algorithm

