#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

#number of subprocess used for collecting data. This process has CPU-bound, the heaviest process is Atari frame
#preprocessing, therefore we need to set this to same as the number of our CPU.
PROCESSES_COUNT = 4
#NUM_ENVS is the number of environment for each subprocess to collect data, NUM_ENVS * PROCESSES_COUNT equal to total number
#of parallel environment existed.
NUM_ENVS = 15

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18

#we need environment construct function and a wrapper, we put episode total reward to the main training process.
def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))
TotalReward = collections.namedtuple('TotalReward', field_names='reward')

#this process will execute in sub-process, we will use mp.Process class to activate it in main process.We input our network,
#device for calculation, the queue for transfer data from sub-process to main process, the queue is for training. The queue
#for many-producers and one-consumer mode, which can include 2 type of product:
#TotalReward: a reward object with float point, indicate the non-discounted reward for 1 finished episode.
#ptan.experience.ExperienceFirstLast: it will wrap up the 1st state, action chosen, discounted reward, next state from
#REWARD_STEPS, which is the experience for training.
def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    
    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)

if __name__ == "__main__":
    #we call mp.set_start_method() to set the parallelization type. Python support multiple options, but Pytorch has
    #multithread limitation, spawn is the best option
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default="final", required=False, help="Name of the run")
    args, unknown = parser.parse_known_args()
    device = "cuda" if args.cuda else "cpu"
    writer = SummaryWriter(comment="-a3c-data_" + NAME + "_" + args.name)
    
    #we make our network and put on cuda and request for weighting share. Default CUDA tensor is shared, but in CPU mode,
    # we need to call share_memory
    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    
    #we create a queue to send the data, PROCESSES_COUNT stored the maximum size of the queue, when the queue is full, any
    #attempt to add new data to the queue will be prohibited, this can help us keep the on policy requirement. We use 
    #mp.Process to start the sub-process and store in queue, and we can close it later. After data_proc.start(), the
    #data_func will be activated by sub-processes.
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)
        
    batch = []
    step_idx = 0
    
    #we get the next item from the queue, process the TotalReward object, and send the reward to reward tracker.
    try:
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, step_idx):
                            break
                        continue
                    
                    #Because queue only has 2 object: TotalReward and experience transfer, after checking it isn't 
                    #TotalReward, it must be experience transfer, therefore we put the experience object into batch until
                    #the batch reaching the needed size.
                    step_idx += 1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue
                    
                    #after getting the required samples, we use unpack_batch() to change them as training data, because
                    #REWARD_STEPS=4, experience sample get 4 steps, therefore discount factor as GAMMA^4 for the last reward
                    states_v, actions_t, vals_ref_v = \
                    common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                    batch.clear()
                    
                    #Calculate Actor Critic loss, same as previous chapter: using online network to calculate policy and 
                    #value estimated logits, then calculate policy, value and entropy loss.
                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                    loss_policy_v = -log_prob_actions_v.mean()
                    
                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
                    
                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()
                    
                    #we send the calculated tensor to TensorBoard tracker, it will get the mean value and store it up.
                    tb_tracker.track("advantage", adv_v.cpu().numpy(), step_idx)
                    tb_tracker.track("values", value_v.cpu().detach().numpy(), step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v.cpu().numpy(), step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v.cpu().detach().numpy(), step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v.cpu().detach().numpy(), step_idx)
                    tb_tracker.track("loss_value", loss_value_v.cpu().detach().numpy(), step_idx)
                    tb_tracker.track("loss_total", loss_v.cpu().detach().numpy(), step_idx)
    
    #finally will activate in abnormal situation, such as ctrl+c, or "game solved" activated. We will terminate subprocess,
    #and combine sub-process, this is required to avoid omission of subprocess.
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()