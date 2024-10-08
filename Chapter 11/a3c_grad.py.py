#!/usr/bin/env python3
import gym
import ptan
import argparse
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

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 15

#BATCH_SIZE changed to GRAD_BATCH and TRAIN_BATCH, GRAD_BATCH is the batch size for sub-process to determine gradient from
#the loss
#TRAIN_BATCH defined using how many subprocess gradient batch to combine to calculate SGD loss. Therefore, for each
#optimization step, we use TRAIN_BATCH * BATCH_SIZE samples for training. Calculating loss and back-propagation has a heavy
#workload, therefore we can use a larger GRAD_BATCH for higher efficiency, we need a smaller TRAIN_BATCH value to let the
#network keep it as on-policy.
GRAD_BATCH = 64
TRAIN_BATCH = 2
ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18

def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

#code in sub-process becomes much complicated, but main process code becomes easier. We send the code below to function:
# proc_name:  used for creating TensorBoard output environment, every subprocess will output to its own TensorBoard dataset
# net: shared neural network
# device: device for calculation(cuda / cpu)
# train_queue: a queue for transfering the calculated gradient to main process.
#The code in our sub-process is similar to the code for main progress in data parallelization version, this is because we
#put the mission into the sub-process instead of main process. But we need the optimizer to collect gradient instead of
#renewing the network, then add the gradient to the queue, other code are almost the same as before.
def grads_func(proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    
    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)
    
    #we collect batch transfer and get the reward when episode ends.
    with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break
                    
                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue
                
                #we use training data to calculate loss, and do back-propagation, then we store the network parameters
                #gradient into Tensor.grad, this can done without synchronize with other workers, because our network
                #parameters are shared with each other, but gradient are set and keeping by each individual sub-process
                states_v, actions_t, vals_ref_v = \
                common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                batch.clear()
                
                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                log_prob_actions_v = adv_v * log_prob_v[range(GRAD_BATCH), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()
                
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
                loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                loss_v.backward()
                
                #we send the value to TensorBoard in training.
                tb_tracker.track("advantage", adv_v.cpu().numpy(), frame_idx)
                tb_tracker.track("values", value_v.cpu().detach().numpy(), frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v.cpu().numpy(), frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v.cpu().detach().numpy(), frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v.cpu().detach().numpy(), frame_idx)
                tb_tracker.track("loss_value", loss_value_v.cpu().detach().numpy(), frame_idx)
                tb_tracker.track("loss_total", loss_v.cpu().detach().numpy(), frame_idx)
                
                #we clip gradient when loop ends, and put the gradient into a buffer to avoid the value being destroyed
                #in the next iteration.
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                grads = [param.grad.data.cpu().numpy() if param.grad is not None else None for param in net.parameters()]
                train_queue.put(grads)
        
        #we put None into the queue to indicate the sub-process has reached "game-solved" state, we should stop training.
        train_queue.put(None)

if __name__ == "__main__":
    #create network and share weighting in main process
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default="final", required=False, help="Name of the run")
    args, unknown = parser.parse_known_args()
    device = "cuda" if args.cuda else "cpu"
    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()
    
    #create data transfer queue and create sub-process
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = "-a3c-grad_" + NAME + "_" + args.name + "#%d" % proc_idx
        data_proc = mp.Process(target=grads_func, args=(proc_name, net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)
    
    #Main difference between data parallelization and gradient parallelization version is in the training loop, the training
    #loop here is much simplier, it is because sub-process finished all calculation for us.We need to process the situation:
    #when one sub-process reached the required mean-reward, it need to stop training, therefore, we exit the loop when
    #reaching the required reward.
    batch = []
    step_idx = 0
    grad_buffer = None
    
    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            
            #to balance the gradient from different sub-process, for every TRAIN_BATCH gradient, we call optimizer step()
            #function, for the step between, we add the gradient together.
            step_idx += 1
            
            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad
            
            #if we get enough data with TRAIN_BATCH, we change the total gradient type to Pytorch FloatTensor, and change
            #the network parameter grad to this total gradient value. Then we call optimizer step(), to accumulate gradient
            #and renew network parameters
            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    grad_v = torch.FloatTensor(grad).to(device)
                    param.grad = grad_v
                    
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None
    
    #when exiting loop, such as Ctrl + C, we need to stop all sub-process to prevent zombie process occupying our GPU
    #resources.
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()