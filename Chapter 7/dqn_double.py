#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

#we start from first state to get q-value, and use Bellman to calculate the value
#Same as chapter 6, the loss is the Mean Square Error of these 2 values.
def calc_loss(batch, net, tgt_net, gamma, device="cpu", double=True):
    #we can activate or deactivate DDQN method for action
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    #we pack batch data as numpy array, if parameters need CUDA device, we add to GPU.
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    #we put the observation to the model and use gather to get the q-value. First parameterm is the parameter position
    #we want to operate, 1 correspond to action parameter, unsqueeze will insert a new dimension,here at final position,
    #the result is the action taken
    #gather result is differentiable, it record the last loss gradient
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        #we apply next state observation to target network and calculate the largest q-value for action dimension(1).
        #max() will return the largest value and the index at the same time, which is max and argmax, we use the value here
        #only, therefore we get array[0]
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    #for the last step q-values, we set it as 0.0 for convergence because there are no next step to collect reward
    #action value won't have next state discounted reward. If we don't set this, it won't converge.    
    next_state_values[done_mask] = 0.0

    #we calculate Bellman approximation and Mean Square Loss here
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

#we break the states to small pieces and pass to network to calculate action value, we choose the maximum value and
#calculate the mean value of these maximum value
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


if __name__ == "__main__":
    #input hyperparameters, check CUDA available, create environment,then we use PTAN DQN wrapper to wrap up the environment
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    #we can activate Double DQN at following line.
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    #we make a writer for the environment and action dimension
    writer = SummaryWriter(comment="-" + params['run_name'] + "-double=" + str(args.double))
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    #the wrapper below can create a copy of DQN network, which is target network, and constantly synchronize with online
    #network
    tgt_net = ptan.agent.TargetNet(net)
    #we create agent to change observation to action value, we also need action selector to choose the action we use
    #We use epsilon greedy method as action selector here
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    #experience source is from one step ExperienceSourceFirstLast and replay buffer, it will store fixed step transitions
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    #create optimizer and frame counter
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    #we initialize evaluation state value
    eval_states = None

    #reward tracker will report mean reward when episode end, and increase frame counter by 1, also getting a transition
    #from frame buffer.
    #buffer.populate(1) will activate following actions:
    #ExperienceReplayBuffer will request for next transition from experience source.
    #Experience source will send the observation to agent to get the action
    #Action selector which use epsilon greedy method will choose an action based on greedy or random
    #Action will be return to experience source and input to the environment to get reward and next observation, 
    # current observation, action, reward, next observation will be stored into replay buffer
    #transfer information will be stored in replay buffer, and oldest observation will be dropped
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            #check undiscounted reward list after finishing an episode, and send to reward tracker to record the data
            #Maybe it just play one step or didn't have finished episode, if it returns true, it means the mean reward
            #reached the reward boundary and we can break and stop training
            new_rewards = exp_source.pop_total_rewards()
            #we check buffer has cached enough data to start training or not. If not, we wait for more data.
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break
            #we check buffer has cached enough data to start training or not. If not, we wait for more data.
            if len(buffer) < params['replay_initial']:
                continue
            #we construct states for evaluation, and set STATES_TO_EVALUATE to 1000 to evaluate 1000 states at once.
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            #here we use Stochastic Gradient Descent(SGD) to calculate loss, zero the gradient,batch from the replay buffer
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            #We use the calc_loss_dqn in this file instead of the one in common.py
            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device,
                               double=args.double)
            loss_v.backward()
            optimizer.step()

            #synchronize the target network with the online network constantly
            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            #for every EVAL_EVERY_FRAME(100) frames, we calculate the mean value of states and write to TensorBoard
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)
