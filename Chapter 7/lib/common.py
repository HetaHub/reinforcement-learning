import sys
import time
import numpy as np
import torch
import torch.nn as nn

# we store hyperparameters inside the dictionary object
HYPERPARAMS = {
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },

}

#we have function for transfering batch to numpy array. Each ExperienceSourceFirstLast has a converted namedtuple
#it included following columns:
#state: observation from environment
#action: action choose from agent(an integer)
#rewards: if we set steps_count = 1 for ExperienceSourceFirstLast, it will get instant reward, if we 
# set a larger number steps, it will be these steps total discounted reward
#last_state: if it correspond to last step in the environment, value will be None, otherwise it
# will be the last observation from the environment
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        #we append the initial state when at last state and don't have next state.
        #also we will mask the last state. Or we just calculate non-last state, but
        #it will be more complicated
        if exp.last_state is None:
            last_states.append(state)
        #the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32),\
        np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)
        
#we start from first state to get q-value, and use Bellman to calculate the value
#Same as chapter 6, the loss is the Mean Square Error of these 2 values.
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
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

    #we apply next state observation to target network and calculate the largest q-value for action dimension(1).
    #max() will return the largest value and the index at the same time, which is max and argmax, we use the value here
    #only, therefore we get array[0]
    next_state_values = tgt_net(next_states_v).max(1)[0]

    #for the last step q-values, we set it as 0.0 for convergence because there are no next step to collect reward
    #action value won't have next state discounted reward. If we don't set this, it won't converge.
    next_state_values[done_mask] = 0.0

    #we calculate Bellman approximation and Mean Square Loss here
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

#epsilon tracker accept EpsilonGreedyActionSelector object and hyperparameters
class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)
    #frame() method use epsilon method to decrease the value, for the first
    #epsilon_frames step, linearly decrease epsilon value, then let it unchanged
    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = max(self.epsilon_final,
        self.epsilon_start - frame / self.epsilon_frames)

#reward tracker will return the total reward of the episode when end
#it will calculate the mean reward after last step and report to TensorBoard
#also it will check whether the problem solved, it use frame number processed
#for each second to determine the speed
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    #we use context manager to implement the code, reward() will be 
    #called at every end of episode
    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" %epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" %(
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False