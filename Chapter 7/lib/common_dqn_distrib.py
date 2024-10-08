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
        self.epsilon_greedy_selector.epsilon = max(self.epsilon_final,\
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

#we construct distribution projection array, input is shape of batch_size, n_atoms batch, reward array,
#is_done tag, also with 4 hyperparameters: Vmin, Vmax, n_atoms and gamma. Variable delta_z is the width
#of the atoms in the value range(we divide the value range from Vmax = -10 to Vmax = 10 as 51 parts and
# call each as atoms)
def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    #we loop for every atom in the distribution to calculate Bellman equation projection to the atom,
    #and the boundary of the value, for example, 1st index is 0, Vmin = -10, to add +1 reward to the
    #sample, next will be -10 * 0.99 + 1 = -8.9 , it will move right with discount value gamma = 0.99
    #if value exceed the boundary of Vmin and Vmax, we set it to the boundary

    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))

        #we calculate the atom number of the sample projection, if the value between 2 atoms, we will
        #expand the value, if it falls apart on some atom, we will add the source distribution to the
        #atom
        b_j = (tz_j - Vmin) / delta_z

        #below processed the situation: projected atom fall on target atom, otherwise, it won't be
        #integer, and can't be variable l and u, l and u corresponding to the lower and upper position
        #of the atom index
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]

        #if projection fall between atoms, we expand the source probability
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    #for the last transition, we don't need to care next distribution and it will only have
    # 1 corresponding probability, but if it fall between atoms, we still need to care the 
    # distribution, if it is the last transition, we will set the distribution to 0, and 
    # calculate the projection
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


