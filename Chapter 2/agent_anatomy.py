import random

class Environment:
    #initialize steps for environment
    def __init__(self):
        self.steps_left = 10
        
    #get observation return environment observation to agent    
    def get_observation(self):
        return [0.0, 0.0, 0.0]
    
    #get action set for action available(not all actions available in any states!)
    def get_actions(self):
        return [0, 1]
    
    #Use is_done to check when is the episode ended, it can be finite or infinite
    def is_done(self):
        return self.steps_left == 0
    
    #Process agent action and return the reward, update steps
    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        
        self.steps_left -= 1
        return random.random()
    
    
    
class Agent:
    #initialize reward
    def __init__(self):
        self.total_reward = 0.0
        
    #record the reward agent get in episodes
    #observe environment, decide action performed, feed the action to environment, get reward
    #as it is basic version here, it will ignore the observation and perform random action
    def step(self, env):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward
        

if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    
    while not env.is_done():
        agent.step(env)
        
    print("Total reward got: %.4f" %agent.total_reward)