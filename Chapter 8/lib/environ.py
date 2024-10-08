from re import S
import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


#we encode all possible actions with enumerator column
class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class StocksEnv(gym.Env):
    #this metadata is used for fullfilling gym.Env requirements compactibility
    #but we don't have render function, so we can ignore it
    metadata = {'render.nodes': ['human']}

    #we have 2 object construction method, 1st is using data directory as parameter
    #to call from_dir, it will load all csv files and construct environment
    #directly construct object, need prices dictionary as input, data.Prices has 5 columns,
    #which is open,high,low,close,volume, we can call data.load_relative to generate the objects
    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)

    #prices include one or more dictionary, search key is the name, data.Prices object is value
    #bars_count: the number of bars passed when observation, default=10
    #commission: percentage commission need to pay, default=0.1%
    #reset_on_close: when agent need to sell the stock, we end the episode, otherwise it will
    #  end after 1 year
    #state_1d: observation data to pass to agent, if true, kbar is 2 dimension, and use column to 
    # construct, 1st row highest, 2nd lowest, 3rd close price... This is suitable for 1d convolution,
    # this is similar as Atari 2d game image has different RGB pixel in a row. If we set to false,
    # we has single data array, each kbar put together, this is good for fully connected network
    #random_ofs_on_reset: if true, we will set a time bias when reset, otherwise start at beginning.
    #reward_on_close: if True, will get the reward after selling, otherwise, we will give each kbar
    # a small reward
    #volumes: show volumes data. Default as false.
    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, commission=DEFAULT_COMMISSION_PERC,\
     reset_on_close=True, state_1d=False, random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        #State and State1D to prepare observation, buying state and reward, we construct state object
        # action space and observation space
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close,\
             reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,\
             reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,\
         shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    # make selection of the instrument and it's offset. Then reset the state.
    #define the environment reset function, we change the time stamp randomly,
    #and change the starting bias, then pass price and bias to the state object,
    #and use the encode() function to get the observation
    def reset(self):
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    #agent choose action and return next observation, reward and done flag.
    #we will implement the functions in state class, therefore, this is a wrapper.
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    #gym.Env API can define render() method handler, it need to translate current
    #state. This method used to check the environment state and debug or track
    #agent action. For example, environment can render the price as chart to show
    #what the agent can see. However, our environment don't support rendering so
    #nothing will happen here.
    def render(self, mode='human', close=False):
        pass
    
    #close() is to destruct the environment and release resources to the object.
    def close(self):
        pass

    #when we create multiple environment, we can use the same random seed.
    #This is not much related in one DQN object but very useful in A3C.
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

class State:
    #check and record object.
    def __init__(self, bars_count, commission_perc, reset_on_close,\
         reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    #reset() will save stored price data and initial bias value
    #we didn't buy any stock, so have_position and open_price both are 0.0
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    #State show as numpy array and return array shape, it will encode as vector
    #and include the price, volume and buying price and profit
    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3 * self.bars_count + 1 + 1, )

    #this function put biased price into numpy array, which is observation for
    #agent
    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    #this will calculate the close price for current bar, it is relative price
    #the highest price relative to the open price, lowest price relative to open
    #price, close price relative to open price
    def _cur_close(self):
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    #this play one step in our environment, it will return the percentage reward,
    #also the done tag
    def step(self, action):
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()

        #if agent decided to buy stock, we change state and pay commission fee,
        #we assume we will use the close price of the current bar to execute trading,
        #but in real life, we can use different price to trade, which is 
        #price slippage
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc

        #if we have position and agent want to sell, we pay commission fee,
        #if reset_on_close activated, we will set the episode flag to done
        #and calculate final reward, and change our states.
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        #we change current bias and give the last kbar reward.
        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done

class State1D(State):
    #the price will be encoded as 2D matrix which suitable for 1D convolution
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    #according to current bias, volume and have_position to encode the price
    #in matrix.
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count - 1
        res[0] = self._prices.high[self._offset - ofs:self._offset + 1]
        res[1] = self._prices.low[self._offset - ofs:self._offset + 1]
        res[2] = self._prices.close[self._offset - ofs:self._offset + 1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset - ofs:self._offset + 1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst + 1] = (self._cur_close() - self.open_price) / self.open_price
        return res