
import numpy as np
import random
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from system import System

# """ MODEL PARAMETERS """

# def paramter_estimation(resolution=1):
#     """ Estimate the rate of Incoming BUY, SELL market orders (at highest data resolution)
#         Estimate the expected change in the mid price due to incoming BUY, SELL market orders
#         In each of the regimes.
#         """

# def compute_rate_of_market_orders(direction, regime):
#     """ Estimates rate of market orders per second for given regime and direction
        
#         :param direction: BUY / SELL market order
#         :param regime: a tuple of (spread regime, volume regime) 
#         :return: a float, rate of market orders per second
#         """

# def compute_estimated_mid_price_change(direction, regime, resolution=0.5):
#     """ Estimates rate of market orders per second for given regime and direction
        
#         :param direction: BUY / SELL market order
#         :param regime: a tuple of (spread regime, volume regime) 
#         :param resolution: the gap in seconds in which we measure the change in midprice
#         :return: a float, estimated change in mid price in ticks 
#         """

""" DEEP Q LEARNING """

# Essentially state space size is a dependent on the time remaining and quantity (inventory control states) and the 2 regime spaces (trading)
def train():
    batch_size = 256
    gamma = 0.999 # Update param for q-value updates
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10 # num episodes before updating target network with policy network weights
    memory_size = 100000
    lr = 0.001 # learning rate for policy network weight updates
    num_episodes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = TradingEnvManager(device)
    strategy = EpsilonGreedyStrategy(epse_start, eps_end, eps_decay)
    agent = Agent(Strtaegy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(em.num_states_available(), em.num_actions_available()).to_device(device)
    target_net = DQN(em.num_states_available(), em.num_actions_available()).to_device(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # sets to inference mode
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr) # Adam optimizer

    episode_profits = [] #TODO: replace this with more useful stats
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()
        profits = 0

        for timestep in count():
            """ 
            Agent taking actions in environment based off
            exploration and policy net in current episode
            """
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()

            memory.push(Experience(state, action, next_state, reward))
            state = next_state
            profits += reward

            """ 
            With these new experiences from exploration and action taking we update the policy
            network parameters so that we can better estimate the expected sum of future rewards
            in future greedy agent actions. We do this by sampling experiences randomly in batches,
            this breaks up any autocorrelation in the data and thus in the actions.
            """
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards # optimal q values

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad() # prevents gradient accumulation 
                loss.backward()
                optimizer.step()
            
            # i.e when investment horizon is reached
            if em.done:
                episode_profits.append(profits)
                compute_performance(episode_profits, 100)
                break
        
        # Synchronise polic and target networks every k steps
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    em.close()


""" DEEP Q LEARNING CLASSES """

class DQN(nn.Module):
    """ Defines a simple feed forward fully connected deep Q network

        :param state_space_size: The number of states in the model
        :param action_space_size: The number of possible actions
        """

    def __init(self, state_space_size, action_space_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features = state_space_size, out_features=24)
        self.fc2 = nn.Linear(in_features = 24, out_features = 32)
        self.out = nn.Linear(in_features = 32, out_features = action_space_size)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

""" 
Defines an experience detailing current state and action taken
and the subsequence state and reward
"""
Experience = namedtuple('Experience', ('state','action','next_state','reward'))

class ReplayMemory():
    """ 
    Stores a series of experiences up to some capcity which the DQN can sample from during training
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # If memory is full overwrite oldests memories first
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    """ 
    Computes the exploration rate for the strategy (whether the agent should explore next action randomly or be greedy and take the action with the highest q-value). The rate decays as the current step increases and the agent learns more about the environment
    """
    def __init__(self, start, end, decay):
        self.start, self, end, self.decay = start, end, decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)

class Agent():
    """ 
    Initialises our trading agent and utilises exploration or exploitation via DQN inference 
    to decide on the next action
    """
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
    
    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions) # explore
        else:
            with torch.no_grad(): # Turns off tracking of forward pass computations as we are just using forward pass for inference
                return policy_net(state).argmax(dim=1).to(device) # exploit

class TradingEnv():
    def __init__(self, ticker, inventory_risk, liquidation_penalty, n_imbalance_regimes=3, n_spread_regimes=2):
        """ Setup the trading environment.

        :param ticker: The symbol for the security
        :param inventory_risk: a multiplicative penalty for holding inventory at each timestep
        :param liquidation_penalty: an increasing function of abs(net_inventory) and bid ask spread which provides a 
        liquidation price penalty at end of investment horizon
        :return: returns an integer number of ticks
        """
        self.n_imbalance_regimes = n_imbalance_regimes
        self.n_imbalance_bins = np.linspace(-1,1,self.n_imbalance_regimes + 1)
        self.imbalance_depth = 1 # 1 => Best Bid / Ask

        self.n_spread_regimes = n_spread_regimes
        self.max_net_inventory = 10 # !0 units of underlying
        self.inventory_risk = inventory_risk
        self.liquidation_penalty = liquidation_penalty
        self.investment_horizon = 20 # 20 timesteps

        self.action_space_size = 4  # Buy LO, Sell LO, Both, Neither
        self.state_space_size = n_imbalance_regimes * n_spread_regimes + 2 # +2 for inventory state and remaining time state
        self.ticker = ticker
        self.system = System(poll=False, read_only=True)
        self.api = self.system.api

        self.times = self.get_available_times()
        self.reset()

    """ INDICATORS """
    def get_available_times(self, order_type="limit"):
        return self.api.get_available_times(self.ticker, order_type)

    def compute_bid_ask_spread(self, ticker, t):
        """ Computes bid ask spread at time t.

            :param ticker: The symbol for the security
            :param t: UNIX time in seconds 
            :return: returns an integer number of ticks
            """
        res = self.api.get_bid_ask_spread(ticker, t)
        self.best_bid = res['best_bid']
        self.best_ask = res['best_ask']

        return max(res['spread'], self.n_spread_regimes)

    def get_mid_price(self, ticker, t):
        res = self.api.get_bid_ask_spread(ticker, t)
        return (res['best_bid'] + res['best_ask']) / 2

    def compute_orderbook_imbalance(self, ticker, t, depth=1):
        """ Computes the volume imbalance at time t for a given order book depth. 
        A depth of 1 => Volume at best bid and best ask.

        :param ticker: The symbol for the security
        :param t: UNIX time in seconds 
        :param n_bins: The number of discrete regimes dividing [-1,1] space
        :return: value of imbalance regime between [1, n_bins]
        """
        imbalance = self.api.get_orderbook_imbalance(ticker, t, depth)
        return int(np.digitize(imbalance, self.n_imbalance_bins))

    def has_market_orders(self, ticker, t, t_next, depth=1):
        """ Computes whether market orders have been placed at the given depth in the book 
        during the period (t, t_next]. 

        :param ticker: The symbol for the security
        :param t: UNIX time in seconds 
        :param t_next: UNIX time in seconds t_next > t
        :param depth: The placement of limit orders from the best bid / ask at time t. (1  => Best bid / ask)
        :return: {'has_buy_mo': True / False, 'has_sell_mo: True / False}
        """

        res = self.api.get_market_orders(ticker, t, t_next, depth)
        return {'has_buy_mo': res['n_market_buys'] >= 1, 'has_sell_mo': res['n_market_sells'] >= 1}

    def reset(self):
        self.t_index = 0
        self.long_positions = 0 # Inventory Long
        self.short_positions = 0 # Inventory Short
        self.wealth = 0

    def next(self):
        "Yields the new state of the environment as 1d numpy array"
        spread_regime = self.compute_bid_ask_spread(self.ticker, self.times[self.t_index])
        imbalance_regime = self.compute_orderbook_imbalance(self.ticker, self.times[self.t_index], self.imbalance_depth)
        time_remaining = (self.investment_horizon - self.t_index) / self.investment_horizon # Normalize
        net_inventory = (self.long_positions - self.short_positions) / self.max_net_inventory # Normalize

        # we encode first the imbalance regime and then increment to correct spread slot 
        state_encoding = np.zeros((self.state_space_size,))
        
        # Zero indexing
        imbalance_regime_idx = (imbalance_regime - 1) * spread_regime
        spread_regime_idx = spread_regime - 1
        state_encoding[imbalance_regime_idx + spread_regime_idx] = 1
        state_encoding[-2] = net_inventory
        state_encoding[-1] = time_remaining

        return state_encoding

    def step(self, action):
        """Takes an action in the current environment (0: BUY LO,1: SELL LO,2: BOTH,3: NIETHER)
        :action : an integer speciying action to be taken
        :return : Tuple (new state, reward, done) where reward is the reward for current state
        """
        reward = 0
        done = False

        reward -= self.inventory_risk * (net_inventory**2)
        net_inventory = self.long_positions - self.short_positions

        if self.investment_horizon - self.t_index == 0:
            mid_price = self.best_bid + self.best_ask / 2
            sign = np.sign(net_inventory)
            liquidation = net_inventory * (mid_price + (-1*sign) * self.liquidation_penalty(net_inventory, self.best_ask - self.best_bid))
            self.wealth += liquidation

            reward += liquidation

            done = True
        else:
            res = self.has_market_orders(self.times[self.t_index], self.times[self.t_index + 1])

            exceeded_long_inventory = net_inventory >= self.max_net_inventory
            exceeded_short_inventory = net_inventory <= -1 * self.max_net_inventory

            if action == 0 and res['has_sell_mo'] and not exceeded_long_inventory:
                self.long_positions += 1
                reward -= self.best_bid
                self.wealth -= self.best_bid

            elif action == 1 and res['has_buy_mo'] and not exceeded_short_inventory:
                self.short_positions += 1
                reward += self.best_ask
                self.wealth += self.best_ask

            elif action == 2:
                if res['has_sell_mo'] and not exceeded_long_inventory:
                    self.long_positions += 1
                    reward -= self.best_bid
                    self.wealth -= self.best_bid
                if res['has_buy_mo'] and not exceeded_short_inventory:
                    self.short_positions += 1
                    reward += self.best_ask
                    self.wealth += self.best_ask
            elif action == 3:
                pass
        
        self.t_index += 1
        return (self.next(), reward, done)

class TradingEnvManager():
    """Abstracts the environment from the learning system via this manager class"""

    def __init__(self, device):
        self.device = device
        self.env = TradingEnv() #TODO
        self.env.reset()
        self.current_state = None # Tracks current state (None at initialisation)
        self.done = False # Tracks ending of episode

    def reset(self):
        self.env.reset()
        self.current_state = None

    def close():
        pass

    def next_state(): # Replaces render in tutorial, returns numpy array
        return self.env.next()
    
    def num_actions_available(self):
        return self.env.action_space_size

    def num_states_available(self):
        return self.env.state_space_size

    def take_action(self, action):
        _, reward, self.done = self.env.step(action.item())
        return torch.tensor([reward], device = self.device)

    def just_starting(self):
        return self.current_state is None

    def get_state(self):
        state = np.ascontiguousarray(self.next_state(), dtype=np.float32) # Ensures efficient storage in memormy
        state = torch.from_numpy(state) # Converts to torch
        state = state.unsqueeze(0).to(self.device) # add a batch dimension
        self.current_state = state

        return state 

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        """ For each state action pair, the policy net returns the q-values for the selected action"""
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_current(target_net, next_states):
        """ For each next state action pair, the target net returns the maximum q-value"""
        return target_net(next_states).max(dim=1)[0].detach()

        
def compute_performance(values, moving_avg_period):
    print("Moving Average Peformance: " % get_moving_average(30, values)[-1])
    print("Last episode performance: " % values[-1])

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg - values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):
    """Converts batch of experiences into tensors containing a batch of each experience property"""

    batch = Experience(*zip(*experiences))  # Takes a batch of experieces and transforms in experience of batches

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

        
#