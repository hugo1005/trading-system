import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from threading import Thread
from time import time

class SecuritiesDashboard:
    def __init__(self, securities, refresh_rate=100):
        """
        Renders a dashboard displaying the best bid, and best ask lines for each security
        :params securities: A list of securities objects
        :params refresh_rate: The time interval between refreshes of the the graph in ms 
        we default this to 400ms which is the fastest a human can detect
        """
        self.securities = securities
        self.refresh_rate = refresh_rate
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.midprices, = self.ax.plot([],[])
        self.bids, = self.ax.plot([],[])
        self.asks, = self.ax.plot([],[])

        # self.start()
        self.start()

    def animate(self, i):
        # We will test with one secuirty first
        security = self.securities['RITC']
        data = security.best_bid_ask
        self.midprices.set_data(data['midprice'].index.values,data['midprice'].values)
        self.bids.set_data(data['best_bid'].index.values,data['best_bid'].values)
        self.asks.set_data(data['best_ask'].index.values,data['best_ask'].values)

        self.ax.set_xlim((data['midprice'].index.min(), data['midprice'].index.max()))
        self.ax.set_ylim((data['best_bid'].values.min(), data['best_ask'].values.max()))
        # self.midprices.set_data(np.arange(0,i),np.arange(0,i))
        return self.midprices

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=self.refresh_rate)
        plt.show()

class SpreadDashboard:
    def __init__(self, leg_1, leg_2, cointegration_coefficient_fn, refresh_rate=100):
        """
        Renders a dashboard displaying the best bid, and best ask lines for each security
        :params securities: A list of securities objects
        :params refresh_rate: The time interval between refreshes of the the graph in ms 
        we default this to 400ms which is the fastest a human can detect
        """
        self.leg_1 = leg_1
        self.leg_2 = leg_2
        self.cointegration_coefficient_fn = cointegration_coefficient_fn
        self.refresh_rate = refresh_rate
        self.fig = plt.figure(figsize=(5,2))
        self.ax = plt.axes()
        self.spread, = self.ax.plot([],[])
        self.spread_data = {'timestamp':[], 'spread':[]}
        # self.start()
        self.start()
    
    def get_spread(self):
        """
        We assume equal weighting within each leg of the portfolio. Cointegration coefficent
        is applied to the close prices of leg_2. (Spread constructed as leg_2 - leg_1)
        :param leg_1: a list of ticker symbols consisting of the first spread component
        :param leg_2: a list of ticker symbols consisting of the second spread component
        :param cointegration_coeff: the latest value of the cointegration coefficient
        :return spread: the current asset spread
        """
        
        leg_1_prices = []
        leg_2_prices = []

        for security in self.leg_1:
            leg_1_prices.append(security.get_midprice())

        for security in self.leg_2:
            leg_2_prices.append(security.get_midprice())
        
        leg_1 = sum(leg_1_prices)
        leg_2 = sum(leg_2_prices) * self.cointegration_coefficient_fn()
        spread = leg_2-leg_1

        self.spread_data['timestamp'].append(pd.to_datetime(time(), unit='s'))
        self.spread_data['spread'].append(spread)

        return pd.DataFrame(self.spread_data).set_index('timestamp')

    def animate(self, i):
        # We will test with one secuirty first
        data = self.get_spread()
        self.spread.set_data(data.index.values, data['spread'].values)
        
        self.ax.set_xlim((data['spread'].index.min(), data['spread'].index.max()))
        self.ax.set_ylim((data['spread'].values.min(), data['spread'].values.max()))
        # self.midprices.set_data(np.arange(0,i),np.arange(0,i))
        return self.spread

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=self.refresh_rate)
        plt.show()

