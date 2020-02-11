import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from threading import Thread
from Options_Execution import OptionsExecution

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

        self.midprices.set_data(data['midprice'].index.values,data['midprice'].values)
        self.ax.set_xlim((data['midprice'].index.min(), data['midprice'].index.max()))
        self.ax.set_ylim((data['best_bid'].values.min(), data['best_ask'].values.max()))
        # self.midprices.set_data(np.arange(0,i),np.arange(0,i))
        return self.midprices

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=self.refresh_rate)
        plt.show()


     def __enter__(self):
        self.main_thread = Thread(target=self.launch_all_threads, name='Main Algo Thread')
        self.main_thread.start()
        
        sleep(5)
        self.securities_dashboard = SecuritiesDashboard(self.securities)


class OptionsDashboard:
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
        self.call_skew_1, = self.ax.plot([],[])
        self.call_skew_2, = self.ax.plot([],[])
        self.put_skew_1, = self.ax.plot([],[])
        self.put_skew_2, = self.ax.plot([],[]))

        # self.start()
        self.start()

    def animate(self, i):
        self.call_skew_1.set_data(call_skew_1.index.values,call_skew_1.values)
        self.call_skew_2.set_data(call_skew_2.index.values,call_skew_2.values)
        self.put_skew_1.set_data(put_skew_1.index.values,put_skew_1.values)
        self.put_skew_2.set_data(put_skew_2.index.values,put_skew_2.values)


        self.ax.set_xlim((call_skew_1.index.min(), call_skew_1.index.max()))
        self.ax.set_ylim(0,0.5)
        # self.midprices.set_data(np.arange(0,i),np.arange(0,i))
        return self.call_skew_1, self.call_skew_2, self.put_skew_1, self.put_skew_2

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=self.refresh_rate)
        plt.show()


     def __enter__(self):
        self.main_thread = Thread(target=self.launch_all_threads, name='Main Algo Thread')
        self.main_thread.start()
        
        sleep(5)
        self.securities_dashboard = SecuritiesDashboard(self.securities)

