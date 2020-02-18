from sources import API
from security import Security
from execution import TradingTick, ExecutionManager

from time import sleep, time
from queue import Queue
from threading import Thread
import requests
import itertools

import scipy.stats as st
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys
import math

import os
import warnings
warnings.filterwarnings('ignore')

clear = lambda: os.system('cls')

API_CONFIG = './configs/api_config.json'
SQL_CONFIG = './configs/sql_config.txt'
DB_PATH = './datasets/hftoolkit_sqlite.db'

CASE_MAX_TICKS = 600
DAY_TICKS = 24
DAYS_PER_WEEK = 5

# TODO: Refactor this to poll for varying cost from API
RIG_COST_PER_DAY = 32000
PRODUCTION_TICKS = 6
CRUDE_PER_RCA = 2

FAIR_PRICE_RCA = 50
FAIR_PRICE_CL = 30

# TODO: Correct these
LIMIT_CL = 20
LIMIT_RCA = 20
CONTRACT_SIZE_CL = 1000
CONTRACT_SIZE_RCA = 1000

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BPTradingManager():
    def __init__(self, tickers, poll_delay=0.005):
        """" 
        Initialises trading manager
        param tickers: list of tickers of securties
        param risk_aversion: probability of a loss exceeding Z(risk_aversion) standard deviations of long term volatility
        """
        print("[TradingManager] Configuring...")

        self.tickers = tickers
        self.securities = {}

        self.poll_delay = poll_delay

        # Request Securities price history from server
        self.api = API(API_CONFIG, DB_PATH, SQL_CONFIG, use_websocket=False)
        self.endpoint, self.headers = self.api.get_source_config('RIT')

       
    """ ------- Trading Start / Stop -------- """

    def __enter__(self):
        for ticker in self.tickers:
            sec = Security(ticker, self.api, is_currency=ticker=='USD')
            self.securities[ticker] = sec
        
        self.execution_manager = ExecutionManager(self.api, self.tickers, self.securities)

        self.poll_securities = Thread(target=self.poll_securities)
        self.poll_securities.start()
        self.execution_manager.start()

        sleep(0.3) # Lets securities start polling

        """ Lets fix securities first!"""
        self.crude_production_mointor = Thread(target=self.monitor_production, name="Tender Watcher")
        self.crude_production_mointor.start()

    def __exit__(self, t, value, traceback):
        self.enable_market_maker = False
        self.accept_tender_orders = False
        self.enable_arbitrage = False
        print("-------------- Trading Period Finished! -----------------")
    
    """ Polling Securities """
    def poll_securities(self):
        print("[PollingSecurities] Started...")
        for t in TradingTick(295, self.api):
            for ticker in self.tickers:
                self.securities[ticker].poll()
            
            sleep(self.poll_delay)

    def monitor_production(self):
        profit_per_rig = 0

        for t in TradingTick(CASE_MAX_TICKS,  self.api):
            if t % DAY_TICKS * DAYS_PER_WEEK == 0:
                clear()
                print("----------- Next Week Started ! --------------")
                CL_vwap = self.securities['CL'].get_vwap()
                RCA_vwap = self.securities['CL'].get_vwap()
                RCA_best_bid, _ = self.securities['RCA'].get_best_bid_ask()

                # TODO: Poll server for this
                rebate_quantity = 20
                
                production_runs_per_day = DAY_TICKS / PRODUCTION_TICKS
                rebate_value = rebate_quantity * RCA_best_bid
                
                rca_used = 2
                crude_produced = rca_used * CRUDE_PER_RCA
                crude_revenue = CL_vwap * crude_produced * CONTRACT_SIZE_CL
                rca_costs = RCA_vwap * rca_used * CONTRACT_SIZE_RCA

                profit_per_rig =  DAYS_PER_WEEK * (production_runs_per_day * (crude_revenue - rca_costs) - RIG_COST_PER_DAY) + rebate_value

                if profit_per_rig > 0:
                    print("\033[92m [PROD MONITOR] Profitable to Produce: $%.2f per rig" % profit_per_rig)
                else:
                    print("\033[93m [PROD MONITOR] Don't Produce this week: $%.2f per rig" % profit_per_rig)

            if t % DAY_TICKS == 0:
                if profit_per_rig > 0:
                    print("\033[92m [PROD MONITOR] Hire Rig (tick: %s)" % t)

            if t % PRODUCTION_TICKS == 0:
                RCA_best_bid, _ = self.securities['RCA'].get_best_bid_ask()

                if profit_per_rig > 0:
                    print("\033[94m [PROD MONITOR] Extract Oil (tick: %s)" % t)
                    
                    CL_best_bid, _ = self.securities['CL'].get_best_bid_ask()
                    CL_vwap = self.securities['CL'].get_vwap()

                    if FAIR_PRICE_CL > CL_best_bid:
                        print("\033[94m [PROD MONITOR] Stockpile Oil, its undervalued (tick: %s, price: %.2f)" % (t, CL_best_bid))
                    elif CL_vwap > CL_best_bid:
                        print("\033[94m [PROD MONITOR] Stockpile Oil, VWAP is higher (VWAP: %.2f, price: %.2f)" % (CL_vwap, CL_best_bid))
                    else:
                        print("\033[93m [PROD MONITOR] Sell Oil (VWAP: %.2f, price: %.2f)" % (CL_vwap, CL_best_bid))

                if FAIR_PRICE_RCA > RCA_best_bid:
                    print("\033[93m [PROD MONITOR] Buy RCA, its undervalued (tick: %s, price: %.2f)" % (t, RCA_best_bid))

                

def main():
    print('reached')
    with BPTradingManager(['CL','RCA']) as tm:
        
        for t in TradingTick(CASE_MAX_TICKS,  API(API_CONFIG, DB_PATH, SQL_CONFIG, use_websocket=False)):
            pass

if __name__ == '__main__':
    # install_thread_excepthook()
    main()
        