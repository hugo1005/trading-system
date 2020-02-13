from execution import TradingTick, ExecutionManager, OptionsExecutionManager
from security import Security, Options
from sources import API
import numpy as np
import pandas as pd
from time import sleep

#Deal with the API for sourcing data and sending orders
#my API key is in api.JSON


api = API('./configs/api_config.JSON')

tickers = {'ticker_C_1':['RTM1C45','RTM1C46','RTM1C47','RTM1C48','RTM1C49','RTM1C50','RTM1C51','RTM1C52','RTM1C53','RTM1C54'],
    'ticker_C_2':['RTM2C45','RTM2C46','RTM2C47','RTM2C48','RTM2C49','RTM2C50','RTM2C51','RTM2C52','RTM2C53','RTM2C54'],
    'ticker_P_1':['RTM1P45','RTM1P46','RTM1P47','RTM1P48','RTM1P49','RTM1P50','RTM1P51','RTM1P52','RTM1P53','RTM1P54'],
    'ticker_P_2':['RTM2P45','RTM2P46','RTM2P47','RTM2P48','RTM2P49','RTM2P50','RTM2P51','RTM2P52','RTM2P53','RTM2P54']}

class OptionsTradingManager:

    def __init__(self,api,r=0):
        self.api =api
        self.r=r
        
        self.options = {}

        self.tickers = pd.DataFrame(data=tickers)

        tickers_reshaped = np.array(self.tickers.values).reshape(-1,)

        for ticker in tickers_reshaped:
            sec = Options(ticker, self.api, is_currency=False)
            sec.start()
            self.options[ticker] = sec
        
        self.options_execution_manager = OptionsExecutionManager(self.api, tickers_reshaped, self.options)

        self.options_execution_manager.start()

    def __enter__(self):
        # TODO: Fix time duration its not 295 seconds!
        sleep(5)

        for t in TradingTick(600, self.api): 
            if t >= 295:
                self.tickers = {'ticker_C_2':self.tickers['ticker_C_2'], 'ticker_P_2':self.tickers['ticker_P_2']}

            self.sigma = self.options_execution_manager.vol_forecast()

            self.specific_option_misprice(self.tickers)
            self.imp_vol_mp(self.tickers,S)
            self.termstructure(self.tickers)
            sleep(0.2)

    def __exit__(self, t, value, traceback):
        print("-------------- Trading Period Finished! -----------------")

    "___________________Term Structure Trading Algorithm________________________"

    def termstructure(self,tickers):
        for i in range(len(tickers)-1):

            C_1 = self.options[tickers['ticker_C_1'][i]].get_midprice()
            P_1 = self.options[tickers['ticker_P_1'][i]].get_midprice()
            C_2 = self.options[tickers['ticker_C_2'][i]].get_midprice()
            P_2 = self.options[tickers['ticker_P_2'][i]].get_midprice()

            S, K_C_1, T_C_1, option_C_1 = self.options.option_disect(tickers['ticker_C_1'][i])
            S, K_P_1, T_P_1, option_P_1 = self.options.option_disect(tickers['ticker_P_1'][i])
            S, K_C_2, T_C_2, option_C_2 = self.options.option_disect(tickers['ticker_C_2'][i])
            S, K_P_2, T_P_2, option_P_2 = self.options.option_disect(tickers['ticker_P_2'][i])

            S = self.options['RTM'].get_midprice()

            C_1_vol = self.options.nr_imp_vol(S, K_C_1, T_C_1, C_1, r, self.sigma, option = 'C')
            P_1_vol = self.options.nr_imp_vol(S, K_P_1, T_P_1, P_1, r, self.sigma, option = 'P')
            C_2_vol = self.options.nr_imp_vol(S, K_C_2, T_C_2, C_2, r, self.sigma, option = 'C')
            P_2_vol = self.options.nr_imp_vol(S, K_P_2, T_P_2, P_2, r, self.sigma, option = 'P')
            
            orders = []

            #if strike >= 100% term structure should be normal
            if K_C_1/S >= 1:
                
                #if inverted call term structure
                if C_2_vol < C_1_vol:
                    print("At Strike",K_C_1,"Buy 2M and Sell 1M Call")

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET','BUY', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_C_2,T_C_2,r,self.sigma,'C','SELL',100))

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','SELL', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_C_1,T_C_1,r,self.sigma,'C','BUY',100))
                
                #if inverted put term structure
                if P_2_vol < P_1_vol:
                    print("At Strike",K_C_1,"Buy 2M and Sell 1M Put")

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET','BUY', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_P_2,T_P_2,r,self.sigma,'P','BUY',100))

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','SELL', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_P_1,T_P_1,r,self.sigma,'P','SELL',100))

            
            #if strike < 100% term structure should be inverted
            if K_C_1/S < 1:
                
                #if normal call term structure
                if C_2_vol >= C_1_vol:
                    print("At Strike",K_C_1,"Buy 1M and Sell 2M Call")

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','BUY', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_C_1,T_C_1,r,self.sigma,'C','SELL',100))

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET','SELL', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_C_2,T_C_2,r,self.sigma,'C','BUY',100))
                
                #if normal put term structure
                if P_2_vol >= P_1_vol:
                    print("At Strike",K_C_1,"Buy 1M and Sell 2M Put")

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','BUY', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_P_1,T_P_1,r,self.sigma,'P','BUY',100))

                    orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET','SELL', 100))
                    orders.append(self.options_execution_manager.delta_hedge(S,K_P_2,T_P_2,r,self.sigma,'P','SELL',100))

        oids = self.options_execution_manager.execute_orders([orders], 'OPTION')

    "___________________Skew Trading Algorithm________________________"

    def imp_vol_mp(self,tickers,S):

        orders = []

        call_skew_1 = []
        call_skew_2 = []
        put_skew_1 = []
        put_skew_2 = []

        for i in range(len(tickers)-1):

            C_1 = self.options[tickers['ticker_C_1'][i]].get_midprice()
            P_1 = self.options[tickers['ticker_P_1'][i]].get_midprice()
            C_2 = self.options[tickers['ticker_C_2'][i]].get_midprice()
            P_2 = self.options[tickers['ticker_P_2'][i]].get_midprice()

            S, K_C_1, T_C_1, option_C_1 = self.options.option_disect(tickers['ticker_C_1'][i])
            S, K_P_1, T_P_1, option_P_1 = self.options.option_disect(tickers['ticker_P_1'][i])
            S, K_C_2, T_C_2, option_C_2 = self.options.option_disect(tickers['ticker_C_2'][i])
            S, K_P_2, T_P_2, option_P_2 = self.options.option_disect(tickers['ticker_P_2'][i])

            S = self.options['RTM'].get_midprice()

            C_1_vol = self.options.nr_imp_vol(S, K_C_1, T_C_1, C_1, r, self.sigma, option = 'C')
            P_1_vol = self.options.nr_imp_vol(S, K_P_1, T_P_1, P_1, r, self.sigma, option = 'P')
            C_2_vol = self.options.nr_imp_vol(S, K_C_2, T_C_2, C_2, r, self.sigma, option = 'C')
            P_2_vol = self.options.nr_imp_vol(S, K_P_2, T_P_2, P_2, r, self.sigma, option = 'P')


            C_1_vol = self.options.nr_imp_vol(S, K_C_1, T_C_1, C_1, r, self.sigma, option = 'C')
            P_1_vol = self.options.nr_imp_vol(S, K_P_1, T_P_1, P_1, r, self.sigma, option = 'P')
            C_2_vol = self.options.nr_imp_vol(S, K_C_2, T_C_2, C_2, r, self.sigma, option = 'C')
            P_2_vol = self.options.nr_imp_vol(S, K_P_2, T_P_2, P_2, r, self.sigma, option = 'P')

            call_skew_1.append(C_1_vol)
            put_skew_1.append(P_1_vol)
            call_skew_2.append(C_2_vol)
            put_skew_2.append(P_2_vol)

            if C_1_vol < self.sigma:
                print("At Strike",K_C_1,"Buy 1M Call") #Buy as the implied vol is priced below what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_C_1,T_C_1,r,self.sigma,'C','SELL',100))
            elif C_1_vol > self.sigma:
                print("At Strike",K_C_1,"Sell 1M Call") #Sell as the implied vol is priced above what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_C_1,T_C_1,r,self.sigma,'C','BUY',100))
            else:
                print("At Strike",K_C_1,"The Call volatility is priced appropriately")

            if P_1_vol < self.sigma:
                print("At Strike",K_P_1,"Buy 1M Put") #Buy as the implied vol is priced below what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_P_1,T_P_1,r,self.sigma,'P','BUY',100))
            elif P_1_vol > self.sigma:
                print("At Strike",K_P_1,"Sell 1M Put") #Sell as the implied vol is priced above what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_P_1,T_P_1,r,self.sigma,'P','SELL',100))
            else:
                print("At Strike",K_P_1,"The Call volatility is priced appropriately")
                                
            if C_2_vol < self.sigma:
                print("At Strike",K_C_2,"Buy 2M Call") #Buy as the implied vol is priced below what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_C_2,T_C_2,r,self.sigma,'C','SELL',100))
            elif C_2_vol > self.sigma:
                print("At Strike",K_C_2,"Sell 2M Call") #Sell as the implied vol is priced above what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_C_2,T_C_2,r,self.sigma,'C','BUY',100))
            else:
                print("At Strike",K_C_2,"The Call volatility is priced appropriately")

            if P_2_vol < self.sigma:
                print("At Strike",K_P_2,"Buy 2M Put") #Buy as the implied vol is priced below what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_P_2,T_P_2,r,self.sigma,'P','BUY',100))
            elif P_2_vol > self.sigma:
                print("At Strike",K_P_2,"Sell 2M Put") #Sell as the implied vol is priced above what is forecast
                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K_P_2,T_P_2,r,self.sigma,'P','SELL',100))
            else:
                print("At Strike",K_P_2,"The Call volatility is priced appropriately")

        oids = self.options_execution_manager.execute_orders([orders], 'OPTION')
        
        plt.plot(call_skew_1,marker='o',markersize=8,color='blue',linewidth=2)
        plt.plot(put_skew_1, marker='o',markersize=8,color='red',linewidth=2)
        plt.plot(call_skew_2,marker='o',markersize=8,color='green',linewidth=2)
        plt.plot(put_skew_2, marker='o',markersize=8,color='orange',linewidth=2)
        plt.legend([call_skew_1, put_skew_1,call_skew_2, put_skew_2], ['call_skew 1m', 'put_skew 1m, call_skew 2m', 'put_skew 2m'])
        red_patch = mpatches.Patch(color='red', label='put_skew 1M')
        blue_patch = mpatches.Patch(color='blue', label='call_skew 1M')
        green_patch = mpatches.Patch(color='green', label='call_skew 2M')
        orange_patch = mpatches.Patch(color='orange', label='put_skew 2M')
        plt.legend(handles=[blue_patch,red_patch,green_patch,orange_patch])
        plt.show()

    "___________________Put Call Parity Trading Algorithm________________________"

    def specific_option_misprice(self,tickers):

        orders = []

        for i in range(len(tickers)-1):
            C_1 = self.options[tickers['ticker_C_1'][i]].get_midprice()
            P_1 = self.options[tickers['ticker_P_1'][i]].get_midprice()

            S, K, T, option = self.options[tickers['ticker_C_1'][i]].option_disect()
            r = self.r

            if C_1 > P_1 + S - K*np.exp(-r*T):
                
                print("At Strike",K,"Buy Put 1M, Sell Call 1M")

                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'C','BUY',100))

                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'P','BUY',100))
                
            elif C_1 < P_1 + S - K*np.exp(-r*T):
                
                print("At Strike",K,"Buy Call 1M, Sell Put 1M")

                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_1'][i] , 'MARKET','BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'C','SELL',100))

                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_1'][i] , 'MARKET','SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'P','SELL',100))
                

            C_2 = self.options[tickers['ticker_C_2'][i]].get_midprice()
            P_2 = self.options[tickers['ticker_P_2'][i]].get_midprice()

            S, K, T, option = self.options[tickers['ticker_C_2'][i]].option_disect()

            if C_2 > P_2 + S - K*np.exp(-r*T):
                
                print("At Strike",K,"Buy Put 2M, Sell Call 2M")

                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET', 'SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'C','BUY',100))

                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET', 'BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'P','BUY',100))

            elif C_2 < P_2 + S - K*np.exp(-r*T):
                
                print("At Strike",K,"Buy Call 2M, Sell Put 2M")

                orders.append(self.options_execution_manager.create_order(tickers['ticker_C_2'][i] , 'MARKET',  'BUY', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'C','SELL',100))

                orders.append(self.options_execution_manager.create_order(tickers['ticker_P_2'][i] , 'MARKET',  'SELL', 100))
                orders.append(self.options_execution_manager.delta_hedge(S,K,T,r,self.sigma,'P','SELL',100))

        oids = self.options_execution_manager.execute_orders([orders], 'OPTION')

"""-------------- RUNTIME --------------"""
def install_thread_excepthook():
    """
    Workaround for sys.excepthook thread bug
    (https://sourceforge.net/tracker/?func=detail&atid=105470&aid=1230540&group_id=5470).
    Call once from __main__ before creating any threads.
    If using psyco, call psycho.cannotcompile(threading.Thread.run)
    since this replaces a new-style class method.
    """
    import sys
    run_old = threading.Thread.run
    def run(*args, **kwargs):
        try:
            run_old(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            sys.excepthook(*sys.exc_info())
    threading.Thread.run = run

#TODO: Get threads print statements working
def main():
    print('reached')
    with OptionsTradingManager(api) as tm:
        
        for t in TradingTick(295,  api):
            pass

if __name__ == '__main__':
    # install_thread_excepthook()
    main()
        