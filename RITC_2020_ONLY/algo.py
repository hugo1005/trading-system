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

# TODO: Set proper trading sizes for all order quantities
class TradingManager():
    def __init__(self, tickers, risk_aversion=0.005, enable_market_maker = True, accept_tender_orders = True, enable_arbitrage=True, poll_delay=0.005):
        """" 
        Initialises trading manager
        param tickers: list of tickers of securties
        param risk_aversion: probability of a loss exceeding Z(risk_aversion) standard deviations of long term volatility
        """
        print("[TradingManager] Configuring...")

        self.tickers = tickers
        self.securities = {}
        self.risk_aversion = risk_aversion

        self.enable_market_maker = enable_market_maker
        self.accept_tender_orders = accept_tender_orders
        self.enable_arbitrage = enable_arbitrage
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
        # So this now works but we need to worry about hedging currency risk and any residual
        # when algo not trading
        # self.market_maker = Thread(target=self.make_markets)
        # self.market_maker.start()

        self.tender_watcher = Thread(target=self.watch_for_tenders, name="Tender Watcher")
        self.tender_watcher.start()

        # self.arbitrage_searcher = Thread(target=self.search_for_arbitrage)
        # self.arbitrage_searcher.start()

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

    """ ------- Market Maker ------- """
    def make_markets(self):
        market_making_order_ids = []
        print('[MarketMaker] Started, awaiting indicators from securities...')
        
        # This just continues to yield ticks until it exceeds 295
        # Note consecutive t values may be the same if code completes
        # within one tick

        for t in TradingTick(295, self.api):
            """ Market Making Logic """
            
            # Pulls any orders that haven't been executed,
            # Serves also to trigger pnl updates
            self.execution_manager.pull_orders(market_making_order_ids)
            orders = []

            if not self.enable_market_maker:
                self.execution_manager.hedge_position('MARKET_MAKER')
                break

            for sec_key in self.securities:
                security = self.securities[sec_key]
                # Currency's tend to be too illiquid
                # Checking for VPIN is a quick hack to check that sufficient time has past to accurately compute indicators
                if security.is_currency == False and 'VPIN' in security.indicators:
                    mid_price = security.get_midprice()

                    # Volume Probability of Informed Trading
                    # "Flow Toxicity and Liquidity in a High-frequency World (Easley et al. 2012)"
                    vpin = security.indicators['VPIN']

                    # We're likely to get caught here by informed traders
                    if vpin > 0.8:
                        print('[MarketMaker] Market too toxic to trade')
                        self.execution_manager.hedge_position('MARKET_MAKER')
                        continue

                    print('[VPIN: %s] %.3f' %(security.ticker, vpin))

                    max_viable_spread = self.compute_max_viable_spread(security)

                    optimal_spread = vpin * max_viable_spread

                    # Place Limit Orders Symmetrically 
                    # spread_from_mid = optimal_spread / 2

                    print('[MarketMaker] Mid Price: %s Max Viable Spread: %s Optimal Spread: %s' % (mid_price, max_viable_spread, optimal_spread))
                    
                    # Assymetric placing of orders to ensure we don't get adversely selected as often
                    imbalance = security.indicators['order_imbalance']
                    spread_ask = optimal_spread * abs(imbalance) if imbalance > 0 else optimal_spread * (1-abs(imbalance))
                    spread_bid = optimal_spread * abs(imbalance) if imbalance < 0 else optimal_spread * (1-abs(imbalance))
                    
                    print('[MarketMaker] Placing assymetric spread: bid: %s ask: %s' % (spread_bid, spread_ask))

                    orders.append(self.execution_manager.create_order(security.ticker, 'LIMIT', 'BUY', 1000, mid_price - spread_bid))

                    orders.append(self.execution_manager.create_order(security.ticker, 'LIMIT', 'SELL', 1000, mid_price + spread_ask))

                    # Executes 2 orders
                    market_making_order_ids = self.execution_manager.execute_orders(orders,'MARKET_MAKER')
 
    def compute_max_viable_spread(self, security):
        # Short Term Volatility
        vol = security.indicators['volatility']

        # Risk Aversion Z-Score 
        z_value = st.norm.ppf(1 - self.risk_aversion)

        # Computes the maximum spread at which market makers
        # will provide liquidity given their risk aversion
        # The general idea is that higher vol means wider spread
        max_viable_spread = z_value * vol 

        return max_viable_spread

    """ ------- Statistical Arbitrage -------- 
    Method used described in article: https://medium.com/@hugojdolan/maths-guide-to-pairs-trading-19f793543cf7
    Originally sourced from "Pairs Trading Quantitative Methods and Analysis (Ganapthy Vidyamurthy)" 
    #TODO: Implement risk controls if the spread deviates massively (this is unlikely as RITC specifically
    # states the equilibrium relationship)
    """
    def search_for_arbitrage(self, trading_size = 500):
        optimal_threshold = self.calibrate_model()
        last_spread = None
        position_status = 'CLOSED'
        position_cointegration = 0

        for t in TradingTick(295, self.api):
            """ Arbitrage Logic """
            if not self.enable_arbitrage:
                # TODO: Any wind down logic
                break

            # Recalibrate the model every 5 seconds (Arbitrary)
            if t % 5 == 0:
                optimal_threshold = self.calibrate_model()

            # ----- Note this has been hardcoded for the RITC 2020 competition ------
            cointegration_coeff = self.securities['USD'].get_midprice()
            spread = self.get_spread(['BEAR','BULL'], ['RITC'], cointegration_coeff)

            if abs(spread) >= optimal_threshold and position_status == 'CLOSED':
                leg_1_dir = 'BUY' if spread > 0 else 'SELL'
                leg_2_dir = 'SELL' if spread > 0 else 'BUY'

                # We execute on new threads to ensure simulataneous position entry
                # Don't procede until all positions have been executed (.join())
                thread1 = self.optimally_execute_order_on_new_thread('BEAR',volume=trading_size/2,
                 hiding_volume=self.compute_hiding_volume('BEAR', trading_size/2, leg_1_dir),action=leg_1_dir)
                thread1.join()

                thread2 = self.optimally_execute_order_on_new_thread('BULL',volume=trading_size/2,
                 hiding_volume=self.compute_hiding_volume('BULL', trading_size/2, leg_1_dir),action=leg_1_dir)
                thread3.join()

                thread3 = self.optimally_execute_order_on_new_thread('RITC',volume=trading_size * cointegration_coeff,
                 hiding_volume=self.compute_hiding_volume('RITC', trading_size * cointegration_coeff, leg_2_dir),action=leg_2_dir)
                thread3.join()

                position_status = 'LONG' if spread < 0 else 'SHORT'
                position_cointegration = cointegration_coeff

            # Spread Cross and Unwind
            if last_spread != None and last_spread * spread < 0:
                leg_1_dir = 'BUY' if position_status == 'LONG' else 'SELL'
                leg_2_dir = 'SELL' if position_status == 'LONG' else 'BUY'

                # We execute on new threads to ensure simulataneous position entry
                # Don't procede until all positions have been executed (.join())
                thread1 = self.optimally_execute_order_on_new_thread('BEAR',volume=trading_size/2,
                 hiding_volume=self.compute_hiding_volume('BEAR', trading_size/2, leg_1_dir),action=leg_1_dir)
                thread1.join()

                thread3 = self.optimally_execute_order_on_new_thread('BULL',volume=trading_size/2,
                 hiding_volume=self.compute_hiding_volume('BULL', trading_size/2, leg_1_dir),action=leg_1_dir)
                thread2.join()

                thread3 = self.optimally_execute_order_on_new_thread('RITC',volume=trading_size * cointegration_coeff,
                 hiding_volume=self.compute_hiding_volume('RITC', trading_size * position_cointegration, leg_2_dir),action=leg_2_dir)
                thread3.join()

                position_status = 'CLOSED'
                position_cointegration = 0

            last_spread = spread

    def calibrate_model(self):
        """
        This model calibration is specific to the 2020 RITC competition.
        :returns optimal_threshold: at which the spread should be traded back to equilibrium
        """
        USD_close = self.securities['USD'].get_ohlc_history(freq="100ms")['c']
        historical_spread, avg_slippage = self.construct_historical_spread(['BEAR','BULL'], ['RITC'], USD_close)
        probabilities = self.get_threshold_probaility_curve(historical_spread, avg_slippage)
        optimal_threshold = self.get_optimal_threshold(probabilities)

        return optimal_threshold

    def construct_historical_spread(self, leg_1, leg_2, cointegration_coeff):
        """
        We assume equal weighting within each leg of the portfolio. Cointegration coefficent
        is applied to the close prices of leg_2. (Spread constructed as leg_2 - leg_1)
        :param leg_1: a list of ticker symbols consisting of the first spread component
        :param leg_2: a list of ticker symbols consisting of the second spread component
        :param cointegration_coeff: a series of cointegration coefficients values or a constant
        :return spread, avg_total_slippage: teh spread series for the ecuirties and the total average slippage
        """
        
        leg_1_closes = []
        leg_2_closes = []

        slippages = []

        for ticker in leg_1:
            leg_1_closes.append(self.securities[ticker].get_ohlc_history(freq="100ms")['c'])
            slippages.append(self.securities[ticker].get_average_slippage())

        for ticker in leg_2:
            leg_2_closes.append(self.securities[ticker].get_ohlc_history(freq="100ms")['c'])
            slippages.append(self.securities[ticker].get_average_slippage())
        
        leg_1 = pd.concat(leg_1_closes, axis=1).sum(axis=1)
        leg_2 = pd.concat(leg_2_closes, axis=1).sum(axis=1) * cointegration_coeff
        spread = (leg_2-leg_1).interpolate().dropna()

        return spread, sum(slippages)

    def get_spread(self, leg_1, leg_2, cointegration_coeff):
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

        for ticker in leg_1:
            leg_1_prices.append(self.securities[ticker].get_midprice())

        for ticker in leg_2:
            leg_2_prices.append(self.securities[ticker].get_midprice())
        
        leg_1 = pd.concat(leg_1_prices, axis=1).sum(axis=1)
        leg_2 = pd.concat(leg_2_prices, axis=1).sum(axis=1) * cointegration_coeff
        spread = leg_2-leg_1

        return spread

    def get_threshold_probaility_curve(historical_spread, slippage):
        """
        Computes a curve which estimates the probability that the spread series
        exceeds discrete thresholds up to 3 stedevs. The curve is then smoothed
        to ensure it is strictly decreasing and to prevent overfitting to noise.
        :param historical_spread: The spread series for specified assets
        :param slippage: sum of average slippages across all assets in spread
        :return smoothed_probability_curve: Estimated of the probability of exceeding thresholds as a dataframe
        containing columns ['threshold', 'probability']
        """
        mean = historical_spread.mean()
        std = historical_spread.std()

        centred_spread = historical_spread - mean
        abs_spread = centred_spread.abs()

        thresholds = np.arange(start=0, stop=std*3, step=slippage)

        n_samples = abs_spread.shape[0]

        threshold_probabilty_curve = []

        for threshold in thresholds:
            probability_of_exceeding_threshold = (abs_spread > threshold).count() / n_samples
            threshold_probabilty_curve.append(probability_of_exceeding_threshold)

        threshold_probabilty_curve = pd.Series(threshold_probabilty_curve)
        
        # Makes curve stricly decreasing
        threshold_probabilty_curve[threshold_probabilty_curve.diff() > 0] = np.nan
        threshold_probabilty_curve = threshold_probabilty_curve.interpolate()
        
        # Kernel Smoothing 
        # Parameter alpha = 10**0 seems to be decent
        clf = KernelRidge(alpha=float(10)**0, kernel='rbf')
        X, y = thresholds, threshold_probabilty_curve.values
        clf.fit(X,y)
        smoothed_probability_curve = pd.DataFrame({'threshold':X,'probability':clf.predict(X)})

        return smoothed_probability_curve

    def get_optimal_threshold(probabilities):
        profit_curve = 2 * probabilities['threshold']
        profitability = profit_curve * probabilities['probabilities']

        argmax = profitability.idxmax()

        return probabilities['threshold'][argmax]

    """ ------- Tenders -------- """

    def watch_for_tenders(self):
        has_hedged_currency = False

        for t in TradingTick(295, self.api):
            
            if t % 5 == 0:
                if not has_hedged_currency:
                    self.execution_manager.hedge_position('TENDER')
                    has_hedged_currency = True
            else:
                has_hedged_currency = False

            if not self.accept_tender_orders:
                # TODO: Any wind down logic
                break;

            res = requests.get(self.endpoint + '/tenders', headers=self.headers)

            if res.ok:
                if 'VPIN' in self.securities['RITC'].indicators:
                    tenders = res.json()
                    
                    while len(tenders) > 0:
                        tender = tenders.pop()
                        print('[Tenders] Evaluating Tender: %s' % tender)

                        is_profitable, hiding_volume = self.process_tender_order('RITC',
                        tender['quantity'], tender['action'], tender['price'])

                        fake_order = self.execution_manager.create_order('RITC', 'TENDER', tender['action'], tender['quantity'], tender['price'])

                        is_within_risk = self.execution_manager.can_execute_orders([fake_order])
                        
                        print('[Tenders] Is within risk: %s Is profitable: %s hiding volume: %s' % (is_within_risk, is_profitable, hiding_volume))

                        if is_profitable and is_within_risk:
                            self.execution_manager.accept_tender(tender)

                            inverse_action = 'BUY' if tender['action'] == 'SELL' else 'SELL'

                            self.optimally_execute_order_on_new_thread('RITC', tender['quantity'], hiding_volume, inverse_action)
                        else:
                            self.execution_manager.decline_tender(tender)
            else:
                print('[Tenders] Could not reach API with code %s' % res.status_code)

    def optimally_execute_order_on_new_thread(self, ticker, volume, hiding_volume, action,
    num_large_orders = 3, num_proceeding_small_orders = 10, large_to_small_order_size_ratio = 5, vpin_threshold=0.6):
        optimal_exec_thread = Thread(target=self.optimally_execute_order, 
        args=(ticker, volume, hiding_volume, action), kwargs={"num_large_orders": num_large_orders, "num_proceeding_small_orders": num_proceeding_small_orders, "large_to_small_order_size_ratio": large_to_small_order_size_ratio, "vpin_threshold": vpin_threshold})
        
        optimal_exec_thread.start()

        return optimal_exec_thread

    def optimally_execute_order(self, ticker, volume, hiding_volume, action,
    num_large_orders = 3, num_proceeding_small_orders = 10, large_to_small_order_size_ratio = 5, vpin_threshold=0.6):
        """
        We need to optimally hedge our tender expsoure and execute within the 
        hidden volume specified to minimise price impact. Our method goes as follows:
        1) We should place larger orders spaced between smaller orders
        in order to allow temporary price impact to recover
        2) We should place market orders when the orderbook imbalance is against us
        we should place limit orders when the orderbook imbalance is in our favour

        :param ticker: ticker of secuirty (string)
        :param volume: total quantity to hedge
        :param hiding_volume: the optimal transcation volume to hide our order in

        This is certaintly an area we could investigate and model more mathematically.
        TODO: num_large_orders = 3, num_proceeding_small_orders = 10, large_to_small_order_size_ratio = 5
        vpin_threshold=0.6 are arbitrary and should be changed to something appropriate.
        """

        transcated_volume = 0

        # Compute the size of a small volume unit
        # Note a large trade is "large_to_small_order_size_ratio" trading units
        num_trading_units = num_large_orders * (large_to_small_order_size_ratio + num_proceeding_small_orders)
        trading_unit_volume = volume / num_trading_units
        trading_unit_hiding_volume = hiding_volume / num_trading_units

        # Define the sequence of order quantities and their corresponding hiding quantities
        order_qty_seq = ([trading_unit_volume * large_to_small_order_size_ratio] + ([trading_unit_volume] * num_proceeding_small_orders)) * num_large_orders

        order_hiding_volume_seq = ([trading_unit_hiding_volume * large_to_small_order_size_ratio] + ([trading_unit_hiding_volume] * num_proceeding_small_orders)) * num_large_orders
        
        order_idx = 0
        
        while transcated_volume < volume:
            print("[Tender] Pct of Qty Hedged: %.2f" % (transcated_volume/volume))
            qty = order_qty_seq[order_idx]
            hide_in = order_hiding_volume_seq[order_idx]
            security = self.securities[ticker]

            # Determine order type based on current VPIN
            # If VPIN is high we go to market because someone else 
            # is quite likely to adversely select us.
            vpin = security.indicators['VPIN']
            
            if vpin > vpin_threshold:
                order_type = 'MARKET'
                price = None
            else:
                order_type = 'LIMIT'
                best_bid, best_ask = security.get_best_bid_ask()         
                price = best_bid if action == 'BUY' else best_ask

            order = self.execution_manager.create_order(ticker, order_type, action, qty, price)
            oid = self.execution_manager.execute_orders([order], source='TENDER')[0]
            
            execution_time = pd.to_datetime(time(), unit='s')
        
            # Don't do anyting until the order has been executed and
            # the total hiding volume has elapsed
            accumulated_volume = security.get_accumulated_transcation_volume(execution_time)
            transacted = self.execution_manager.is_order_transacted(oid)
            while accumulated_volume <= hide_in or not transacted:
                sleep(0.005)
                # print(transacted)
                # print("[TENDER] Hiding order... (pct_done: %.2f)" % (accumulated_volume/hide_in))

                # We don't want our limit order slipping down the book
                # if the price moves away from us
                if order_type == 'LIMIT' and not transacted:
                    qty_filled = self.execution_manager.get_order_filled_qty(oid)
                    self.execution_manager.pull_orders([oid])

                    # Need to force market order if we're not getting any traction
                    if accumulated_volume > hide_in:
                        print('[TENDER] Hedging order not yet executed, switching to MARKET order')
                        order_type = 'MARKET'
                        price = None

                        order = self.execution_manager.create_order(ticker, order_type, action, qty, price)
                        oid = self.execution_manager.execute_orders([order], source='TENDER')[0]
                        break
                    else:
                        # Update the amount and chase the best price
                        best_bid, best_ask = security.get_best_bid_ask() 
                        order['price'] = best_bid if action == 'BUY' else best_ask
                        order['quantity'] -= qty_filled
                    
                        oid = self.execution_manager.execute_orders([order], source='TENDER')[0]

                transacted = self.execution_manager.is_order_transacted(oid)
                accumulated_volume = security.get_accumulated_transcation_volume(execution_time)
            
            # Move onto the next order in the sequence
            transcated_volume += qty
            order_idx += 1

    # TODO: Refactor this with function compute_hiding_volume()      
    def process_tender_order(self, ticker, volume, action, price):
        """Evaluates a tender based on computing the optimal volume to conceal the requested order
            and the potential 
            :param ticker: A string ticker / symbol (Always going to be RITC ETF)
            :param volume: The size of the order requested on the secuirty by tender
            :param action: The direction in which we are obliged to BUY / SELL the security and volume requested by the tender.
            :return is_tradable, hiding_volume
        """
        security = self.securities[ticker]
    
        best_bid, best_ask = security.get_best_bid_ask()

        # Compute the premium we will earn per share at current market price
        premium = best_ask - price if action == "BUY" else price - best_bid
        directional_qty = volume if action == "BUY" else -1 * volume
        
        # Percentage of private information leaked due to our trading activity
        # Modelled as a linear function, assuming that any order of maximum size (10,000)
        # will reveal our intentions in their entirety
        # From last years charts this seems reasonable (Can be adjusted if necessary)
        leakage_probabililty = volume / security.max_trade_size
        
        # See "Optimal Execution Horizon, Prado, O'Hara (2015)"
        
        vpin = security.indicators['VPIN']
        std_price_changes = security.indicators['std_price_changes'] # Note this is not true volatility,
        volume_bucket_size = security.indicators['std_price_changes_volume']
        buy_volume_percentage =  (vpin + 1) / 2
        max_spread = self.compute_max_viable_spread(security)
        
        # Permanent Price Impact (Linearly expands the max_spread,
        # thus impacting the mid price in the correct direction and scales as a function of |volume|)
        # This should be tested for a range of values but if an order is somewhere in the range of 1,000
        # to 10,000 units we could reasonably expect some midprice move 1 - 20 cents 
        # so for the moment we will go with K = 0.00001 => volume = 10,000 will impact permanently
        # at around 0.10 cents
        permanent_price_impact = 0.00001

        # Computes the optimal volume to hide an order of the tender size
        # in order to minimise adverse market impact via private information leakage
        hiding_volume = self.computeOptimalVolume(directional_qty,leakage_probabililty,buy_volume_percentage,std_price_changes,volume_bucket_size,max_spread,permanent_price_impact)
        
        # How long does an average volume bucket last (in units specified)
        vol_bucket_avg_duration = security.indicators['vol_bucket_avg_duration']
        hiding_buckets = hiding_volume / volume_bucket_size

        # Basically i'm saying that the change in prices can be probabalistically bounded
        # by sqrt(t) * stdev(prices) where time is now volume buckets

        potential_adverse_price_change = math.sqrt(hiding_buckets * vol_bucket_avg_duration.seconds) * math.sqrt(security.indicators['volatility'])
        print("Potential adverse price change: %s Avg Bucket Duration: %s Sqrt(Volatility): %s" % (potential_adverse_price_change,vol_bucket_avg_duration.seconds , math.sqrt(security.indicators['volatility']) ))
        print("Premium: %s" % premium)
        is_profitable = premium - potential_adverse_price_change > 0
        
        # We discover that it is insufficient to simply determine the premium
        # We must also account for trend in prices, we will use order imbalance as typically
        # We must react to a fairly immediate move of the market against us after we accept the tender
        
        oi = security.indicators['order_imbalance']
        avg_returns = security.get_net_returns(10)

        print("Order Imablance: %s" % oi)
        print("Avg Returns: %.3f" % avg_returns)

        counter_trend = avg_returns * directional_qty < 0
        counter_trend_lead = oi * directional_qty < 0
        # signficant_counter_trend = counter_trend and abs(avg_returns) > 
        significant_counter_trend_lead = counter_trend_lead and abs(oi) > 0.7
        is_tradable = is_profitable and not significant_counter_trend_lead

        return is_tradable, hiding_volume
    
    def compute_hiding_volume(ticker, volume, action, permanent_price_impact = 0.01):
        """Computes the optimal volume to conceal the requested order
            :param ticker: A string ticker / symbol (Always going to be RITC ETF)
            :param volume: The size of the order requested on the secuirty by tender
            :param action: The direction in which we are obliged to BUY / SELL the security and volume requested by the tender.
            
            :param permanent_price_impact
            Permanent Price Impact (Linearly expands the max_spread,
            thus impacting the mid price in the correct direction and scales as a function of |volume|)
            This should be tested for a range of values but if an order is somewhere in the range of 1,000
            to 10,000 units we could reasonably expect some midprice move 1 - 20 cents 
            so for the moment we will go with K = 0.00001 => volume = 10,000 will impact permanently
            at around 0.10 cents

            :return hiding_volume
        """
        security = self.securities[ticker]
    
        best_bid, best_ask = security.get_best_bid_ask()

        directional_qty = volume if action == "BUY" else -1 * volume
        
        # Percentage of private information leaked due to our trading activity
        # Modelled as a linear function, assuming that any order of maximum size (10,000)
        # will reveal our intentions in their entirety
        # From last years charts this seems reasonable (Can be adjusted if necessary)
        leakage_probabililty = volume / security.max_trade_size
        
        # See "Optimal Execution Horizon, Prado, O'Hara (2015)"
        vpin = security.indicators['VPIN']
        std_price_changes = security.indicators['std_price_changes'] # Note this is not true volatility,
        volume_bucket_size = security.indicators['std_price_changes_volume']
        buy_volume_percentage =  (vpin + 1) / 2
        max_spread = self.compute_max_viable_spread(security)
        
        # Permanent Price Impact (Linearly expands the max_spread,
        # thus impacting the mid price in the correct direction and scales as a function of |volume|)
        # This should be tested for a range of values but if an order is somewhere in the range of 1,000
        # to 10,000 units we could reasonably expect some midprice move 1 - 20 cents 
        # so for the moment we will go with K = 0.00001 => volume = 10,000 will impact permanently
        # at around 0.10 cents
        permanent_price_impact = 0.00001

        # Computes the optimal volume to hide an order of the tender size
        # in order to minimise adverse market impact via private information leakage
        hiding_volume = self.computeOptimalVolume(m=directional_qty,
         phi=leakage_probabililty,
         vB=buy_volume_percentage,
         sigma=std_price_changes,
         volSigma=volume_bucket_size, 
         S_S=max_spread, K=permanent_price_impact)
        
        # How long does an average volume bucket last (in units specified)
        vol_bucket_avg_duration = security.indicators['vol_bucket_avg_duration']
        hiding_buckets = hiding_volume / volume_bucket_size

        return hiding_volume

    """ ------- Computing Optimal Execution Horizon -------- """

    def signum(self,x):
        """ Returns the sign of a value
        :params x: Any real number
        :returns : sign(x)
        """
        if(x < 0):return -1
        elif(x > 0):return 1
        else:return 0

    def getOI(self, v,m,phi,vB,sigma,volSigma):
        """Gets the order imbalance using closed form derived from VPIN
        modified to encorporate leakage of private information into the market
        ie. the amount you plan to transact in small chunks
        :params v: Total Volume in the Bucket
        :params m: The total volume you intend to trade
        :params vB: The Percentage of Buy Volume in market
        :params phi: The percentage of information leakage from the private VPIN
        :params sigma: volatility of the mid price
        :params volSigma: The sqrt(V/volSigma) multiplies rescales the volatility which is per unit Vol Sigma 
        :returns : The post information leakage order imblance
        """
        return phi*(float(m-(2*vB-1)*abs(m))/v+2*vB-1) + (1-phi)*(2*vB-1)

    def getBounds(self, m,phi,vB,sigma,volSigma,S_S,zLambda,k = 0):
        """Computes boundaries which vB must satisfy such that the optimal
        volume to hide order V* >= m.
        :params m: The total volume you intend to trade
        :params phi: The percentage of information leakage from the private VPIN
        :params vB: The Percentage of Buy Volume in market
        :params sigma: volatility of the mid price
        :params volSigma: The sqrt(V/volSigma) multiplies rescales the volatility which is per unit Vol Sigma 
        :params S_S: The expected maximum trading range in which market makers will provide liquidity
        typically computed as the market makers risk aversion factor (zLambda) * long term volatility 
        :params K: A factor for permamanent price impact (Disabled when k=0)
        :returns : The decision boundaries on vB
        """
        vB_l = float(self.signum(m)+1)/2-zLambda*sigma*abs(m)**0.5/ float(4*phi*(S_S+abs(m)*k)*volSigma**0.5)
        vB_u = float(self.signum(m)+1)/2+zLambda*sigma*abs(m)**0.5/ float(4*phi*(S_S+abs(m)*k)*volSigma**0.5)
        vB_z = (self.signum(m)*phi/float(phi-1)+1)/2.
        return vB_l,vB_u,vB_z

    def computeOptimalVolume(self, m,phi,vB,sigma,volSigma,S_S,zLambda,k = 0):
        """Computes the optimal V* to hide an order of size and direction m in.
        :params m: The total volume you intend to trade
        :params phi: The percentage of information leakage from the private VPIN
        :params vB: The Percentage of Buy Volume in market
        :params sigma: volatility of the mid price
        :params volSigma: The sqrt(V/volSigma) multiplies rescales the volatility which is per unit Vol Sigma 
        :params S_S: The expected maximum trading range in which market makers will provide liquidity
        typically computed as the market makers risk aversion factor (zLambda) * long term volatility 
        :params K: A factor for permamanent price impact (Disabled when k=0)
        :returns : The optimal hiding volume V*
        """
        # compute vB boundaries:
        if phi<= 0:phi+= 10**-12
        if phi>= 1:phi-= 10**-12

        vB_l,vB_u,vB_z = self.getBounds(m,phi,vB,sigma,volSigma,S_S,zLambda,k)

        # try alternatives
        if (2*vB-1)*abs(m)<m:
            v1 = (2*phi*((2*vB-1)*abs(m)-m)*(S_S+abs(m)*k)*volSigma**0.5 / float(zLambda*sigma))**(2./3)
            oi = self.getOI(v1,m,phi,vB,sigma,volSigma)
            if oi>0:
                if vB<= vB_u: return v1
                if vB>vB_u: return abs(m)

        elif (2*vB-1)*abs(m)>m:
            v2 = (2*phi*(m-(2*vB-1)*abs(m))*(S_S+abs(m)*k)*volSigma**0.5 / float(zLambda*sigma))**(2./3)
            oi = self.getOI(v2,m,phi,vB,sigma,volSigma)
            if oi<0:
                if vB>= vB_l: return v2
                if vB<vB_l: return abs(m)
                
        elif (2*vB-1)*abs(m) == m: return abs(m)

        if m<0:
            if vB<vB_z: return phi*(abs(m)-m/float(2*vB-1))
            if vB>= vB_z: return abs(m)
        else:
            if vB>= vB_z: return phi*(abs(m)-m/float(2*vB-1))
            if vB<vB_z: return abs(m)

    """ ------- Deprecated -------- """

    def replicate_security(self, security, underlyings, optimiser="MAX", volume=10000):
        """Optimises for the most (MAX) or least (MIN) expensive way to replicate 
        an security with a certain quantity of the asecurity and a weighted basket of underlying securities. We do this to imperfectly hedge a tender requested by an institution by implementing a BUY / SELL (ASEET) + SELL / BUY (MAX / MIN REPLICATING PORTFOLIO) strategy which provides a small profit on the hedge.
            :param security: A string ticker / symbol
            :param underlyings: A list of tickers / symbols
            :param optimiser: Optimise for maximum or minimum price
            :param volume: The size of the order requested on the secuirty by tender
            :return: (Volume in security, Volume in underlying, VWAP of replication)
        """
        pass

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
    with TradingManager(['RITC','BEAR','BULL','USD']) as tm:
        
        for t in TradingTick(295,  API(API_CONFIG, DB_PATH, SQL_CONFIG, use_websocket=False)):
            pass

if __name__ == '__main__':
    # install_thread_excepthook()
    main()
        