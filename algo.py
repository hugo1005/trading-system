from sources import API

from time import sleep, time
from queue import Queue
from threading import Thread
import requests

import scipy.stats as st
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys
import math

API_CONFIG = './configs/api_config.json'
SQL_CONFIG = './configs/sql_config.txt'
DB_PATH = './datasets/hftoolkit_sqlite.db'

class TradingTick(): 
    # Cosntructor 
    def __init__(self, limit, api): 
        self.limit = limit 
        self.endpoint, self.headers = api.get_source_config('RIT')
    
    def get_tick(self):
        # get tick number
        resp = requests.get(self.endpoint + '/case', headers = self.headers)

        if resp.ok:
            case = resp.json()
            return case['tick']
        else:
            print('[TradingManager] API not responding!')
            return -1

    # Called when iteration is initialized 
    def __iter__(self): 
        return self
  
    def __next__(self): 
        # Stop iteration if limit is reached 
        tick = self.get_tick()

        if tick > self.limit or tick < 0: 
            raise StopIteration 
  
        # Else increment and return old value 
        return tick

class Security:
    def __init__(self, ticker, api, poll_delay=0.01):
        self.endpoint, self.headers = api.get_source_config('RIT')
        self.ticker = ticker
        self.init_params()
        self.is_running = False
        self.poll_delay = poll_delay

        self.indicators = {}
        self.book_history = pd.DataFrame(columns=['timestamp','type','price','quantity'])
        self.tas_history = pd.DataFrame(columns=['timestamp','id','type','price','quantity'])
        self.best_bid_ask = pd.DataFrame(columns=['timestamp','best_bid','best_ask', 'midprice'])

    def start():
        print('[Security:%s] Started Polling...' % ticker)
        self.is_running = True
        self.polling_thread = Thread(target=self.poll)
        self.polling_thread.start()

    def shutdown():
        self.is_running = False

    def init_params(self):
        res = requests.get(self.endpoint + '/securities', params={'ticker': self.ticker}, headers = self.headers)

        if res.ok:
            spec = res.json()[0]
            self.lo_rebate = spec['limit_order_rebate'],
            self.mo_fee = spec['trading_fee'],
            self.quoted_decimals = spec['quoted_decimals'],
            self.max_trade_size = spec['max_trade_size'],
            self.max_orders_per_second = spec['api_orders_per_second'],
            self.execution_delay_ms = spec['execution_delay_ms']
        else:
            print('[%s] Parameters could not be found!' % self.ticker)

    """ ------- API Polling -----------"""
    def poll(self):
        while self.is_running:
            # Time and sales
            res_tas = requests.get(self.endpoint + '/securities/tas', params={'ticker': self.ticker}, headers = self.headers)

            # Orderbook
            res_book = requests.get(self.endpoint + '/securities/book', params={'ticker': self.ticker, 'limit': 1000}, headers = self.headers)

            if res_book.ok and res_tas.ok:
                book = res_book.json()
                tas = res_tas.json()
                self.update_book_tas(book, tas)
                self.recompute_indicators()

            sleep(self.poll_delay)
    
    def update_book_tas(self, book, tas):
        # Extract bid ask data from order book response
        bids = pd.DataFrame(book['bid'])
        asks = pd.DataFrame(book['ask'])
        bids = bids.drop(columns=['type'])
        asks = asks.drop(columns=['type'])
        best_bid, best_ask = bids['price'][0], asks['price'][0]
        midprice = (best_bid + best_ask) / 2
        timsetamp = time.time()

        # Update security book history for bid and ask
        self.book_history.append({'timestamp': timsetamp, 'type':'bid', 'price': bids['price'],'quantity': bids['quantity']})

        self.book_history.append({'timestamp': timsetamp, 'type':'ask', 'price': ask['price'],'quantity': ask['quantity']})        
        
        # Update best bid and ask
        self.best_bid_ask.append({'timestamp':[timsetamp], 'best_bid': [best_bid], 'best_ask': [best_ask], 'midprice': [midprice]})

        # Parse time and sales data 
        tas_df = pd.DataFrame(tas)
        tas_df['type'] = tas_df['type'] > midprice
        tas_df['type'] = tas_df['type'].replace(to_replace={0:'BUY',1:'SELL'})
        
        # We attempt to create a more high resolution timestamp than the tick 
        # provided by ritc, we will be accurate to the +-self.poll_delay
        existing_ids = self.tas_history['id'].unique().values
        tas_new_entries = tas_df[~tas_df['id'].isin(existing_ids)]

        self.tas_history.append({'id': tas_new_entries['id'], 'timestamp': timsetamp, 'type': tas_new_entries['type'], 'price': tas_new_entries['price'], 'quantity': tas_new_entries['quantity']})

    def recompute_indicators(self):
        # TODO: Change these hyperparmaters appropriately to correct volumes
        bucket_size = 50
        vpin_results = self.compute_historical_vpin(self.tas_history, BAR_SIZE='0.1s', N_BUCKET_SIZE=bucket_size, SAMPLE_LENGTH=250)["VPIN"][-1]
        vol = self.compute_historical_volatility()

        self.indicators['VPIN'] = vpin_results['VPIN'].values[-1]
        self.indicators['volatility'] = vol
        self.indicators['std_price_changes'] = vpin_results['std_price_changes'].values[-1]
        self.indicators['std_price_changes_volume'] = bucket_size
        self.indicators['vol_bucket_avg_duration'] = vpin_results['bucket_duration'].mean()
        self.indicators['vol_bucket_mid_price_std'] = vpin_results['bucket_mid_prices'].std()

    """ ------- Security State -----------"""
    def get_midprice(self):
        return self.best_bid_ask['midprice'][-1]

    def get_bid_ask_spread(self):
        best_bid = self.best_bid_ask['best_bid'][-1]
        best_ask = self.best_bid_ask['best_ask'][-1]

        return best_ask - best_bid

    def get_best_bid_ask(self):
        best_bid = self.best_bid_ask['best_bid'][-1]
        best_ask = self.best_bid_ask['best_ask'][-1]
        
        return best_bid, best_ask

    def get_accumulated_transcation_volume(self, start_timestamp):
        return self.tas_history[self.tas_history['timestamp'] > start_timestamp]['quantity'].sum()

    def get_ohlc_history(self, freq='0.1s'):
        """
        Gets open high low close history for security
        :param freq: frequency of time aggregation
        :returns: open high low close values aggregated at the specified frequency
        """
        resample = self.best_bid_ask['timestamp','midprice'].resample(freq, on='timestamp')
        ohlc = resample.agg(['first','max','min','last'])
        ohlc.columns = ['o','h','l','c']
        return ohlc
    
    def get_average_slippage(self):
        return (self.best_bid_ask['best_ask'] - self.best_bid_ask['best_bid']).mean()
    
    """ ------- Computing VPIN -------- """
    @staticmethod
    def standardise_order_qty(df, target):
        """Takes orders of any given volume and splits them up into orders of single units.
        We do this as order volume is often misleading as large orders are often split into smaller orders
        anyway. So it is better to standardise the size of the order to 1 and then bucket them later
            :param df: contains the data to be standardised
            :param target: the name of the column thats being standardised
            :return: Dataframe of results
        """
        expanded = []

        filter_vol =  (df['volume'] == 0 )| (df['volume'].isna())
        df_filtered =  df[~filter_vol]

        timestamps = df_filtered.index.values
        vols = df_filtered['volume'].values
        targets = df_filtered[target].values

        for row in zip(timestamps, vols, targets):
            timestamp, vol, target = row
            expanded += [(timestamp, target)] * int(vol)

        return pd.DataFrame.from_records(expanded, index=0)

    @staticmethod
    def compute_historical_vpin(all_trades,BAR_SIZE='1min',N_BUCKET_SIZE=50,SAMPLE_LENGTH=250):
        """Estimates VPIN  (Volume Synchronised Probaility of Informed Trading)
            :param all_trades: contains time and sales information in a data frame
            :param BAR_SIZE: minimum time aggregation for time and sales data
            :param N_BUCKET_SIZE: the volume in each VPIN calculation
            :param SAMPLE_LENGTH: the smoothing factor for rolling average of VPIN
            :returns {"VPIN": vpin_df, "trades_adj": trades_adj, "pct_changes_df": pct_changes_df, "std_price_changes": std_price_changes, 'bucket_duration':diffs, 'bucket_mid_prices': bucket_mid_prices}
        """
        usd_trades = all_trades # Deprecated

        volume = usd_trades['quantity']
        trades = usd_trades['price']

        def cleanup(x):
            if isinstance(x, str) and 'e-' in x:
                return 0
            else:
                return float(x)

        volume = volume.apply(lambda x: cleanup(x))
        volume = volume.astype('float32')
        
        # Aggregates Volume and Price information to BAR_SIZE frequency
        # assign trade sign to 1 minute time bar by averaging buys and sells and taking the more common one
        # HUGO: Facilitates "bulk classification" of trade direction over 1 min in Probablistic terms
        trades_resampled = trades.resample(BAR_SIZE)
        trades_1min = trades_resampled.pct_change().fillna(0)
        price_changes_1min = trades_resampled.diff().fillna(0)
        

        volume_1min = volume.resample(BAR_SIZE).sum()
        typestr = (usd_trades['type'])
        typestr_1min = typestr.astype('float32').resample(BAR_SIZE).mean().round()

        df = pd.DataFrame({'type': typestr_1min, 'volume': volume_1min})
        df_trades = pd.DataFrame({'volume': volume_1min, 'price_delta_pct': trades_1min})
        
        volume_agg_direction = df 
        price_delta_agg = df_trades

        # HUGO: Recall we take each 1 minute volume grouping and split it up into minimum size transaction units
        # HUGO: This ensures that we make no assumptions regarding how large orders may be submitted over time

        expanded = self.standardise_order_qty(df, 'type')
        std_returns_dist = self.standardise_order_qty(df_trades, 'price_delta_pct')

        std_order_direction = expanded

        # --------------- find single-period VPIN ---------------------------
        def grouper(n, iterable):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, n))
                if not chunk:
                    return
                yield chunk
        
        OI = []
        start = 0 
        
        for each in grouper(N_BUCKET_SIZE, std_order_direction[1]):
            slce = pd.Series(each)
            counts = slce.value_counts()
            if len(counts) > 1:
                OI.append(np.abs(counts[1] - counts[0])/N_BUCKET_SIZE)
            else:
                if 0 in counts:
                    OI.append(counts[0]/N_BUCKET_SIZE)
                else:
                    OI.append(counts[1]/N_BUCKET_SIZE)

        # -------- find time boundaries for volume buckets ---------------
        buckets = []
        mid_buckets = []
        diffs = []
        pct_changes = []
        price_changes = []
        bucket_mid_prices = []

        V = N_BUCKET_SIZE
        running_volume = 0.0
        start_idx = None 

        for idx in std_order_direction.index:
            if not start_idx:
                start_idx = idx 

            if running_volume >= V:
                buckets.append((start_idx, idx))
                
                # Find the approximate pct change during the bucket
                pct_changes.append(df_trades[start_idx:idx]['price_delta_pct'].sum())
                price_changes.append(price_changes_1min[start_idx:idx].sum())
                mid_prices.append(trades_resampled[start_idx:idx].mean())
                # find mid time of volume buckets
                # find volume bucket duration
                diff = idx - start_idx
                mid_buckets.append(idx + (diff/2))  
                diffs.append(diff)
                
                start_idx = None
                running_volume = 0
            running_volume += 1

        # HUGO: Computes the VPIN rolling mean and assign the mid time as the index of the bucket
        # This is convenient indexing
        # HUGO: We then  adjust the original trades dataframe to have the same fill forward index 
        # as the vpin_df
        vpin_df = pd.Series(OI[:-1], index=mid_buckets).rolling(SAMPLE_LENGTH).mean()
        trades_adj = trades.resample(BAR_SIZE).sum().reindex_like(vpin_df, method='ffill')
        pct_changes_df = pd.Series(pct_changes, index=mid_buckets)
        std_price_changes = pd.Series(price_changes, index=mid_buckets).rolling(SAMPLE_LENGTH).std()
        
        return {"VPIN": vpin_df, "trades_adj": trades_adj, "pct_changes_df": pct_changes_df, "std_price_changes": std_price_changes, 'bucket_duration':diffs, 'bucket_mid_prices': bucket_mid_prices}
    
    """ ------- Computing Volatility ----- """
    def compute_historical_volatility(self):
        return self.best_bid_ask['midprice'].std()

class ExecutionManager():
    def __init__(self, api, tickers):
        self.is_running = False
        self.api = api
        self.endpoint, self.headers = self.api.get_source_config('RIT')
        self.tickers = tickers

        order_properties = ["order_id", "period" "tick", "trader_id", "ticker","type","quantity","action","price","quantity_filled","vwap","status"]


        """" Order Management Variables """
        self.orders = {
            'OPEN': pd.DataFrame(columns=order_properties),
            'CANCELLED':  pd.DataFrame(columns=order_properties),
            'TRANSACTED':  pd.DataFrame(columns=order_properties)
        }

        """" Risk Management (Per Ticker) """
        self.net_positions = {ticker:0 for ticker in tickers}
        
        """" Risk Limits """
        res = requests.get(self.endpoint + '/limits', headers=self.headers)

        if res.ok:
            limits = res.json()
            self.gross_limits = limits['gross_limit']
            self.net_limits = limits['net_limit']
        else:
            print('[Execution Manager] Error could not obtain position limits from API!')

    """ Order Execution """
    def accept_tender(tender):
        accept_res = requests.post(self.endpoint + '/tenders', params={'id': tender['tender_id']}, headers=headers)

        if accept_res.ok:
            print('[AcceptTenders] Accepted : price: %s qty: %s action: %s' % (tender['price'],
            tender['quantity'], tender['action']))

            # Assuming tender orders don't show up like regular orders
            # So we must account for that here
            ticker = 'RITC'
            qty = tender['quantity']
            direction = 1 if tender['action'] == "BUY" else -1
            qty_directional = qty * direction

            self.net_positions[ticker] += qty_directional
            
            # Just to be sure we don't accidentally front run a tender
            sleep(0.05)
        else:
            print('[AcceptTenders] Could not reach API with code %s' % accept_res.status_code)

    def create_order(self, ticker, order_type, action, qty, price=None):
        return {'ticker': ticker, 'type': order_type, 'action': action, 'quantity': qty, 'price': price}

    def execute_orders(self, orders):
        """
        Sends orders to the RIT API, handles any POST request rate limiting by the API
        :params orders: List of json objects as specified in create_order method
        :return order_ids: returns a list of executed order ids
        """
        executed_orders = []

        if self.can_execute_orders(orders):

            # API is rate limited to 5 orders per second  
            while len(orders) > 0:
                order = orders.pop()
                res = requests.post(self.endpoint + '/orders', data=orders, headers = self.headers)
                content = res.json()

                if res.ok:
                    print('[Trading Manager] Order placed: [Price: %s, Qty: %s Type: %s Ticker: %s ]' % (order['price'],order['quantity'],order['type'],order['ticker']))

                    executed_orders.append(content)
                    """"
                    Example content:
                    {
                        "order_id": 1221,
                        "period": 1,
                        "tick": 10,
                        "trader_id": "trader49",
                        "ticker": "CRZY",
                        "type": "LIMIT",
                        "quantity": 100,
                        "action": "BUY",
                        "price": 14.21,
                        "quantity_filled": 10,
                        "vwap": 14.21,
                        "status": "OPEN"
                    }
                    """
                elif res.status_code == 429:
                    # Try again after wait time
                    sleep(content['wait'] + 0.01)
                    orders.append(order)

        return [order[id] for order in executed_orders]
    
    def pull_orders(self, order_ids):
        """
        Pulls specified open orders from the book.
        :param order_ids: A list of order id's, this facilitates cancelling of specific orders. Note we do not facilitate this to be left undefined to ensure
        the good practice of tracking open orders.
        """
        # Ensures orders have been enqueued to the book 
        sleep(0.25)
        
        cancelled_ids = self.orders['CANCELLED']['order_id'].values
        transcated_ids = self.orders['TRANSACTED']['order_id'].values
        
        for oid in order_ids:
            if oid in cancelled_ids:
                print(['[ExecutionManager] Order [%s] has already been cancelled!' % oid])
            elif oid in transcated_ids:
                print(['[ExecutionManager] Order [%s] has been transcated, details:' % oid])
            else:
                res = requests.delete(self.endpoint + '/orders', params={'id': oid})
                
                if res.ok:
                    print('[ExecutionManager] Order [%s] cancelled.' % oid)
                    
                else:
                    print('[ExecutionManager] Order [%s] could not be cancelled, code: %s' % (oid, res.status_code))

    def is_order_transacted(self, order_id):
        transacted_ids = self.orders['TRANSACTED']['order_id'].values
        return order_id in transacted_ids

    def get_order_filled_qty(self, order_id):
        status = ['OPEN', 'TRANSACTED', 'CANCELLED']
        
        for stat in status:
            orders = self.orders[stat]
            results = orders[orders['order_id'] == order_id]['quantity_filled']

            if results.shape[0] > 0:
                return results[0]

        return -1 # Could not be found

    """ Risk Control Logic """

    def can_execute_orders(self, orders):
        """Evaluates whether a set of orders can be made without exceeding risk limits.
        This has the advantage of centralising all risk management.
            :param orders: List of orders
            :return: True or False
        """
        orders_df = pd.DataFrame(orders)
        orders_df['direction'] = orders_df['action'].replace(to_replace={'BUY': 1, 'SELL': -1})
        
        orders_df['multiplier'] = 1
        orders_df['multiplier'][orders_df['ticker'] == 'RITC'] = 2

        orders_df['directional_qty'] = orders_df['multiplier'] * orders_df['direction'] * orders_df['quantity']

        current_net_position = sum([self.net_positions[t] for t in self.net_positions])
        current_gross_position = sum([abs(self.net_positions[t]) for t in self.net_positions])

        additional_net_position = orders_df['directional_qty'].sum()
        additional_gross_position = orders_df['directional_qty'].abs().sum()

        is_within_limits =  (self.net_limit > additional_net_position + current_net_position) or \
            (self.gross_limit > additional_gross_position + current_gross_position)
        
        return is_within_limits

    def update_net_position(self, order_id):
        """
        Update net position, retrieves information about transacted order and updates the net positions.
        :params order_id: id of order
        """
        
        # update net_positions from transacted order
        order_details = self.orders['TRANSACTED'][self.orders['TRANSACTED']['order_id'] == order_id]
        
        print(order_details)

        ticker = order_details['ticker'][0]
        qty = order_details['quantity_filled'][0]
        direction = 1 if order_details['action'][0] == "BUY" else -1
        qty_directional = qty * direction

        self.net_positions[ticker] += qty_directional
    
    """ Order Fill Monitoring """

    def start(self):
        self.is_running = True
        self.polling_thread = Thread(target=self.poll)
        self.polling_thread.start()

    def shutdown(self):
        self.is_running = False

    def poll():
        """
        Polls the api for updates on order status (on a new thread)
        """
        while self.is_running:
            for order_status in self.orders:
                self.update_orders_for_status(order_status)
            
            # Min time before more orders are processed
            sleep(0.25)

    def update_orders_for_status(self, status='OPEN'):
        """
        Polls the API to update our current open, closed and transcated orders.
        Updates net security positions based off newly transacted orders.
        :params status: one of OPEN, CANCELLED, TRANSACTED
        """
        res = requests.get(params={'status': status}, headers = self.headers)

        if res.ok:
            orders = res.json()

            if len(orders) > 0:
                updated_orders = pd.DataFrame(orders)
                
                # Handles updating of net positions (Cancelled order may have been partially filled!)
                if status == 'TRANSACTED' or status == 'CANCELLED':
                    new_transactions_ids = list(set(updated_orders['order_id'].values).difference(set(self.orders[status]['order_id'].values)))

                    self.orders[status] = pd.DataFrame(orders)

                    for oid in new_transactions_ids:
                        self.update_net_position(order_id)
                else:
                    self.orders[status] = pd.DataFrame(orders)
        else:
            print('[ExecutionManager] Polling response failed with code: %s' % res.status_code)

# TODO: Set proper trading sizes for all order quantities
class TradingManager():
    def __init__(self, tickers, risk_aversion=0.005, enable_market_maker = True, accept_tender_orders = True, enable_arbitrage=True):
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

        # Request Securities price history from server
        self.api = API(API_CONFIG, DB_PATH, SQL_CONFIG, use_websocket=False)
        self.endpoint, self.headers = self.api.get_source_config('RIT')
        self.execution_manager = ExecutionManager(self.api, self.tickers)

       
    """ ------- Trading Start / Stop -------- """

    def __enter__(self):
        for ticker in self.tickers:
            sec = Security(ticker, self.api)
            sec.start() # Starts polling for data

            self.securities[ticker] = sec

        sleep(0.3) # Lets securities start polling

        self.market_maker = Thread(target=self.make_markets)
        self.market_maker.start()

        self.tender_watcher = Thread(target=self.watch_for_tenders)
        self.tender_watcher.start()

        self.arbitrage_searcher = Thread(target=self.search_for_arbitrage)
        self.arbitrage_searcher.start()

    def __exit__(self):
        for security in self.securities:
            security.shutdown() # Stops polling, kills threads

        self.enable_market_maker = False
        self.accept_tender_orders = False
        self.enable_arbitrage = False

    """ ------- Market Maker ------- """
    def make_markets(self):
        market_making_order_ids = []
        
        # This just continues to yield ticks until it exceeds 295
        # Note consecutive t values may be the same if code completes
        # within one tick

        for t in TradingTick(295):
            """ Market Making Logic """
            
            # Pulls any orders that haven't been executed,
            # Serves also to trigger pnl updates
            self.execution_manager.pull_orders(market_making_order_ids)

            if not self.enable_market_maker:
                # TODO: Any wind down logic
                break;

            for security in self.securities:
                mid_price = security.get_midprice()

                # Volume Probability of Informed Trading
                # "Flow Toxicity and Liquidity in a High-frequency World (Easley et al. 2012)"
                vpin = security.indicators['VPIN']

                max_viable_spread = compute_max_viable_spread(security)

                optimal_spread = vpin * max_viable_spread

                # Place Limit Orders Symmetrically 
                spread_from_mid = optimal_spread / 2

                orders = []
                orders.append(self.execution_manager.create_order(security.ticker, 'LIMIT', 'BUY', 1, mid_price - spread_from_mid))

                orders.append(self.execution_manager.create_order(security.ticker, 'LIMIT', 'SELL', 1, mid_price + spread_from_mid))

                # Executes 2 orders
                market_making_order_ids = self.execution_manager.execute_orders(orders)
 
    def compute_max_viable_spread(self, security):
        # Long Run Volatility
        vol = security.indicators['volatility']

        # Risk Aversion Z-Score 
        z_value = st.norm.cdf(1 - self.risk_aversion)

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

        for t in TradingTick(295):
            """ Arbitrage Logic """
            if not self.enable_arbitrage:
                # TODO: Any wind down logic
                break;

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
        USD_close = self.securities['USD'].get_ohlc_history(freq="0.1s")['c']
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
            leg_1_closes.append(self.securities[ticker].get_ohlc_history(freq="0.1s")['c'])
            slippages.append(self.securities[ticker].get_average_slippage())

        for ticker in leg_2:
            leg_2_closes.append(self.securities[ticker].get_ohlc_history(freq="0.1s")['c'])
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
        
        for t in TradingTick(295):

            if not self.accept_tender_orders:
                # TODO: Any wind down logic
                break;

            res = requests.get(self.endpoint + '/tenders', headers=headers)

            if res.ok:
                tenders = res.json()

                for tender in tenders:
                    is_profitable, hiding_volume = self.process_tender_order('RITC',
                     tender['quantity'], tender['action'], tender['price'])

                    fake_order = self.execution_manager.create_order('RITC', 'TENDER', tender['action'], tender['quantity'], tender['price'])

                    is_within_risk = self.execution_manager.can_execute_orders([fake_order])

                    if is_profitable and is_within_risk:
                        self.execution_manager.accept_tender(tender)

                        inverse_action = 'BUY' if tender['action'] == 'SELL' else 'SELL'

                        self.optimally_execute_order('RITC', tender['quantity'], hiding_volume, inverse_action)

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
        
        while transcated_volume <= volume:
            qty = order_qty_seq[order_idx]
            hide_in = order_hiding_volume_seq[order_seq]
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
            oid = self.execution_manager.execute_orders([order])[0]
            
            execution_time = time()
        
            # Don't do anyting until the order has been executed and
            # the total hiding volume has elapsed
            while not self.execution_manager.is_order_transacted(oid) or security.get_accumulated_transcation_volume(execution_time) <= hide_in:
                sleep(0.005)
                
                # We don't want our limit order slipping down the book
                # if the price moves away from us
                if order_type == 'LIMIT':
                    qty_filled = self.execution_manager.get_order_filled_qty(oid)
                    self.execution_manager.pull_orders(self, [oid])

                    # Need to force market order if we're not getting any traction
                    if security.get_accumulated_transcation_volume(execution_time) > hide_in:
                        order_type = 'MARKET'
                        price = None

                        order = self.execution_manager.create_order(ticker, order_type, action, qty, price)
                        oid = self.execution_manager.execute_orders([order])[0]
                        break
                    else:
                        # Update the amount and chase the best price
                        best_bid, best_ask = security.get_best_bid_ask() 
                        order['price'] = best_bid if action == 'BUY' else best_ask
                        order['quantity'] -= qty_filled

                        oid = self.execution_manager.execute_orders([order])[0]
            
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
            :return is_profitable, hiding_volume
        """
        security = self.securities[ticker]
    
        best_bid, best_ask = security.get_best_bid_ask()

        # Compute the premium we will earn per share at current market price
        premium = best_bid - price if action == "BUY" else price - best_ask

        # Note this is the inverse, if they want us to BUY then we have to go and SELL
        # this qty in the market
        directional_qty = -1 * volume if action == "BUY" else volume
        
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

        # Basically i'm saying that the change in prices can be probabalistically bounded
        # by sqrt(t) * stdev(prices) where time is now volume buckets
        potential_adverse_price_change = math.sqrt(hiding_buckets) * self.indicators['vol_bucket_mid_price_std'] 

        is_profitable = premium - potential_adverse_price_change > 0
        
        return is_profitable, hiding_volume
    
    def compute_hiding_volume(ticker, volume, action, permanent_price_impact = 0.00001):
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

        # Note this is the inverse, if they want us to BUY then we have to go and SELL
        # this qty in the market
        directional_qty = -1 * volume if action == "BUY" else volume
        
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

    def signum(x):
        """ Returns the sign of a value
        :params x: Any real number
        :returns : sign(x)
        """
        if(x < 0):return -1
        elif(x > 0):return 1
        else:return 0

    def getOI(v,m,phi,vB,sigma,volSigma):
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

    def getBounds(m,phi,vB,sigma,volSigma,S_S,zLambda,k = 0):
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
        vB_l = float(signum(m)+1)/2-zLambda*sigma*abs(m)**0.5/ float(4*phi*(S_S+abs(m)*k)*volSigma**0.5)
        vB_u = float(signum(m)+1)/2+zLambda*sigma*abs(m)**0.5/ float(4*phi*(S_S+abs(m)*k)*volSigma**0.5)
        vB_z = (signum(m)*phi/float(phi-1)+1)/2.
        return vB_l,vB_u,vB_z

    def computeOptimalVolume(m,phi,vB,sigma,volSigma,S_S,zLambda,k = 0):
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

        vB_l,vB_u,vB_z = getBounds(m,phi,vB,sigma,volSigma,S_S,zLambda,k)

        # try alternatives
        if (2*vB-1)*abs(m)<m:
            v1 = (2*phi*((2*vB-1)*abs(m)-m)*(S_S+abs(m)*k)*volSigma**0.5 / float(zLambda*sigma))**(2./3)
            oi = getOI(v1,m,phi,vB,sigma,volSigma)
            if oi>0:
                if vB<= vB_u: return v1
                if vB>vB_u: return abs(m)

        elif (2*vB-1)*abs(m)>m:
            v2 = (2*phi*(m-(2*vB-1)*abs(m))*(S_S+abs(m)*k)*volSigma**0.5 / float(zLambda*sigma))**(2./3)
            oi = getOI(v2,m,phi,vB,sigma,volSigma)
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
    with TradingManager(['RITC','BEAR','BULL','USD','CAD']) as tm:
        
        for t in TradingTick(295):
            pass

if __name__ == '__main__':
    # install_thread_excepthook()
    main()
        