from time import sleep, time
from threading import Thread
import requests
import itertools

import pandas as pd
import numpy as np
import math
import scipy.stats as st

import warnings
warnings.filterwarnings('ignore')

import sys
import os
clear = lambda: os.system('cls')

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

class ExecutionManager():
    def __init__(self, api, tickers, securities):
        self.is_running = False
        self.api = api
        self.endpoint, self.headers = self.api.get_source_config('RIT')
        self.tickers = tickers
        self.securities = securities

        order_properties = ["order_id", "period" "tick", "trader_id", "ticker","type","quantity","action","price","quantity_filled","vwap","status"]

        """" Order Management Variables """
        self.orders = {
            'OPEN': pd.DataFrame(columns=order_properties),
            'CANCELLED':  pd.DataFrame(columns=order_properties),
            'TRANSACTED':  pd.DataFrame(columns=order_properties)
        }

        """" Risk Management (Per Ticker) """
        self.net_positions = {ticker:0 for ticker in tickers}
        self.net_market_making_positions = {ticker:0 for ticker in tickers}
        self.market_making_orders = []
        
        """" Risk Limits """
        res = requests.get(self.endpoint + '/limits', headers=self.headers)

        if res.ok:
            # For some reason it returns a list rather than single dict
            limits = res.json()[0] 
            self.gross_limit = limits['gross_limit']
            self.net_limit = limits['net_limit']
        else:
            print('[Execution Manager] Error could not obtain position limits from API!')

    """ Order Execution """
    def accept_tender(self, tender):
        accept_res = requests.post(self.endpoint + '/tenders/%s' % tender['tender_id'], headers=self.headers)

        if accept_res.ok:
            print('[Tenders] Accepted : price: %s qty: %s action: %s' % (tender['price'],
            tender['quantity'], tender['action']))
            print(accept_res.json())

            # Assuming tender orders don't show up like regular orders
            # So we must account for that here
            ticker = 'RITC'
            qty = tender['quantity']
            direction = 1 if tender['action'] == "BUY" else -1
            qty_directional = qty * direction

            # Note Tender is always on the ETF which is worth 2x the underlying with respect to risk limits
            self.net_positions[ticker] += 2 * qty_directional
            
            # Just to be sure we don't accidentally front run a tender
            sleep(0.05)
        else:
            print('[AcceptTenders] Could not reach API with code %s : %s' % (accept_res.status_code, accept_res.json()))

    def decline_tender(self, tender):
        accept_res = requests.delete(self.endpoint + '/tenders/%s' % tender['tender_id'], headers=self.headers)

        if accept_res.ok:
            print('[Tenders] Declined Tender : price: %s qty: %s action: %s' % (tender['price'],
            tender['quantity'], tender['action']))
        else:
            print('[AcceptTenders] Could not reach API with code %s' % accept_res.status_code)

    def create_order(self, ticker, order_type, action, qty, price=None):
        return {'ticker': ticker, 'type': order_type, 'action': action, 'quantity': qty, 'price': price}

    def split_order(self, order, max_qty):
        """If the order is larger than the trade size limit it splits it up into multiple
        smaller orders of maximal size
        :param order: the order which is too large
        :param max_qty: the maximum size of the order for the specific security
        :return a list of smaller orders
        """
        print("[SPLITTER] Order was too large to submit, splitting: %s" % order)
        original_size = order['quantity']
        num_new_orders = math.ceil(original_size / max_qty)
        size_new_orders = original_size / num_new_orders

        new_order = self.create_order(order['ticker'], order['type'], order['action'], size_new_orders, order['price'])

        return [new_order] * num_new_orders
         

    def execute_orders(self, orders, source):
        """
        Sends orders to the RIT API, handles any POST request rate limiting by the API
        :params orders: List of json objects as specified in create_order method
        :param source: The source of activity (MARKET_MAKER, ARBITRAGE, TENDER)
        :return order_ids: returns a list of executed order ids
        """
        executed_orders = []
        
        if self.can_execute_orders(orders):
            # print(["[ExecManager] Executing Orders: %s" % orders])
            # API is rate limited to 5 orders per second  
            while len(orders) > 0:
                order = orders.pop()
                max_qty = self.securities[order['ticker']].max_trade_size
                
                if order['quantity'] > max_qty:
                   orders += self.split_order(order, max_qty)
                   continue

                res = requests.post(self.endpoint + '/orders', params=order, headers = self.headers)
                content = res.json()

                if res.ok:
                    # print('[Trading Manager] Order placed: [Price: %s, Qty: %s Type: %s Ticker: %s ]' % (order['price'],order['quantity'],order['type'],order['ticker']))

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

                    if source == 'MARKET_MAKER':
                        self.market_making_orders.append(content['order_id'])

                elif res.status_code == 429:
                    # Try again after wait time
                    print('Error occured processing order')
                    print(res.json())
                    sleep(content['wait'] + 0.01)
                    orders.append(order)
                else:
                    print(res.json())
        
        print("[Execution] Executed orders: %s" % [order['order_id'] for order in executed_orders])
        return [order['order_id'] for order in executed_orders]
    
    def pull_orders(self, order_ids):
        """
        Pulls specified open orders from the book.
        :param order_ids: A list of order id's, this facilitates cancelling of specific orders. Note we do not facilitate this to be left undefined to ensure
        the good practice of tracking open orders.
        """
        # Ensures orders have been enqueued to the book 
        sleep(0.25)
        # Clear console prints 
        # clear() 
        
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
        # print("[Execution] Order Id: %s Orders Transacted: %s" % (order_id, transacted_ids))
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

        ticker = order_details['ticker'].iloc[0]
        qty = order_details['quantity_filled'].iloc[0]
        direction = 1 if order_details['action'].iloc[0] == "BUY" else -1
        qty_directional = qty * direction if ticker != 'RITC' else 2 * qty * direction

        if order_id in self.market_making_orders:
            self.net_market_making_positions[ticker] += qty_directional

        self.net_positions[ticker] += qty_directional
    
    """ Hedging Logic """
    def hedge_position(self, source):
        """
        Currently doesn't handle the currency risk, but will get rid of excess
        net net_positionsoriginating from a particular source activity
        :param source: one of (MAKRET_MAKER, ARBITRAGE, TENDER)
        """
        if source == "MARKET_MAKER":
            for ticker in self.net_market_making_positions:
                net_pos = self.net_market_making_positions[ticker]
                action = 'BUY' if net_pos < 0 else 'SELL'
                
                # TODO: Later we will refactor with optimal limit + market order combinations
                # once we sort out tender orders
                if net_pos != 0:
                    order = self.create_order(ticker, 'MARKET', action, net_pos)
                    self.execute_orders([order], 'MARKET_MAKER')
                    print('[Hedging] Market Making Positions Hedged')
        if source == "TENDER" or source == "ARBITRAGE":
            
            net_currency_exposure = self.securities['USD'].position
            print('[Hedging] Hedging Currency Expsoure: $%s' % net_currency_exposure)
            action = 'BUY' if net_currency_exposure < 0 else 'SELL'
            
            if net_currency_exposure != 0:
                    order = self.create_order('USD', 'MARKET', action, abs(net_currency_exposure))
                    self.execute_orders([order], 'MARKET_MAKER')
                    # print('[Hedging] Hedging Currency Expsoure')
 
    
   
    """ Order Fill Monitoring """

    def start(self):
        self.is_running = True
        self.polling_thread = Thread(target=self.poll)
        self.polling_thread.start()

    def shutdown(self):
        self.is_running = False

    def poll(self):
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
        res = requests.get(self.endpoint + '/orders', params={'status': status}, headers = self.headers)
        
        if res.ok:
            orders = res.json()

            if len(orders) > 0:
                updated_orders = pd.DataFrame(orders)
                updated_orders = updated_orders.rename(columns={'id':'order_id'})
                # Handles updating of net positions (Cancelled order may have been partially filled!)
                if status == 'TRANSACTED' or status == 'CANCELLED':
                    new_transactions_ids = list(set(updated_orders['order_id'].values).difference(set(self.orders[status]['order_id'].values)))
                    
                    self.orders[status] = pd.DataFrame(orders)

                    for oid in new_transactions_ids:
                        self.update_net_position(oid)
                else:
                    self.orders[status] = pd.DataFrame(orders)
        else:
            print('[ExecutionManager] Polling response failed with code: %s' % res.status_code)

class OptimalTenderExecutor:
    def __init__(self, execution_manager, ticker, num_large_orders = 3, num_proceeding_small_orders = 10,
     large_to_small_order_size_ratio = 5, vpin_threshold=0.6, risk_aversion=0.005):
        self.execution_manager = execution_manager
        self.api = self.execution_manager.api
        self.ticker = ticker

        self.volume_transacted = 0
        self.net_position = 0
        self.hiding_volume = 0
        self.net_action = 'BUY'

        self.num_large_orders = num_large_orders
        self.num_proceeding_small_orders = num_proceeding_small_orders
        self.large_to_small_order_size_ratio = large_to_small_order_size_ratio
        self.vpin_threshold = vpin_threshold
        self.risk_aversion = risk_aversion

        # Compute Trading Params
        self.compute_execution_params()

        # Start Trading Thread
        self.can_execute = True
        self.optimal_exec_thread = Thread(target=self.optimally_execute)
        self.optimal_exec_thread.start() 

    def stop(self):
        self.can_execute = False

    def add_tender_order(self, tender):
        """ 
        Evaluates the profitability and risk of a tender order. If it is profitable it offsets
        the total quantity left to be hedged and proceeds to hedge remainder in separate thread.
        :param tender: The tender order in JSON as provided by the API
        """
        qty = tender['quantity']
        # Calculate Remaining Volume
        remaining_net_position = np.sign(self.net_position) * (abs(self.net_position) - self.volume_transacted)

        # Every time a new order is added to execute it offsets the total amount remaining to be hedged
        additional_directional_qty = qty if tender['action'] == 'BUY' else -1 * qty

        # The amount remaining to be hedged
        new_net_position = remaining_net_position + additional_directional_qty
        
        # TODO: Fix this - The tender quantity is no longer an accurate measure of risk
        #  The risk of this is much lower as it will be offset by an existing outstanding tender
        #  Positions
        print('[Tenders] Evaluating Tender: %s' % tender)

        is_profitable, self.hiding_volume = self.process_tender_order('RITC',
                        tender['quantity'], tender['action'], tender['price'])
        
        fake_order = self.execution_manager.create_order('RITC', 'TENDER', tender['action'], tender['quantity'], tender['price'])
        is_within_risk = self.execution_manager.can_execute_orders([fake_order])
        
        print('[Tenders] Is within risk: %s Is profitable: %s hiding volume: %s' % (is_within_risk, is_profitable, self.hiding_volume))

        if is_profitable and is_within_risk:
            self.execution_manager.accept_tender(tender)
            
             # Update the position we need to disspose of and reset the volume transacted to meet this new target
            self.volume_transacted = 0
            self.net_position = new_net_position
            self.net_action = 'BUY' if self.net_position < 0 else 'SELL'

            # Compute Trading Params
            self.compute_execution_params()
        else:
            self.execution_manager.decline_tender(tender)
    
    def compute_execution_params(self):
            # Compute the size of a small volume unit
            # Note a large trade is "large_to_small_order_size_ratio" trading units
            self.volume = abs(self.net_position)
            self.num_trading_units = self.num_large_orders * (self.large_to_small_order_size_ratio + self.num_proceeding_small_orders)
            self.trading_unit_volume = self.volume / self.num_trading_units
            self.trading_unit_hiding_volume = self.hiding_volume / self.num_trading_units

            # Define the sequence of order quantities and their corresponding hiding quantities
            self.order_qty_seq = ([self.trading_unit_volume * self.large_to_small_order_size_ratio] + ([self.trading_unit_volume] * self.num_proceeding_small_orders)) * self.num_large_orders

            self.order_hiding_volume_seq = ([self.trading_unit_hiding_volume * self.large_to_small_order_size_ratio] + ([self.trading_unit_hiding_volume] * self.num_proceeding_small_orders)) * self.num_large_orders
            
            self.order_idx = 0

            print("[Optimal Executor] Params Updated - Net Position: %s Volume: %s Action: %s Transacted: %s" % ( self.net_position, self.volume, self.net_action, self.volume_transacted))

    def optimally_execute(self):
        for t in TradingTick(295, self.api):
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

            if not self.can_execute:
                break

            if self.volume_transacted < self.volume:
                print("[Tender] Pct of Qty Hedged: %.2f" % (self.volume_transacted/self.volume))
                qty = self.order_qty_seq[self.order_idx]
                hide_in = self.order_hiding_volume_seq[self.order_idx]
                security = self.execution_manager.securities[self.ticker]

                # Determine order type based on current VPIN
                # If VPIN is high we go to market because someone else 
                # is quite likely to adversely select us.
                vpin = security.indicators['VPIN']
                
                if vpin > self.vpin_threshold:
                    order_type = 'MARKET'
                    price = None
                else:
                    order_type = 'LIMIT'
                    best_bid, best_ask = security.get_best_bid_ask()         
                    price = best_bid if action == 'BUY' else best_ask

                order = self.execution_manager.create_order(self.ticker, order_type, self.net_action, qty, price)
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

                            order = self.execution_manager.create_order(self.ticker, order_type, self.net_action, qty, price)
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
                self.volume_transacted += qty
                self.order_idx += 1 
    

    

    def process_tender_order(self, ticker, volume, action, price):
        """Evaluates a tender based on computing the optimal volume to conceal the requested order
            and the potential 
            :param ticker: A string ticker / symbol (Always going to be RITC ETF)
            :param volume: The size of the order requested on the secuirty by tender
            :param action: The direction in which we are obliged to BUY / SELL the security and volume requested by the tender.
            :return is_tradable, hiding_volume
        """
        security = self.execution_manager.securities[ticker]
    
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
    
    def compute_hiding_volume(self, ticker, volume, action, permanent_price_impact = 0.01):
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
        security = self.execution_manager.securities[ticker]
    
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
    
class OptimalArbitrageExecutor:
    def __init__(self, execution_manager, tickers, trading_size, num_large_orders = 3, num_proceeding_small_orders = 10,
     large_to_small_order_size_ratio = 5, vpin_threshold=0.6, risk_aversion=0.005):
        self.execution_manager = execution_manager
        self.api = self.execution_manager.api

        # Optimality Parameters
        self.num_large_orders = num_large_orders
        self.num_proceeding_small_orders = num_proceeding_small_orders
        self.large_to_small_order_size_ratio = large_to_small_order_size_ratio
        self.vpin_threshold = vpin_threshold
        self.risk_aversion = risk_aversion
        self.trading_size = trading_size

        # Live Trade Execution tracking
        self.tracking = {}

        # Start Trading Thread
        self.can_execute = True

        for ticker in tickers:
            self.tracking[ticker] = {
                'volume_transacted': 0,
                'net_position': 0,
                'hiding_volume': 0,
                'net_action': 'BUY',
                'thread': Thread(target=self.optimally_execute, args=(ticker))
            }

            # Compute Trading Params
            self.compute_execution_params(ticker)

            # Start Optimally Trading
            self.tracking[ticker]['thread'].start()
        
    def stop(self):
        self.can_execute = False

    def close_arbitrage_position(self):
        orders = []
        
        for ticker in self.tracking:
            track = self.tracking[ticker]
            closing_action = 'SELL' if track['net_action'] == 'BUY' else 'BUY'
            qty = track['volume_transacted']
            
            if qty > 0:
                order = self.execution_manager.create_order(ticker, 'MARKET', closing_action, qty)
                orders.append(order)

        self.execution_manager.execute_orders(orders, 'ARBITRAGE')

    def open_arbitrage_position(self, cointegration_coeff, leg_1_dir, leg_2_dir):
        """ 
        Opens positions on prespecified hardcoded securities
        """
        bear = self.tracking['BEAR']
        bull = self.tracking['BULL']
        ritc = self.tracking['RITC']

        #  Todo close out any net positions with market orders
        self.close_arbitrage_position()

        for ticker in self.tracking:
            qty = self.trading_size if ticker == 'RITC' else self.trading_size / 2
            action = leg_2_dir if ticker == 'RITC' else leg_1_dir
            directional_qty = qty * -1 if action == 'SELL' else qty
            
            track = self.tracking[ticker] 

            track['volume_transacted'] = 0
            track['net_position'] = directional_qty
            track['hiding_volume'] = self.compute_hiding_volume(ticker, trade_size, action)
            track['net_action'] = action

            # Compute Trading Params
            self.compute_execution_params(ticker)
    
    def compute_execution_params(self, ticker):
            # Compute the size of a small volume unit
            # Note a large trade is "large_to_small_order_size_ratio" trading units
            track = self.tracking[ticker]
            track['volume'] = abs(track['net_position'])
            self.num_trading_units = self.num_large_orders * (self.large_to_small_order_size_ratio + self.num_proceeding_small_orders)
            track['trading_unit_volume'] = track['volume'] / self.num_trading_units
            track['trading_unit_hiding_volume'] = track['hiding_volume'] / self.num_trading_units

            # Define the sequence of order quantities and their corresponding hiding quantities
            track['order_qty_seq'] = ([track['trading_unit_volume'] * self.large_to_small_order_size_ratio] + ([track['trading_unit_volume']] * self.num_proceeding_small_orders)) * self.num_large_orders

            track['order_hiding_volume_seq'] = ([track['trading_unit_hiding_volume'] * self.large_to_small_order_size_ratio] + (track['trading_unit_hiding_volume'] * self.num_proceeding_small_orders)) * self.num_large_orders
            
            track['order_idx'] = 0

            print("[Arbitrage Optimal Executor] Params Updated [%s] - Net Position: %s Volume: %s Action: %s Transacted: %s" % (ticker, track['net_position'], track['volume'], track['net_action'], track['volume_transacted']))

    def optimally_execute(self, ticker):
        track = self.tracking[ticker]

        for t in TradingTick(295, self.api):
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

                if not self.can_execute:
                    break

                if track['volume_transacted'] < track['volume']:
                    print("[Tender] Pct of Qty Hedged: %.2f" % (self.volume_transacted/self.volume))
                    qty = track['order_qty_seq'][track['order_idx']]
                    hide_in = track['order_hiding_volume_seq'][track['order_idx']]
                    security = self.execution_manager.securities[ticker]

                    # Determine order type based on current VPIN
                    # If VPIN is high we go to market because someone else 
                    # is quite likely to adversely select us.
                    vpin = security.indicators['VPIN']
                    
                    if vpin > self.vpin_threshold:
                        order_type = 'MARKET'
                        price = None
                    else:
                        order_type = 'LIMIT'
                        best_bid, best_ask = security.get_best_bid_ask()         
                        price = best_bid if action == 'BUY' else best_ask

                    order = self.execution_manager.create_order(ticker, order_type, track['net_action'], qty, price)
                    oid = self.execution_manager.execute_orders([order], source='ARBITRAGE')[0]
                    
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

                                order = self.execution_manager.create_order(ticker, order_type, track['net_action'], qty, price)
                                oid = self.execution_manager.execute_orders([order], source='ARBITRAGE')[0]
                                break
                            else:
                                # Update the amount and chase the best price
                                best_bid, best_ask = security.get_best_bid_ask() 
                                order['price'] = best_bid if action == 'BUY' else best_ask
                                order['quantity'] -= qty_filled
                            
                                oid = self.execution_manager.execute_orders([order], source='ARBITRAGE')[0]

                        transacted = self.execution_manager.is_order_transacted(oid)
                        accumulated_volume = security.get_accumulated_transcation_volume(execution_time)
                    
                    # Move onto the next order in the sequence
                    track['volume_transacted'] += qty
                    track['order_idx'] += 1 
    

    

    def process_tender_order(self, ticker, volume, action, price):
        """Evaluates a tender based on computing the optimal volume to conceal the requested order
            and the potential 
            :param ticker: A string ticker / symbol (Always going to be RITC ETF)
            :param volume: The size of the order requested on the secuirty by tender
            :param action: The direction in which we are obliged to BUY / SELL the security and volume requested by the tender.
            :return is_tradable, hiding_volume
        """
        security = self.execution_manager.securities[ticker]
    
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
    
    def compute_hiding_volume(self, ticker, volume, action, permanent_price_impact = 0.01):
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
        security = self.execution_manager.securities[ticker]
    
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