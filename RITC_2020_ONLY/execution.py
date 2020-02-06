from time import sleep, time
from threading import Thread
import requests
import itertools

import pandas as pd
import numpy as np
import math

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
        if source == "TENDER":
            
            net_currency_exposure = self.securities['USD'].position
            print('[Hedging] Hedging Currency Expsoure: $%s' % net_currency_exposure)
            action = 'BUY' if net_currency_exposure < 0 else 'SELL'

            if net_currency_exposure != 0:
                    order = self.create_order('USD', 'MARKET', action, abs(net_currency_exposure))
                    self.execute_orders([order], 'MARKET_MAKER')
                    print('[Hedging] Hedging Currency Expsoure')
 
    
   
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