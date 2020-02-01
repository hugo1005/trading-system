import requests
import json
from time import sleep
from threading import Thread
from database import Database
# import websocket
# from stream import BitstampWebsocket

class API():

    def __init__(self, api_config, db_path, db_config_path, use_websocket=False):
        """ Initialises API which handles data capture and providing a unified api for frontend
        :param database: A database object for data capture
        :param api_config: A relative path to the configuration file
        :return: None
        """
        with open(api_config) as config_file:
            self.config = json.load(config_file)
        
        self.use_websocket = use_websocket
        self.stream = None
        
        self.db_path, self.db_config_path = db_path, db_config_path
    
    """
    Access Live Data
    """
    def get_live_book():
        pass

    def get_live_tas():
        pass
    
    def get_live_pos_limits():
        pass

    """
    Exexcute on RITC
    """
    def exec_order():
        pass

    def cancel_orders():
        pass


    """
    Access Stored data
    """
    def get_available_times(self, ticker, order_type):
         with Database(self.db_path, self.db_config_path) as db:
            print('[API] Fetching times from db...')
            c = db.conn.cursor()
            
            if order_type == 'limit':
                sql = 'SELECT DISTINCT tick FROM book WHERE ticker = ? ORDER BY tick ASC'
            elif order_type == 'market':
                sql = 'SELECT DISTINCT tick FROM tas WHERE ticker = ? ORDER BY tick ASC'
            
            c.execute(sql, (ticker,))
            times = c.fetchall()

            return [t[0] for t in times]

    def get_bid_ask_spread(self, ticker, t):
        with Database(self.db_path, self.db_config_path) as db:
            c = db.conn.cursor()
            
            sql_best_bid = 'SELECT MAX(price) FROM book WHERE ticker = ? AND tick = ? AND action = ?'
            c.execute(sql_best_bid, (ticker,t, "BUY"))
            best_bid = c.fetchone()[0]

            sql_best_ask = 'SELECT MIN(price) FROM book WHERE ticker = ? AND tick = ? AND action = ?'
            c.execute(sql_best_ask, (ticker,t, "SELL"))
            best_ask = c.fetchone()[0]

            sql_tick_size = 'SELECT quoted_decimals FROM securities WHERE ticker = ?'
            c.execute(sql_tick_size, (ticker,))
            tick_size = c.fetchone()[0]

            return {'spread': int((best_ask - best_bid) / tick_size), 'best_bid': best_bid, 'best_ask': best_ask, 'tick_size': tick_size}

    def get_orderbook_imbalance(self, ticker, t, depth):
        with Database(self.db_path, self.db_config_path) as db:
            c = db.conn.cursor()
            
            sql_bid = 'SELECT price, SUM(quantity) FROM book WHERE ticker = ? AND tick = ? AND action = ? GROUP BY price ORDER BY price DESC'
            c.execute(sql_bid, (ticker,t, "BUY"))
            # bid_vol = c.fetchall()[depth - 1][1] # zero indexing
            bid_vol = sum([res[1] for res in c.fetchall()[:depth]])

            sql_ask = 'SELECT price, SUM(quantity) FROM book WHERE ticker = ? AND tick = ? AND action = ? GROUP BY price ORDER BY price ASC'
            c.execute(sql_ask, (ticker,t, "SELL"))
            # ask_vol = c.fetchall()[depth - 1][1]
            ask_vol = sum([res[1] for res in c.fetchall()[:depth]])
            # print(bid_vol, ask_vol)
            return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def get_market_orders(self, ticker, t1, t2, depth):
        res = self.get_bid_ask_spread(ticker, t1)
        
        buy_price = res['best_bid'] - res['tick_size'] * (depth - 1) 
        sell_price = res['best_ask']  + res['tick_size'] * (depth - 1) 

        with Database(self.db_path, self.db_config_path) as db:
            c = db.conn.cursor()
            
            sql_bid = 'SELECT SUM(quantity) FROM tas WHERE ticker = ? AND tick > ? AND tick <= ? AND action = ? AND price = ?'
            c.execute(sql_bid, (ticker, t1, t2, "SELL", buy_price)) # Note: Must be a sell MO for BUY LO
            n_market_sells = c.fetchone()[0] # zero indexing
            n_market_sells = n_market_sells if n_market_sells != None else 0

            sql_ask = 'SELECT SUM(quantity) FROM tas WHERE ticker = ? AND tick > ? AND tick <= ? AND action = ? AND price = ?'
            c.execute(sql_ask, (ticker, t1, t2, "BUY", sell_price))
            n_market_buys = c.fetchone()[0]
            n_market_buys = n_market_buys if n_market_buys != None else 0

            return {'n_market_buys': n_market_buys, 'n_market_sells': n_market_sells}

    """
    Polling API's and data capture:
    """
    def get_source_config(self, source):
        endpoint = self.config['api_access'][source]['endpoint']
        headers = self.config['api_access'][source]['headers']

        return endpoint, headers

    def start(self):
        self.is_running = True
        self.polling_thread = Thread(target=self.poll)
        self.polling_thread.start()
    
    def shutdown(self):
        self.is_running = False

        print('[API] Stopping API Polling')
    
    def poll(self):
        with Database(self.db_path, self.db_config_path) as db:
            print('[API] Initialised API Polling with database [%s]' % db.name)
            
            wait_in_s = self.config['poll_params']['frequency_ms'] / 1000
            self.poll_securities(db)

            if self.use_websocket:
                self.stream = BitstampWebsocket(db)
                self.stream.start()
            else:
                while self.is_running:
                    print('Polling')
                    self.poll_once(db)
                    print('Polling Done')
                    sleep(wait_in_s)
            # while self.is_running:
            #     print('Polling')
            #     self.poll_once(db)
            #     print('Polling Done')
            #     sleep(wait_in_s)

    def poll_once(self, db):
        sources = self.config['poll_params']['sources']

        for source in sources:
            for ticker in sources[source]:
                book = self.poll_orderbook(ticker, source)
                tas = self.poll_time_and_sales(ticker, source)

                # Note we must always update the book first
                # Otherwise we may have missing midprice information
                # Which means we cannot filter buy / sell market orders accurately
                db.update_book(book, ticker, source)
                db.update_tas(tas, ticker, source)

    def poll_securities(self, db):
        sources = self.config['poll_params']['sources']

        for source in sources:
            for ticker in sources[source]:
                if source == 'BITSTAMP':
                    db.create_security(ticker, source)
                elif source == 'RIT':
                    endpoint, headers = self.get_source_config(source)
                    res = requests.get(endpoint + '/securities', params={'ticker': ticker}, headers = headers)

                    if res.ok:
                        spec = res.json()[0]
                        db.create_security(ticker,source,spec['limit_order_rebate'],spec['trading_fee'],spec['quoted_decimals'],spec['max_trade_size'],spec['api_orders_per_second'],spec['execution_delay_ms'])
                
    def poll_time_and_sales(self, ticker, source):
        # BTC [{date, tid, price, type = 0,1, amount}]
        # RIT [{id, period = 1, tick, price, quantity}]
        res = None
        endpoint, headers = self.get_source_config(source)

        if source == 'BITSTAMP':
            res = requests.get(endpoint + '/transactions/%s' % ticker, headers = headers) 
        elif source == "RIT":
            res = requests.get(endpoint + '/securities/tas', params={'ticker': ticker}, headers = headers)
        else:
            raise ValueError('[API] Unrecognised exchange [%s] passed to get_time_and_sales' % source)

        if res.ok:
            transactions = res.json() 

        return transactions 

    def poll_orderbook(self, ticker, source):
        # BTC {timestamp, bids:[[price,size]], asks:[[price,size]]}
        # RIT {bid: [{order_id, period, tick, trader_id, ticker, type, quantity, action, price, quantity filled, vwap, status}], ask: []}
        res = None
        endpoint, headers = self.get_source_config(source)

        if source == 'BITSTAMP':
            res = requests.get(endpoint + '/order_book/%s' % ticker, headers = headers)
        elif source == "RIT":
            res = requests.get(endpoint + '/securities/book', params={'ticker': ticker, 'limit': 1000}, headers = headers)
        else:
            raise ValueError('[API] Unrecognised exchange [%s] passed to get_orderbook' % source)
        
        if res:
            book = res.json()

        return book 


