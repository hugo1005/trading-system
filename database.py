import sqlite3
from sqlite3 import Error
import pandas as pd
import time
import numpy as np
import math

class IterBook():
    """ Takes a list of bid and ask prices
    :param book_df: An orderbook dataframe or json file
    :return: None
    """
    def __init__(self, book_data, tick, ticker, action):
        self.book_data = book_data
        self.i = 0
        self.tick = tick
        self.ticker = ticker
        self.action = action

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.book_data):
            raise StopIteration

        next_item = self.book_data[self.i]
        self.i += 1

        return (self.i, self.tick, self.ticker, self.action, float(next_item[0]), float(next_item[1]))

class Database():
    def __init__(self, db_path='./datasets/hftoolkit_sqlite.db', setup_path='./configs/sql_config.txt', verbose=False):
        self.setup_path = setup_path
        self.db_path = db_path
        self.name = db_path.split('/')[-1]
        self.verbose = verbose

    def log(self, s):
        if self.verbose:
            print(s)

    def __enter__(self):
        if self.create_connection(self.db_path):
            self.run_setup_script(self.setup_path)
            self.log('[HFToolkit] Connected to db @ %s' % self.db_path)
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            # Always close db when finished
            self.log('[HFToolkit] Disconnected from db @ %s' % self.db_path)
            self.conn.commit()
            self.conn.close()

    def reconnect():
        self.conn.commit()
        self.conn.close()
        self.create_connection(self.db_path)
    
    def create_connection(self, db_file):
        """ create a database connection to the SQLite database specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
        except Error as e:
            print('[DATABASE CONNECTION] Error:')
            print(e)
        
        return self.conn
                
    def run_setup_script(self, script_path):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            f = open(script_path, 'r')
            setup_script = f.read()

            c = self.conn.cursor()
            c.executescript(setup_script)
        except (Error, IOError) as e:
            print('[DATABASE SETUP] Error:')
            print(e)

    def update_best_bid_ask(self,best_bid, best_ask):
        self.best_bid, self.best_ask = best_bid, best_ask
        self.midprice = (self.best_bid + self.best_ask) / 2

    def get_best_bid_ask(self,ticker, source):
        return self.best_bid, self.best_ask

    def update_tas(self, tas, ticker, source, stream=False):
        """ Updates the time and sales with new entries
        :param tas: Time and sales json object
        :return: None
        """
        c = self.conn.cursor()
        try:
            if source == 'BITSTAMP':
                if stream:
                    
                    # Only enter the relevant entries!
                    sql = """INSERT OR REPLACE INTO tas (tick, market_id, price, action, quantity, ticker) VALUES (?,?,?,?,?,?)
                    """
                    timestamp = float(tas['microtimestamp']) * (10**-6)
                    action = ('BUY' if int(tas['type']) == 0 else 'SELL')

                    c.execute(sql, (timestamp, tas['id'], tas['price'], 'BUY',tas['amount'], ticker))
                    # TODO: update book metrics for other data sources
                    sql = """UPDATE book_metrics SET market_orders = market_orders + 1 WHERE period_id = ? AND tick = ? AND ticker = ?"""
                    # sql = """INSERT OR REPLACE INTO book_metrics (period_id,tick,ticker,market_orders) VALUES (?,?,?,?)"""

                    # c.execute(sql, (0, math.ceil(timestamp), ticker, 1))
                    c.execute(sql, (0, math.floor(timestamp), ticker))
                else:
                    tas_df = pd.DataFrame(tas) # date, tid, price, type, amount
                    tas_df['type'] = tas_df['type'].replace(to_replace={0:'BUY',1:'SELL'})
                    tas_df['ticker'] = ticker

                    # Only enter the relevant entries!
                    sql = """INSERT OR REPLACE INTO tas (tick, market_id, price, action, quantity, ticker)
                    VALUES (?,?,?,?,?,?)
                    """

                    c.executemany(sql, tas_df.values)
            elif source == 'RIT':
                tas_df = pd.DataFrame(tas)
                tas_df['type'] = tas_df['type'] > self.midprice
                tas_df['type'] = tas_df['type'].replace(to_replace={0:'BUY',1:'SELL'})
                tas_df['ticker'] = ticker

                sql = """INSERT OR REPLACE INTO tas (market_id, period_id, tick, price,quantity, price, action, ticker)
                VALUES (?,?,?,?,?,?,?,?)
                """

                c.executemany(sql, tas_df.values)
            else:
                raise ValueError('[DATABASE] Unrecognised exchanged [%s] passed to update_tas!' % source)

            self.conn.commit()
        except Error as e:
            print('[DATABASE TAS] Error:')
            print(e)
    
    def update_book(self, book, ticker, source):
        """ Updates the orderbook history with new entries
        :param book: An orderbook json object
        :return: None
        """

        c = self.conn.cursor()
        try:
            if source == 'BITSTAMP':
                tick = float(book['microtimestamp']) * (10**-6)
                bids = book['bids']
                asks = book['asks']

                iter_bids = IterBook(bids, tick, ticker, 'BUY')
                iter_asks = IterBook(asks, tick, ticker, 'SELL')
                best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
                
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                bids_df = pd.DataFrame(bids, columns=['price','qty']).astype('float')
                asks_df = pd.DataFrame(asks, columns=['price','qty']).astype('float')
                bid_vol = bids_df.groupby('price').sum().sort_index(ascending=False).reset_index()
                ask_vol = asks_df.groupby('price').sum().sort_index(ascending=True).reset_index()
                
                # Orderbook imbalance for depths 1,2,3
                depth = lambda book, d: book['qty'][:d].sum()
                imbal = lambda bv, av, d: (depth(bv, d) - depth(av, d)) / (depth(bv, d) + depth(av, d))
                i1 = imbal(bid_vol, ask_vol, 1)
                i2 = imbal(bid_vol, ask_vol, 2)
                i3 = imbal(bid_vol, ask_vol, 3)

                # Orderbook depths (Used for measuring resilience to large orders)
                bid_d5, bid_d10, bid_d20 = depth(bid_vol, 5), depth(bid_vol, 10), depth(bid_vol, 20)
                ask_d5, ask_d10, ask_d20 = depth(ask_vol, 5), depth(ask_vol, 10), depth(ask_vol, 20)

                # A crude approximation to the slope of the book. A steeper slope corresponds to a greater level of informed trading
                elasticity = lambda v1, p1, v2, p2: ((math.log(v2)/math.log(v1)) - 1)/math.fabs((p2/p1) - 1)
                slope_bid = (elasticity(bid_d10, bids_df['price'][10], bid_d20, bids_df['price'][20]) + elasticity(bid_d5, bids_df['price'][5], bid_d10, bids_df['price'][10])) / 2
                slope_ask = (elasticity(ask_d10, asks_df['price'][10], ask_d20, asks_df['price'][20]) + elasticity(ask_d5, asks_df['price'][5], ask_d10, asks_df['price'][10])) / 2

                slope = (slope_bid + slope_ask) / 2

                # Only enter the relevant entries!
                sql = """INSERT OR REPLACE INTO book (order_id,tick,ticker,action,price,quantity)
                VALUES (?,?,?,?,?,?)
                """
                c.executemany(sql, iter_bids)
                c.executemany(sql, iter_asks)

                sql = """INSERT OR REPLACE INTO book_metrics (period_id,tick,ticker,imbalance_d1,imbalance_d2,imbalance_d3,bid_ask_spread,mid_price,bid_d5,bid_d10,bid_d20,ask_d5,ask_d10,ask_d20,slope)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """
                
                c.execute(sql, (0, math.floor(tick), ticker, i1, i2, i3, spread, mid_price, bid_d5,bid_d10,bid_d20,ask_d5,ask_d10,ask_d20,slope))

                self.update_best_bid_ask(best_bid, best_ask)

            elif source == 'RIT':
                # TODO: Update book_metrics as well!
                bids = pd.DataFrame(book['bid'])
                asks = pd.DataFrame(book['ask'])
                bids = bids.drop(columns=['type'])
                asks = asks.drop(columns=['type'])

                best_bid, best_ask = bids['price'][0], asks['price'][0]

                sql = """INSERT OR REPLACE INTO book (order_id,period_id,tick,trader_id,ticker,quantity,action,price,quantity_filled,vwap,status)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """

                c.executemany(sql, bids.values)
                c.executemany(sql, asks.values)

                self.update_best_bid_ask(best_bid, best_ask)
            else:
                raise ValueError('[DATABASE] Unrecognised exchanged [%s] passed to update_book!' % source)

            self.conn.commit()
        except Error as e:
            print('[DATABASE TAS] Error:')
            print(e)

    def create_security(self,ticker,exchange,limit_order_rebate=0,trading_fee=0,quoted_decimals=2,max_trade_size=10000,api_orders_per_second=10000,execution_delay_ms=10):
        """ create a new security or update an existing one
        :return: None
        """
        
        print('[DATABASE] Creating security [%s]' % ticker)

        c = self.conn.cursor()

        sql = """INSERT OR REPLACE INTO securities (ticker,exchange,limit_order_rebate,trading_fee,quoted_decimals,max_trade_size,api_orders_per_second,execution_delay_ms)
        VALUES (?,?,?,?,?,?,?,?)
        """

        c.execute(sql, (ticker,exchange,limit_order_rebate,trading_fee,quoted_decimals,max_trade_size,api_orders_per_second,execution_delay_ms))
        

