from time import sleep, time
from threading import Thread
import requests
import itertools

import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore')

class Security:
    def __init__(self, ticker, api, poll_delay=0.01, is_currency=False):
        self.endpoint, self.headers = api.get_source_config('RIT')
        self.ticker = ticker
        self.init_params()
        self.is_running = False
        self.poll_delay = poll_delay
        self.is_currency = is_currency
        self.init_time = time()

        self.indicators = {}
        self.book_history = pd.DataFrame(columns=['type','price','quantity'])
        self.tas_history = pd.DataFrame(columns=['id','type','price','quantity'])
        self.best_bid_ask = pd.DataFrame(columns=['best_bid','best_ask', 'midprice'])
        self.position = 0

    def start(self):
        print('[Security:%s] Started Polling...' % self.ticker)
        self.is_running = True
        self.polling_thread = Thread(target=self.poll, name='Thread Security [%s]' % self.ticker)
        self.polling_thread.start()

    def shutdown(self):
        self.is_running = False

    def init_params(self):
        res = requests.get(self.endpoint + '/securities', params={'ticker': self.ticker}, headers = self.headers)

        if res.ok:
            spec = res.json()[0]
            self.lo_rebate = spec['limit_order_rebate']
            self.mo_fee = spec['trading_fee']
            self.quoted_decimals = spec['quoted_decimals']
            self.max_trade_size = spec['max_trade_size']
            self.max_orders_per_second = spec['api_orders_per_second']
            self.execution_delay_ms = spec['execution_delay_ms']

            print('[%s] Max Trade Size: %s' % (self.ticker, self.max_trade_size))
        else:
            print('[%s] Parameters could not be found!' % self.ticker)

    """ ------- API Polling -----------"""
    def poll(self):
        
        # Time and sales
        res_tas = requests.get(self.endpoint + '/securities/tas', params={'ticker': self.ticker}, headers = self.headers)

        # Orderbook
        res_book = requests.get(self.endpoint + '/securities/book', params={'ticker': self.ticker, 'limit': 1000}, headers = self.headers)
        
        # print('Updating... %s' % self.ticker, res_book.ok, res_tas.ok)
        if res_book.ok and res_tas.ok:
            book = res_book.json()
            tas = res_tas.json()
            # print('[SECURITY %s] Fetching from API' % self.ticker)
            self.update_book_tas(book, tas)

            if time() - self.init_time >= 20:
                # print('[SECURITY %s] Updating Indicators' % self.ticker)
                self.recompute_indicators()
    
    def update_book_tas(self, book, tas):
        # Extract bid ask data from order book response
        bids = pd.DataFrame(book['bids'])
        asks = pd.DataFrame(book['asks'])
    
        bids = bids.drop(columns=['type'])
        asks = asks.drop(columns=['type'])
        best_bid, best_ask = bids['price'][0], asks['price'][0]
        midprice = (best_bid + best_ask) / 2
        timestamp = time()

        # Update security book history for bid and ask
        bids['type'] = 'bid'
        asks['type'] = 'ask'
        bids['timestamp'] = pd.to_datetime(timestamp, unit='s')
        asks['timestamp'] = pd.to_datetime(timestamp, unit='s')
        bids = bids.set_index('timestamp')
        asks = asks.set_index('timestamp')
        self.book_history = pd.concat([self.book_history, bids, asks])

        # Update best bid and ask
        new_best_bid_ask = pd.DataFrame({'best_bid': best_bid, 'best_ask':best_ask, 'midprice': midprice}, index=[pd.to_datetime(timestamp, unit='s')])
        self.best_bid_ask = pd.concat([self.best_bid_ask, new_best_bid_ask])
        
        # Parse time and sales data 
        # Sometimes for currencies there may be no transactions 
        if len(tas) > 0:
            tas_df = pd.DataFrame(tas)
            tas_df['type'] = tas_df['price'] > midprice
            tas_df['type'] = tas_df['type'].replace(to_replace={0:'BUY',1:'SELL'})

            # We attempt to create a more high resolution timestamp than the tick 
            # provided by ritc, we will be accurate to the +-self.poll_delay
            existing_ids = self.tas_history['id'].unique()
            tas_new_entries = tas_df[~tas_df['id'].isin(existing_ids)]
            tas_new_entries['timestamp'] = pd.to_datetime(timestamp, unit='s')

            self.tas_history = pd.concat([self.tas_history, tas_new_entries[['id','timestamp', 'type','price','quantity']].set_index('timestamp')])

    def recompute_indicators(self):
        # TODO: Change these hyperparmaters appropriately to correct volumes
        bucket_size = 10
        vol = self.compute_historical_volatility()
        
        res = requests.get(self.endpoint + '/securities', params={'ticker':self.ticker}, headers=self.headers)
        if res.ok:
            self.position = res.json()[0]['position']
        else:
            print('[Indicators] Could not reach API! %s' % res.json())

        if self.is_currency == False:
            # print('[SECURITY %s] Computing VPIN...' % self.ticker)
            vpin_results = self.compute_historical_vpin(self.tas_history, BAR_SIZE='10ms', N_BUCKET_SIZE=bucket_size, SAMPLE_LENGTH=12)
            # print("VPIN: %s" % vpin_results)
            self.indicators['VPIN'] = vpin_results['VPIN'].values[-1]
            self.indicators['order_imbalance'] = vpin_results['imbalance'].values[-1]
            self.indicators['std_price_changes'] = vpin_results['std_price_changes'].values[-1]
            self.indicators['std_price_changes_volume'] = bucket_size
            self.indicators['vol_bucket_avg_duration'] = vpin_results['bucket_duration'].mean()
            self.indicators['vol_bucket_mid_price_std'] = vpin_results['bucket_mid_prices'].std()
        
        self.indicators['volatility'] = vol

    """ ------- Security State -----------"""
    def get_midprice(self):
        return self.best_bid_ask['midprice'].iloc[-1]

    def get_bid_ask_spread(self):
        best_bid = self.best_bid_ask['best_bid'].iloc[-1]
        best_ask = self.best_bid_ask['best_ask'].iloc[-1]

        return best_ask - best_bid

    def get_best_bid_ask(self):
        best_bid = self.best_bid_ask['best_bid'].iloc[-1]
        best_ask = self.best_bid_ask['best_ask'].iloc[-1]
        
        return best_bid, best_ask

    def get_net_returns(self, lookback_seconds):
        lookback_datetime = pd.to_datetime(time() - lookback_seconds)
        returns = self.best_bid_ask[self.best_bid_ask.index >= lookback_datetime]['midprice'].pct_change().resample('1s').sum()

        return returns.sum()

    def get_accumulated_transcation_volume(self, start_timestamp):
        return self.tas_history[self.tas_history.index > start_timestamp]['quantity'].sum()

    def get_ohlc_history(self, freq='100ms'):
        """
        Gets open high low close history for security
        :param freq: frequency of time aggregation
        :returns: open high low close values aggregated at the specified frequency
        """
        # print("BID ASK DF")
        # print(self.best_bid_ask[['timestamp']])
        resample = self.best_bid_ask['midprice'].resample(freq)
        ohlc = resample.agg(['first','max','min','last'])

        ohlc.columns = ['o','h','l','c']
        return ohlc
    
    def get_midprice_history(self, freq='100ms'):
        """
        Gets open high low close history for security
        :param freq: frequency of time aggregation
        :returns: open high low close values aggregated at the specified frequency
        """
        # print("BID ASK DF")
        # print(self.best_bid_ask[['timestamp']])
        resample = self.best_bid_ask['midprice'].resample(freq).mean()

        return resample

    def get_average_slippage(self):
        return (self.best_bid_ask['best_ask'] - self.best_bid_ask['best_bid']).mean()
    
    """ ------- Computing VPIN -------- """
    
    def standardise_order_qty(self, df, target):
        """Takes orders of any given volume and splits them up into orders of single units.
        We do this as order volume is often misleading as large orders are often split into smaller orders
        anyway. So it is better to standardise the size of the order to 100 and then bucket them later
            :param df: contains the data to be standardised
            :param target: the name of the column thats being standardised
            :return: Dataframe of results
        """
        expanded = []
        # print('df')
        # print(df)

        filter_vol =  (df['volume'] == 0 )| (df['volume'].isna())
        df_filtered =  df[~filter_vol]

        timestamps = df_filtered.index.values
        vols = df_filtered['volume'].values
        targets = df_filtered[target].values
        # print('df_filtered')
        # print(df_filtered)
        # print('targets')
        # print(targets)
        for row in zip(timestamps, vols, targets):
            timestamp, vol, std_value = row
            expanded += [(timestamp, std_value)] * math.ceil(vol / 500)

        return pd.DataFrame.from_records(expanded, index=0)

    
    def compute_historical_vpin(self, all_trades,BAR_SIZE='1min',N_BUCKET_SIZE=10,SAMPLE_LENGTH=12):
        """Estimates VPIN  (Volume Synchronised Probaility of Informed Trading). Apoologies if this code is a little
        poorly written, I lifted out of a github with a working VPIN calulcation to ensure I didnt mis interpret anything
        in the original paper
            :param all_trades: contains time and sales information in a data frame
            :param BAR_SIZE: minimum time aggregation for time and sales data
            :param N_BUCKET_SIZE: the volume in each VPIN calculation
            :param SAMPLE_LENGTH: the smoothing factor for rolling average of VPIN
            :returns {"VPIN": vpin_df, "trades_adj": trades_adj, "pct_changes_df": pct_changes_df, "std_price_changes": std_price_changes, 'bucket_duration':diffs, 'bucket_mid_prices': bucket_mid_prices}
        """
        usd_trades = all_trades.copy() 
        # print(usd_trades.tail())
        usd_trades['type'] = usd_trades['type'].replace(to_replace={'BUY':1,'SELL':0}, regex=True)
        # print('[VPIN %s] Num Buys: %s Num Sells: %s' % (self.ticker, usd_trades['type'].sum(), (usd_trades['type'].shape[0]-usd_trades['type'].sum()) ))
        typestr = usd_trades[['type']]
        volume = usd_trades[['quantity']]
        trades = usd_trades[['price']]
        
        # print('[VPIN %s] Total Quantity Traded: %s' % (self.ticker, volume.sum()))
        # print('[VPIN %s] Average Price: %s' % (self.ticker, trades.mean()))
        
        # Aggregates Volume and Price information to BAR_SIZE frequency
        # assign trade sign to 1 minute time bar by averaging buys and sells and taking the more common one
        # HUGO: Facilitates "bulk classification" of trade direction over 1 min in Probablistic terms
        trades_resampled = trades.resample(BAR_SIZE).sum()
        
        trades_1min = trades_resampled.pct_change().fillna(0)
        price_changes_1min = trades_resampled.diff().fillna(0)
        volume_1min = volume.resample(BAR_SIZE).sum()
        typestr_1min = typestr.resample(BAR_SIZE).mean().round()
        # print('[VPIN %s] STR Num Buys: %s Num Sells: %s' % (self.ticker, typestr_1min['type'].sum(), (typestr_1min['type'].shape[0]-typestr_1min['type'].sum()) ))
        
        df = pd.concat([typestr_1min, volume_1min], axis=1)
        df_trades = pd.concat([volume_1min, trades_1min], axis=1)
        df.columns = ['type', 'volume']
        df_trades.columns = ['volume', 'price_delta_pct']

        volume_agg_direction = df 
        price_delta_agg = df_trades

        # HUGO: Recall we take each 1 minute volume grouping and split it up into minimum size transaction units
        # HUGO: This ensures that we make no assumptions regarding how large orders may be submitted over time
        expanded = self.standardise_order_qty(df, 'type')
        std_returns_dist = self.standardise_order_qty(df_trades, 'price_delta_pct')

        std_order_direction = expanded
        # print('[VPIN %s] Expanded prices length: %s' % (self.ticker, std_returns_dist.shape[0]))
        # print(expanded[1].value_counts())
        # --------------- find single-period VPIN ---------------------------
        def grouper(n, iterable):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, n))
                if not chunk:
                    return
                yield chunk
        
        OI = []
        OI_signed = []
        start = 0 
        
        for each in grouper(N_BUCKET_SIZE, std_order_direction[1]):
            slce = pd.Series(each)
            counts = slce.value_counts()
            # print('[VPIN %s] Order Type Cuunts in Bucket: [%s]' % (self.ticker, counts))
            if len(counts) > 1:
                OI_signed.append((counts[1] - counts[0])/N_BUCKET_SIZE)
                OI.append(np.abs(counts[1] - counts[0])/N_BUCKET_SIZE)
            else:
                if 0 in counts:
                    OI_signed.append((-1*counts[0])/N_BUCKET_SIZE)
                    OI.append(counts[0]/N_BUCKET_SIZE)
                else:
                    OI_signed.append((counts[1])/N_BUCKET_SIZE)
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
                bucket_mid_prices.append(trades_resampled[start_idx:idx].mean())
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
        # print("Length of OI: %s" % len(OI))
        # print("Sample Length: %s" % SAMPLE_LENGTH)
        vpin_df = pd.Series(OI[:-1], index=mid_buckets).rolling(SAMPLE_LENGTH).mean()
        oi_df = pd.Series(OI_signed[:-1], index=mid_buckets).rolling(SAMPLE_LENGTH).mean()
        # print(vpin_df.tail())
        # Unused:
        # trades_adj = trades.resample(BAR_SIZE).sum().reindex_like(vpin_df, method='ffill')
        pct_changes_df = pd.Series(pct_changes, index=mid_buckets)
        std_price_changes = pd.Series(price_changes, index=mid_buckets).rolling(SAMPLE_LENGTH).std()
        # print("VPIN COMPUTED [%s]" % self.ticker)
        return {"VPIN": vpin_df,"imbalance":oi_df, "pct_changes_df": pct_changes_df, "std_price_changes": std_price_changes, 'bucket_duration':pd.Series(diffs), 'bucket_mid_prices':pd.Series(bucket_mid_prices)}
    
    """ ------- Computing Volatility ----- """
    def compute_historical_volatility(self):
        return self.best_bid_ask['midprice'].std()

class Options(Security):

    def __init__(self, ticker, api, poll_delay=0.01, is_currency=False):
        super().__init__( ticker, api,poll_delay=0.01,is_currency=is_currency) #calls all of the arguments from the super class 'Security'

        self.strike = int(str(self.ticker)[-2:-1])
        self.maturity = int(str(self.ticker)[3]) / 12
        self.option_type = str(self.ticker)[4]

    """___________________Vanilla Option Pricer________________________"""

    def vanilla(self, S, K, T, r, sigma,ticker, option = 'C',):

        S = self.get_midprice()
        K = self.strike
        T = self.maturity
        option = self.option_type
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option == 'C':
            result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        if option == 'P':
            result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        return result

    def option_disect(self,ticker):
        S = self.get_midprice()
        K = self.strike
        T = self.maturity
        option = self.option_type

        return S, K, T, option

    """___________________Newton Raphson Implied Volatility Calculator________________________"""

    def nr_imp_vol(self,S, K, T, f, r, sigma,ticker, option = 'C' ):   
    
        #S: spot price
        #K: strike price
        #T: time to maturity
        #f: Option value
        #r: interest rate
        #sigma: volatility of underlying asset
        #option: where it is a call or a put option

        S = self.securities['RTM'].get_midprice()
        K = self.strike
        T = self.maturity
        option = self.option_type
        f = self.securities[ticker].get_midprice()
        sigma = self.case['RTM'].get_forecast() #not made yet as not sure where forecasted vol will be

        
        d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option == 'C':
            fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - f
            vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
            
        if option == 'P':
            fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - f
            vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
        
        tolerance = 0.000001 #limit of margin accepted for newton raphson algorithm
        x0 = sigma #we take our known 
        xnew  = x0
        xold = x0 - 1
            
        while abs(xnew - xold) > tolerance:
        
            xold = xnew
            xnew = (xnew - fx - f) / vega
            
            return abs(xnew)

    def vol_forecast(self):

        news = requests.get(self.endpoint + '/news', params={'limit':1}, headers=self.headers)
        if news.ok:
            body = news.json()[0]['body'] #call the body of the news article

            if body[4] == 'l': #'the latest annualised' - direct figure
                sigma = int(body[-3:-2])/100

            elif body[4] == 'a': #'the annualized' - expectation of range
                sigma = (int(body[-26:-25]) + int(body[-32:-31]))/200
                
            else: sigma = 0.2

        else:
            print('[Indicators] Could not reach API! %s' % res.json())
            sigma = 0.2

        return sigma