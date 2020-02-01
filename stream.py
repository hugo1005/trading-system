# -*- coding: utf-8 -*-
import websocket
import json
import ssl
from functools import partial
from threading import Thread

class BitstampWebsocket():
    def __init__(self, database, endpoint='wss://ws.bitstamp.net'):
        self.db = database
        self.endpoint = endpoint

    def on_message(self, ws, data):
        res = json.loads(data)
        channel_name = res['channel']
        # print('called %s' % channel_name)
        if res['event'] == 'bts:request_reconnect':
            self.start()
        elif res['event'] != 'bts:subscription_succeeded':
            if channel_name == 'order_book_btcusd':
                print('[WS] Updating orderbook')
                self.db.update_book(res['data'], 'btcusd', 'BITSTAMP')
            elif channel_name == 'live_trades_btcusd':
                print('[WS] Updating tas')
                self.db.update_tas(res['data'], 'btcusd', 'BITSTAMP', True)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("### closed ###")
        self.start()

    def on_open(self, ws):
        
        params = {
            'event': 'bts:subscribe',
            'data': {
                'channel': 'order_book_btcusd'
            }
        }
        sub = json.dumps(params)
        ws.send(sub)

        params = {
            'event': 'bts:subscribe',
            'data': {
                'channel': 'live_trades_btcusd'
            }
        }
        sub = json.dumps(params)
        ws.send(sub)

        print("[WS] Subscribed to data stream")

    def start(self):
        websocket.enableTrace(True)
        host = self.endpoint
        ws = websocket.WebSocketApp(host,
                                    on_message=partial(self.on_message),
                                    on_error=partial(self.on_error),
                                    on_close=partial(self.on_close))
        ws.on_open = partial(self.on_open)
        ws.run_forever(sslopt={'cert_reqs': ssl.CERT_NONE})

    """ Old code """
    # def start(self):
    #     open = partial(self.on_open)
    #     msg = partial(self.on_message)
    #     err = partial(self.on_error)
    #     close = partial(self.on_close)
    #     # msg1 = partial(self.on_message, channel_name='book')
    #     # open2 = partial(self.on_open, channel_name='tas')
    #     # msg2 = partial(self.on_message, channel_name='tas')
    #     print("[WS] Connecting...")
    #     self.ws = websocket.WebSocketApp(self.endpoint, on)
    #     self.on_open = open
    #     self.on_message = msg
    #     self.on_error = err
    #     self.ws.run_forever(sslopt={'cert_reqs': ssl.CERT_NONE})
    #     # self.ws2 = websocket.WebSocketApp(self.endpoint, on_open=open2, on_message=msg2, on_error=self.on_error)
        
    #     # self.ws2.run_forever(sslopt={'cert_reqs': ssl.CERT_NONE})

    # def subscribe_data(self, ws):
    #     params = {
    #         'event': 'bts:subscribe',
    #         'data': {
    #             'channel': 'order_book_btcusd'
    #         }
    #     }
    #     sub = json.dumps(params)
    #     ws.send(sub)

    #     params = {
    #         'event': 'bts:subscribe',
    #         'data': {
    #             'channel': 'live_trades_btcusd'
    #         }
    #     }
    #     sub = json.dumps(params)
    #     ws.send(sub)

    #     print("[WS] Subscribed to data stream")

    # def on_open(self, ws):
    #     self.subscribe_data(ws)
        

    # def on_message(self, ws, data):
    #     res = json.loads(data)
    #     channel_name = res['channel']
    #     print(res['event'])
    #     # print('called %s' % channel_name)
    #     if res['event'] == 'bts:request_reconnect':
    #         self.start()
    #     elif res['event'] != 'bts:subscription_succeeded':
    #         if channel_name == 'order_book_btcusd':
    #             # print('[WS] Updating orderbook')
    #             self.db.update_book(res['data'], 'btcusd', 'BITSTAMP')
    #         elif channel_name == 'live_trades_btcusd':
    #             # print('[WS] Updating tas')
    #             self.db.update_tas(res['data'], 'btcusd', 'BITSTAMP', True)

    # def on_error(self, ws, msg):
    #     print(msg)

    # def on_close(self):
    #     self.start()

        
            

