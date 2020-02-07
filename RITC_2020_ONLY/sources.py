import requests
import json
from time import sleep
from threading import Thread
from database import Database
# import websocket
# from stream import BitstampWebsocket

class API():

    def __init__(self, api_config, db_path='', db_config_path='', use_websocket=False):
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
    Polling API's and data capture:
    """
    def get_source_config(self, source):
        endpoint = self.config['api_access'][source]['endpoint']
        headers = self.config['api_access'][source]['headers']

        return endpoint, headers

