from sources import API

API_CONFIG = './configs/api_config.json'
SQL_CONFIG = './configs/sql_config.txt'
DB_PATH = './datasets/hftoolkit_sqlite.db'

class System():
    def __init__(self, api_config = './configs/api_config.json', db_path = './datasets/hftoolkit_sqlite.db', db_config_path = './configs/sql_config.txt', poll=True, read_only=False):

        print('[HFToolkit] Started...')
        self.api_config, self.db_path, self.db_config_path = api_config, db_path, db_config_path
        self.poll = poll
        # backend_thread = Thread(target=self.init_backend)
        # backend_thread.start()

        if read_only:
             self.api = API(self.api_config, self.db_path, self.db_config_path, use_websocket=False)
    
    def __enter__(self):
        self.api = API(self.api_config, self.db_path, self.db_config_path, use_websocket=True)
        
        if self.poll:
            self.api.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.api.shutdown()

def main():
    with System() as sys:
        while True:
            pass

if __name__ == "__main__":
    main()