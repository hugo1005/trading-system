# Instructions
If you are running the algorithmic case simply
python3 algo.py will do the trick -> Make sure to use the ritc_2020_only folder.

sources.py handles API connections
security.py handles polling of time and sales + orderbook data and providing relevant limits and indicator calculations
execution.py handles all execution of trades, pulling orders and managing risk / hedging
algo.py holds the main trading logic and thread initilisation.
configs folder contains all necessary config parameters for API

The remaining code files handle data capture and storage for a later date

Contributors:
hugo1005 (Trading System)
j0shward (Volatility)

