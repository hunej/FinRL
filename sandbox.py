#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime

from finrl.apps import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from finrl.trade.backtest import backtest_strat, baseline_strat

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
    
#%%
# Download and save the data in a pandas DataFrame:
df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2020-12-01',
                     ticker_list = config.DOW_30_TICKER).fetch_data()
#%%
# Perform Feature Engineering:
df = FeatureEngineer(df.copy(),
                    use_technical_indicator=True,
                    use_turbulence=False).preprocess_data()


# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  covs = return_lookback.cov().values 
  cov_list.append(covs)
  
df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
df.head()    
#%%
stock_dimension = len(train.tic.unique())
state_space = stock_dimension
# Initialize env:
env_setup = EnvSetup(stock_dim = stock_dimension,
                        state_space = state_space,
                        initial_amount = 1000000,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST)
                        
env_train = env_setup.create_env_training(data = train,
                                          env_class = StockPortfolioEnv)    