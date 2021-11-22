# https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/FinRL_demo_docker.ipynb
#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

# %matplotlib inline
from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.neo_finrl.data_processor import DataProcessor


from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
    
# from config.py start_date is a string
config.START_DATE = '2020-01-01'

# from config.py end_date is a string
config.END_DATE = '2021-11-01'

print(config.DOW_30_TICKER)
df = YahooDownloader(start_date = '2020-01-01',
                     end_date = '2021-11-01',
                     ticker_list = config.DOW_30_TICKER).fetch_data()

#%%
df.shape
df.sort_values(['date','tic'],ignore_index=True).head()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)

processed.sort_values(['date','tic'],ignore_index=True).head(10)
#%%
train = data_split(processed, '2020-01-01','2021-09-01')
trade = data_split(processed, '2021-09-01','2021-11-01')
print(len(train))
print(len(trade))

train.head()
trade.head()
config.TECHNICAL_INDICATORS_LIST


#%%
from StockTradingEnvV2 import StockTradingEnvV2


print(StockTradingEnvV2.__doc__)
#%%
information_cols = ['open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'rsi_30', 'cci_30', 'dx_30', 'turbulence']

e_train_gym = StockTradingEnvV2(df = train, 
                              hmax = 100, 
                              out_of_cash_penalty=-1e6,
                              daily_information_cols = information_cols,
                              print_verbosity = 500)

#%%
# for this example, let's do multiprocessing with n_cores-2

import multiprocessing

n_cores = multiprocessing.cpu_count() - 2
# n_cores = 12
print(f"using {n_cores} cores")


# env_train, _ = e_train_gym.get_multiproc_env(n = n_cores)
env_train, _ = e_train_gym.get_sb_env()

#%%
# agent = DRLAgent(env = env_train)
# model_a2c = agent.get_model("a2c")
# trained_a2c = agent.train_model(model=model_a2c, 
#                              tb_log_name='a2c',
#                              total_timesteps=50000)


agent = DRLAgent(env = env_train)
print(config.PPO_PARAMS)
from torch.nn import Softsign, ReLU
ppo_params ={'n_steps': 128, 
             'ent_coef': 0.01, 
             'learning_rate': 0.00025, 
             'batch_size': 256, 
            'gamma': 0.99}

policy_kwargs = {
#     "activation_fn": ReLU,
    "net_arch": [1024, 1024, 1024], 
#     "squash_output": True
}

model = agent.get_model("ppo",  model_kwargs = ppo_params, policy_kwargs = policy_kwargs, verbose = 0)
# model.load("quicksave_ppo_dow.model")

model.learn(total_timesteps = 400000, 
            log_interval = 1, tb_log_name = 'ppo_1024_5_more_ooc_penalty',
            reset_num_timesteps = True)

# print(e_train_gym.actions_memory[:2])
model.save("quicksave_ppo_dow.model")
data_turbulence = processed[(processed.date<'2021-09-01') & (processed.date>='2020-01-01')]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])
insample_turbulence.turbulence.describe()

turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)


turbulence_threshold



#%%

def DRL_prediction(model, environment):
    test_env, test_obs = environment.get_sb_env()
    """make a prediction"""
    account_memory = []
    actions_memory = []
    test_env.reset()
    for i in range(len(environment.df.index.unique())):
        action, _states = model.predict(test_obs)
        #account_memory = test_env.env_method(method_name="save_asset_memory")
        #actions_memory = test_env.env_method(method_name="save_action_memory")
        test_obs, rewards, dones, info = test_env.step(action)
        if not dones[0]:
          account_memory = test_env.env_method(method_name="save_asset_memory")
          actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("hit end!")
            break
    return account_memory[0], actions_memory[0]

trade = data_split(processed, '2021-09-01','2021-11-01')
e_trade_gym = StockTradingEnvV2(df = trade,hmax = 10, 
                              daily_information_cols = information_cols,
                              print_verbosity = 500)

df_account_value, df_actions = DRL_prediction(model=model,
                        environment = e_trade_gym,)


df_account_value.shape

df_account_value.head(50)

df_actions.to_dict(orient = 'rows')[:3]


#%%

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

#%%

print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value, 
             baseline_ticker = '^DJI', 
             baseline_start = df_account_value.date.values[0],
             baseline_end = df_account_value.date.values[-1], value_col_name = 'total_assets')

#%%

#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')