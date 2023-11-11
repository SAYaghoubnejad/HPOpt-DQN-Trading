# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as GRU
from EncoderDecoderAgent.CNN.Train import Train as CNN
from EncoderDecoderAgent.CNN2D.Train import Train as CNN2d
from EncoderDecoderAgent.CNNAttn.Train import Train as CNN_ATTN
from EncoderDecoderAgent.CNN_GRU.Train import Train as CNN_GRU

# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL

# Imports Optimizer
from Optimizer.SimpleBayesianOptimizer import SimpleBayesianOptimizer

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np
from utils import save_pkl, load_pkl

DATA_LOADERS = {
    'BTC-USD': YahooFinanceDataLoader('BTC-USD',
                                      split_point='2022-01-01',
                                      load_from_file=False),

    'GOOGL': YahooFinanceDataLoader('GOOGL',
                                    split_point='2018-01-01',
                                    load_from_file=True),

    'AAPL': YahooFinanceDataLoader('AAPL',
                                   split_point='2018-01-01',
                                   begin_date='2010-01-01',
                                   end_date='2020-08-24',
                                   load_from_file=True),

    'DJI': YahooFinanceDataLoader('DJI',
                                  split_point='2016-01-01',
                                  begin_date='2009-01-01',
                                  end_date='2018-09-30',
                                  load_from_file=True),

    'S&P': YahooFinanceDataLoader('S&P',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'AMD': YahooFinanceDataLoader('AMD',
                                  split_point=2000,
                                  end_date='2018-09-25',
                                  load_from_file=True),

    'GE': YahooFinanceDataLoader('GE',
                                 split_point='2015-01-01',
                                 load_from_file=True),

    'KSS': YahooFinanceDataLoader('KSS',
                                  split_point='2018-01-01',
                                  load_from_file=True),

    'HSI': YahooFinanceDataLoader('HSI',
                                  split_point='2015-01-01',
                                  load_from_file=True),

    'AAL': YahooFinanceDataLoader('AAL',
                                  split_point='2018-01-01',
                                  load_from_file=True)
}


class SensitivityRun:
    def __init__(self,
                 dataset_name,
                 gamma,
                 batch_size,
                 replay_memory_size,
                 feature_size,
                 target_update,
                 n_episodes,
                 n_step,
                 window_size,
                 device,
                 evaluation_parameter='gamma',
                 transaction_cost=0):
        """

        @param data_loader:
        @param dataset_name:
        @param gamma:
        @param batch_size:
        @param replay_memory_size:
        @param feature_size:
        @param target_update:
        @param n_episodes:
        @param n_step:
        @param window_size:
        @param device:
        @param evaluation_parameter: shows which parameter are we evaluating and can be: 'gamma', 'batch size',
            or 'replay memory size'
        @param transaction_cost:
        """
        self.data_loader = DATA_LOADERS[dataset_name]
        self.train_data_last_price = self.data_loader.data_train_with_date.close[-1]
        self.dataset_name = dataset_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.feature_size = feature_size
        self.target_update = target_update
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.device = device
        self.evaluation_parameter = evaluation_parameter
        # The state mode is only for autoPatternExtractionAgent. Therefore, for pattern inputs, the state mode would be
        # set to None, because it can be recovered from the name of the data loader (e.g. dataTrain_patternBased).

        self.STATE_MODE_OHLC = 1
        self.STATE_MODE_CANDLE_REP = 4  # %body + %upper-shadow + %lower-shadow
        self.STATE_MODE_WINDOWED = 5  # window with k candles inside + the trend of those candles

        self.dataTrain_autoPatternExtractionAgent = None
        self.dataTest_autoPatternExtractionAgent = None
        self.dataTrain_patternBased = None
        self.dataTest_patternBased = None
        self.dataTrain_autoPatternExtractionAgent_candle_rep = None
        self.dataTest_autoPatternExtractionAgent_candle_rep = None
        self.dataTrain_autoPatternExtractionAgent_windowed = None
        self.dataTest_autoPatternExtractionAgent_windowed = None
        self.dataTrain_sequential = None
        self.dataTest_sequential = None
        self.dqn_pattern = None
        self.dqn_vanilla = None
        self.dqn_candle_rep = None
        self.dqn_windowed = None
        self.mlp_pattern = None
        self.mlp_vanilla = None
        self.mlp_candle_rep = None
        self.mlp_windowed = None
        self.cnn1d = None
        self.cnn2d = None
        self.gru = None
        self.deep_cnn = None
        self.cnn_gru = None
        self.cnn_attn = None
        self.experiment_path = os.path.join(os.getcwd(), 'Results/' + self.evaluation_parameter + '/')
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        self.reset()
        self.test_portfolios = {'DQN-pattern': {},
                                'DQN-vanilla': {},
                                'DQN-candlerep': {},
                                'DQN-windowed': {},
                                'MLP-pattern': {},
                                'MLP-vanilla': {},
                                'MLP-candlerep': {},
                                'MLP-windowed': {},
                                'CNN1d': {},
                                'CNN2d': {},
                                'GRU': {},
                                'Deep-CNN': {},
                                'CNN-GRU': {},
                                'CNN-ATTN': {}}

        self.train_portfolios = {'DQN-pattern': {},
                                'DQN-vanilla': {},
                                'DQN-candlerep': {},
                                'DQN-windowed': {},
                                'MLP-pattern': {},
                                'MLP-vanilla': {},
                                'MLP-candlerep': {},
                                'MLP-windowed': {},
                                'CNN1d': {},
                                'CNN2d': {},
                                'GRU': {},
                                'Deep-CNN': {},
                                'CNN-GRU': {},
                                'CNN-ATTN': {}}

    def reset(self):
        self.load_data()
        self.load_agents()

    def load_data(self):
        self.dataTrain_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                           self.STATE_MODE_OHLC,
                                           'action_auto_pattern_extraction',
                                           self.device,
                                           self.gamma,
                                           self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTest_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_OHLC,
                                           'action_auto_pattern_extraction',
                                           self.device,
                                           self.gamma,
                                           self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)
        self.dataTrain_patternBased = \
            DataForPatternBasedAgent(self.data_loader.data_train,
                                     self.data_loader.patterns,
                                     'action_pattern',
                                     self.device, self.gamma,
                                     self.n_step, self.batch_size,
                                     self.transaction_cost)

        self.dataTest_patternBased = \
            DataForPatternBasedAgent(self.data_loader.data_test,
                                     self.data_loader.patterns,
                                     'action_pattern',
                                     self.device,
                                     self.gamma,
                                     self.n_step,
                                     self.batch_size,
                                     self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_candle_rep = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_train,
                self.STATE_MODE_CANDLE_REP,
                'action_candle_rep',
                self.device,
                self.gamma, self.n_step, self.batch_size,
                self.window_size,
                self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_candle_rep = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_CANDLE_REP,
                                           'action_candle_rep',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                           self.STATE_MODE_WINDOWED,
                                           'action_auto_extraction_windowed',
                                           self.device,
                                           self.gamma, self.n_step,
                                           self.batch_size,
                                           self.window_size,
                                           self.transaction_cost)

        self.dataTrain_sequential = DataSequential(self.data_loader.data_train,
                                                   'action_sequential',
                                                   self.device,
                                                   self.gamma,
                                                   self.n_step,
                                                   self.batch_size,
                                                   self.window_size,
                                                   self.transaction_cost)

        self.dataTest_sequential = DataSequential(self.data_loader.data_test,
                                                  'action_sequential',
                                                  self.device,
                                                  self.gamma,
                                                  self.n_step,
                                                  self.batch_size,
                                                  self.window_size,
                                                  self.transaction_cost)

    def load_agents(self):
        self.dqn_pattern = DeepRL(self.data_loader,
                                  self.dataTrain_patternBased,
                                  self.dataTest_patternBased,
                                  self.dataset_name,
                                  None,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_vanilla = DeepRL(self.data_loader,
                                  self.dataTrain_autoPatternExtractionAgent,
                                  self.dataTest_autoPatternExtractionAgent,
                                  self.dataset_name,
                                  self.STATE_MODE_OHLC,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_candle_rep = DeepRL(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                     self.dataTest_autoPatternExtractionAgent_candle_rep,
                                     self.dataset_name,
                                     self.STATE_MODE_CANDLE_REP,
                                     self.window_size,
                                     self.transaction_cost,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.dqn_windowed = DeepRL(self.data_loader,
                                   self.dataTrain_autoPatternExtractionAgent_windowed,
                                   self.dataTest_autoPatternExtractionAgent_windowed,
                                   self.dataset_name,
                                   self.STATE_MODE_WINDOWED,
                                   self.window_size,
                                   self.transaction_cost,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step)

        self.mlp_pattern = SimpleMLP(self.data_loader,
                                     self.dataTrain_patternBased,
                                     self.dataTest_patternBased,
                                     self.dataset_name,
                                     None,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_vanilla = SimpleMLP(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent,
                                     self.dataTest_autoPatternExtractionAgent,
                                     self.dataset_name,
                                     self.STATE_MODE_OHLC,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_candle_rep = SimpleMLP(self.data_loader,
                                        self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                        self.dataTest_autoPatternExtractionAgent_candle_rep,
                                        self.dataset_name,
                                        self.STATE_MODE_CANDLE_REP,
                                        self.window_size,
                                        self.transaction_cost,
                                        self.feature_size,
                                        BATCH_SIZE=self.batch_size,
                                        GAMMA=self.gamma,
                                        ReplayMemorySize=self.replay_memory_size,
                                        TARGET_UPDATE=self.target_update,
                                        n_step=self.n_step)

        self.mlp_windowed = SimpleMLP(self.data_loader,
                                      self.dataTrain_autoPatternExtractionAgent_windowed,
                                      self.dataTest_autoPatternExtractionAgent_windowed,
                                      self.dataset_name,
                                      self.STATE_MODE_WINDOWED,
                                      self.window_size,
                                      self.transaction_cost,
                                      self.feature_size,
                                      BATCH_SIZE=self.batch_size,
                                      GAMMA=self.gamma,
                                      ReplayMemorySize=self.replay_memory_size,
                                      TARGET_UPDATE=self.target_update,
                                      n_step=self.n_step)

        self.cnn1d = SimpleCNN(self.data_loader,
                               self.dataTrain_autoPatternExtractionAgent,
                               self.dataTest_autoPatternExtractionAgent,
                               self.dataset_name,
                               self.STATE_MODE_OHLC,
                               self.window_size,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step)

        self.cnn2d = CNN2d(self.data_loader,
                           self.dataTrain_sequential,
                           self.dataTest_sequential,
                           self.dataset_name,
                           self.feature_size,
                           self.transaction_cost,
                           BATCH_SIZE=self.batch_size,
                           GAMMA=self.gamma,
                           ReplayMemorySize=self.replay_memory_size,
                           TARGET_UPDATE=self.target_update,
                           n_step=self.n_step,
                           window_size=self.window_size)

        self.gru = GRU(self.data_loader,
                       self.dataTrain_sequential,
                       self.dataTest_sequential,
                       self.dataset_name,
                       self.transaction_cost,
                       self.feature_size,
                       BATCH_SIZE=self.batch_size,
                       GAMMA=self.gamma,
                       ReplayMemorySize=self.replay_memory_size,
                       TARGET_UPDATE=self.target_update,
                       n_step=self.n_step,
                       window_size=self.window_size)

        self.deep_cnn = CNN(self.data_loader,
                            self.dataTrain_sequential,
                            self.dataTest_sequential,
                            self.dataset_name,
                            self.transaction_cost,
                            BATCH_SIZE=self.batch_size,
                            GAMMA=self.gamma,
                            ReplayMemorySize=self.replay_memory_size,
                            TARGET_UPDATE=self.target_update,
                            n_step=self.n_step,
                            window_size=self.window_size)

        self.cnn_gru = CNN_GRU(self.data_loader,
                               self.dataTrain_sequential,
                               self.dataTest_sequential,
                               self.dataset_name,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step,
                               window_size=self.window_size)

        self.cnn_attn = CNN_ATTN(self.data_loader,
                                 self.dataTrain_sequential,
                                 self.dataTest_sequential,
                                 self.dataset_name,
                                 self.transaction_cost,
                                 self.feature_size,
                                 BATCH_SIZE=self.batch_size,
                                 GAMMA=self.gamma,
                                 ReplayMemorySize=self.replay_memory_size,
                                 TARGET_UPDATE=self.target_update,
                                 n_step=self.n_step,
                                 window_size=self.window_size)

    def train(self):
        self.dqn_pattern.train(self.n_episodes)
        self.dqn_vanilla.train(self.n_episodes)
        self.dqn_candle_rep.train(self.n_episodes)
        self.dqn_windowed.train(self.n_episodes)
        self.mlp_pattern.train(self.n_episodes)
        self.mlp_vanilla.train(self.n_episodes)
        self.mlp_candle_rep.train(self.n_episodes)
        self.mlp_windowed.train(self.n_episodes)
        self.cnn1d.train(self.n_episodes)
        self.cnn2d.train(self.n_episodes)
        self.gru.train(self.n_episodes)
        self.deep_cnn.train(self.n_episodes)
        self.cnn_gru.train(self.n_episodes)
        self.cnn_attn.train(self.n_episodes)

    def evaluate_sensitivity(self):
        key = None
        if self.evaluation_parameter == 'gamma':
            key = self.gamma
        elif self.evaluation_parameter == 'batch size':
            key = self.batch_size
        elif self.evaluation_parameter == 'replay memory size':
            key = self.replay_memory_size
        else:
            key = f'G: {self.gamma}, BS: {self.batch_size}, RMS: {self.replay_memory_size}, n: {self.n_step}, episodes: {self.n_episodes}'

        self.test_portfolios['DQN-pattern'][key] = self.dqn_pattern.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['DQN-vanilla'][key] = self.dqn_vanilla.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['DQN-candlerep'][
            key] = self.dqn_candle_rep.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['DQN-windowed'][key] = self.dqn_windowed.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['MLP-pattern'][key] = self.mlp_pattern.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['MLP-vanilla'][key] = self.mlp_vanilla.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['MLP-candlerep'][
            key] = self.mlp_candle_rep.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed'][key] = self.mlp_windowed.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['CNN1d'][key] = self.cnn1d.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['CNN2d'][key] = self.cnn2d.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['GRU'][key] = self.gru.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['Deep-CNN'][key] = self.deep_cnn.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU'][key] = self.cnn_gru.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()
        self.test_portfolios['CNN-ATTN'][key] = self.cnn_attn.test(initial_investment=self.train_data_last_price).get_daily_portfolio_value()

        self.train_portfolios['DQN-pattern'][key] = self.dqn_pattern.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['DQN-vanilla'][key] = self.dqn_vanilla.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['DQN-candlerep'][
            key] = self.dqn_candle_rep.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['DQN-windowed'][key] = self.dqn_windowed.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['MLP-pattern'][key] = self.mlp_pattern.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['MLP-vanilla'][key] = self.mlp_vanilla.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['MLP-candlerep'][
            key] = self.mlp_candle_rep.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['MLP-windowed'][key] = self.mlp_windowed.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['CNN1d'][key] = self.cnn1d.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['CNN2d'][key] = self.cnn2d.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['GRU'][key] = self.gru.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['Deep-CNN'][key] = self.deep_cnn.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['CNN-GRU'][key] = self.cnn_gru.test(test_type='train').get_daily_portfolio_value()
        self.train_portfolios['CNN-ATTN'][key] = self.cnn_attn.test(test_type='train').get_daily_portfolio_value()

    def average_return(self):
        self.avg_returns = {}
        for model_name in self.train_portfolios.keys():
            for gamma in self.train_portfolios[model_name]:
                self.avg_returns[model_name] = (self.train_portfolios[model_name][gamma][-1] - self.train_portfolios[model_name][gamma][0]) \
                * 100 / self.train_portfolios[model_name][gamma][0]
        return self.avg_returns

    def plot_and_save_sensitivity(self, data_set='test'):
        data = self.train_portfolios if data_set == 'train' else self.test_portfolios

        portfolio_plot_path = os.path.join(self.experiment_path, f'plots/portfolio/on_{data_set}')
        if not os.path.exists(portfolio_plot_path):
            os.makedirs(portfolio_plot_path)

        sns.set(rc={'figure.figsize': (20, 10)})
        sns.set_palette(sns.color_palette("Paired", 15))

        for model_name in data.keys():
            first = True
            ax = None
            for gamma in data[model_name]:
                profit_percentage = [
                    (data[model_name][gamma][i] - data[model_name][gamma][0]) /
                    data[model_name][gamma][0] * 100
                    for i in range(len(data[model_name][gamma]))]
                
                prediction_df = None
                if data_set == 'test':
                    difference = len(data[model_name][gamma]) - len(self.data_loader.data_test_with_date)
                    prediction_df = pd.DataFrame({'date': self.data_loader.data_test_with_date.index,
                                    'portfolio': profit_percentage[difference:]})
                elif data_set == 'train':
                    difference = len(data[model_name][gamma]) - len(self.data_loader.data_train_with_date)
                    prediction_df = pd.DataFrame({'date': self.data_loader.data_train_with_date.index,
                                    'portfolio': profit_percentage[difference:]})
                    
                if not first:
                    prediction_df.plot(ax=ax, x='date', y='portfolio', label=gamma)
                else:
                    ax = prediction_df.plot(x='date', y='portfolio', label=gamma)
                    first = False
            if ax == None:
                continue       
            ax.set(xlabel='Time', ylabel='%Rate of Return')
            ax.set_title(f'Tuning Hyperparameters of {model_name} using {self.evaluation_parameter}')
            plt.legend()
            fig_file = os.path.join(portfolio_plot_path, f'{model_name}.jpg')
            plt.savefig(fig_file, dpi=300)

    def plot_and_save_return(self):
        prediction_plot_path = os.path.join(self.experiment_path, 'plots/prediction')
        if not os.path.exists(prediction_plot_path):
            os.makedirs(prediction_plot_path)
        
        sns.set(rc={'figure.figsize': (20, 10)})

        for model_name in self.test_portfolios.keys():

            train_df = pd.DataFrame(self.data_loader.data_train_with_date.close, index=self.data_loader.data.index)
            test_df = pd.Series(self.data_loader.data_test_with_date.close, index=self.data_loader.data.index)
            ax2 = train_df.plot(label='Train')
            test_df.plot(ax=ax2, color='r', label='Test')
            for gamma in self.test_portfolios[model_name]:
                prediction_df = pd.Series(self.test_portfolios[model_name][gamma], index=self.data_loader.data.index[-len(self.test_portfolios[model_name][gamma]):])
                prediction_df = prediction_df.reindex(self.data_loader.data.index, fill_value=np.nan)
                prediction_df.plot(ax=ax2, label=gamma)

            ax2.set(xlabel='Time', ylabel='Close Price')
            ax2.set_title(f'Train, Test and Prediction of model {model_name} on dataset {self.dataset_name}')
            plt.legend()
            fig_file = os.path.join(prediction_plot_path, f'{model_name}.jpg')
            plt.savefig(fig_file, dpi=300)


    def save_portfolios(self):
        path = os.path.join(self.experiment_path, 'portfolios.pkl')
        save_pkl(path, self.test_portfolios)

    def save_experiment(self):
        self.plot_and_save_sensitivity(data_set='test')
        self.plot_and_save_sensitivity(data_set='train')
        self.plot_and_save_return()
        self.save_portfolios()

iter = 1
init_set = 2
optimizer_name = 'Simple BO'

# gamma, log2(batch_size), log2(replay_memory_size), log2(n_step), n_episodes / 10
bounds = torch.tensor([[0.4, 3.0, 3.0, 1.0, 1.0], [1.0, 9.0, 9.0, 6.0, 6.0]]) 
types = [torch.float64, torch.int64, torch.int64, torch.int64, torch.int64]

n_step = 8
window_size = 3
dataset_name = "BTC-USD"
n_episodes = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on:', device)
feature_size = 64
target_update = 5

gamma_default = 0.9
batch_size_default = 16
replay_memory_size_default = 32

run = SensitivityRun(
    dataset_name,
    gamma_default,
    batch_size_default,
    replay_memory_size_default,
    feature_size,
    target_update,
    n_episodes,
    n_step,
    window_size,
    device,
    evaluation_parameter=optimizer_name,
    transaction_cost=0)

def objective_func(params):
    run.gamma = round(params[0].item(), 2)
    run.batch_size = 2 ** params[1].to(torch.int32).item()
    run.replay_memory_size = 2 ** params[2].to(torch.int32).item()
    run.n_step = 2 ** params[3].to(torch.int32).item()
    run.n_episodes = 10 * params[4].to(torch.int32).item()
    run.reset()
    run.train()
    run.evaluate_sensitivity()
    eval = torch.max(torch.tensor(list(run.average_return().values()), dtype=torch.float64))
    eval = torch.tensor(5)
    return eval

optimizer = SimpleBayesianOptimizer(objective_func, bounds, types)
optimizer.simple_BO(n_steps=iter, n_init_points=init_set)

run.save_experiment()

path = os.path.join(os.getcwd(), f'run/{optimizer_name}/')
if not os.path.exists(path):
    os.makedirs(path)
save_pkl(os.path.join(path, 'run.pkl'), run)
save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)