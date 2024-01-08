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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
import os
from utils import save_pkl, load_pkl, set_random_seed
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

DATA_LOADERS = {
    'BTC-USD': YahooFinanceDataLoader(
        'BTC-USD',
        split_point='2021-01-01',
        validation_split_point='2023-01-01',
        load_from_file=True
    ),

    'GOOGL': YahooFinanceDataLoader(
        'GOOGL',
        split_point='2018-01-01',
        validation_split_point='2018-01-01',
        load_from_file=True
    ),

    'AAPL': YahooFinanceDataLoader(
        'AAPL',
        split_point='2018-01-01',
        validation_split_point='2018-01-01',
        begin_date='2010-01-01',
        end_date='2020-08-24',
        load_from_file=True
    ),

    'DJI': YahooFinanceDataLoader(
        'DJI',
        split_point='2016-01-01',
        validation_split_point='2018-01-01',
        begin_date='2009-01-01',
        end_date='2018-09-30',
        load_from_file=True
    ),

    'S&P': YahooFinanceDataLoader(
        'S&P',
        split_point=2000,
        validation_split_point=2000,
        end_date='2018-09-25',
        load_from_file=True
    ),

    'AMD': YahooFinanceDataLoader(
        'AMD',
        split_point=2000,
        validation_split_point=2000,
        end_date='2018-09-25',
        load_from_file=True
    ),

    'GE': YahooFinanceDataLoader(
        'GE',
        split_point='2015-01-01',
        validation_split_point='2015-01-01',
        load_from_file=True
    ),

    'KSS': YahooFinanceDataLoader(
        'KSS',
        split_point='2018-01-01',
        validation_split_point='2018-01-01',
        load_from_file=True
    ),

    'HSI': YahooFinanceDataLoader(
        'HSI',
        split_point='2015-01-01',
        validation_split_point='2015-01-01',
        load_from_file=True
    ),

    'AAL': YahooFinanceDataLoader(
        'AAL',
        split_point='2018-01-01',
        validation_split_point='2018-01-01',
        load_from_file=True
    )
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
        self.experiment_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            'BaseModel/Results/' + self.evaluation_parameter + '/')
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        self.reset()
        self.test_portfolios = {
            'DQN-pattern': {},
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
        
        self.train_portfolios = {
            'DQN-pattern': {},
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

        self.validation_portfolios = {
            'DQN-pattern': {},
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
            DataAutoPatternExtractionAgent(
                self.data_loader.data_train,
                self.STATE_MODE_OHLC,
                'action_auto_pattern_extraction',
                self.device,
                self.gamma,
                self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataTest_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_test,
                self.STATE_MODE_OHLC,
                'action_auto_pattern_extraction',
                self.device,
                self.gamma,
                self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataValidation_autoPatternExtractionAgent = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_validation,
                self.STATE_MODE_OHLC,
                'action_auto_pattern_extraction',
                self.device,
                self.gamma,
                self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataTrain_patternBased = \
            DataForPatternBasedAgent(
                self.data_loader.data_train,
                self.data_loader.patterns,
                'action_pattern',
                self.device, self.gamma,
                self.n_step, self.batch_size,
                self.transaction_cost)

        self.dataTest_patternBased = \
            DataForPatternBasedAgent(
                self.data_loader.data_test,
                self.data_loader.patterns,
                'action_pattern',
                self.device,
                self.gamma,
                self.n_step,
                self.batch_size,
                self.transaction_cost)

        self.dataValidation_patternBased = \
            DataForPatternBasedAgent(
                self.data_loader.data_validation,
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
            DataAutoPatternExtractionAgent(
                self.data_loader.data_test,
                self.STATE_MODE_CANDLE_REP,
                'action_candle_rep',
                self.device,
                self.gamma, self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataValidation_autoPatternExtractionAgent_candle_rep = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_validation,
                self.STATE_MODE_CANDLE_REP,
                'action_candle_rep',
                self.device,
                self.gamma, self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_train,
                self.STATE_MODE_WINDOWED,
                'action_auto_extraction_windowed',
                self.device,
                self.gamma, self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataTest_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_test,
                self.STATE_MODE_WINDOWED,
                'action_auto_extraction_windowed',
                self.device,
                self.gamma, self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataValidation_autoPatternExtractionAgent_windowed = \
            DataAutoPatternExtractionAgent(
                self.data_loader.data_validation,
                self.STATE_MODE_WINDOWED,
                'action_auto_extraction_windowed',
                self.device,
                self.gamma, self.n_step,
                self.batch_size,
                self.window_size,
                self.transaction_cost)

        self.dataTrain_sequential = DataSequential(
            self.data_loader.data_train,
            'action_sequential',
            self.device,
            self.gamma,
            self.n_step,
            self.batch_size,
            self.window_size,
            self.transaction_cost)

        self.dataTest_sequential = DataSequential(
            self.data_loader.data_test,
            'action_sequential',
            self.device,
            self.gamma,
            self.n_step,
            self.batch_size,
            self.window_size,
            self.transaction_cost)

        self.dataValidation_sequential = DataSequential(
            self.data_loader.data_validation,
            'action_sequential',
            self.device,
            self.gamma,
            self.n_step,
            self.batch_size,
            self.window_size,
            self.transaction_cost)

    def load_agents(self):
        self.dqn_pattern = DeepRL(
            self.data_loader,
            self.dataTrain_patternBased,
            self.dataTest_patternBased,
            self.dataValidation_patternBased,
            self.dataset_name,
            None,
            self.window_size,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step)

        self.dqn_vanilla = DeepRL(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent,
            self.dataTest_autoPatternExtractionAgent,
            self.dataValidation_autoPatternExtractionAgent,
            self.dataset_name,
            self.STATE_MODE_OHLC,
            self.window_size,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step)

        self.dqn_candle_rep = DeepRL(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent_candle_rep,
            self.dataTest_autoPatternExtractionAgent_candle_rep,
            self.dataValidation_autoPatternExtractionAgent_candle_rep,
            self.dataset_name,
            self.STATE_MODE_CANDLE_REP,
            self.window_size,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step)

        self.dqn_windowed = DeepRL(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent_windowed,
            self.dataTest_autoPatternExtractionAgent_windowed,
            self.dataValidation_autoPatternExtractionAgent_windowed,
            self.dataset_name,
            self.STATE_MODE_WINDOWED,
            self.window_size,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step)

        self.mlp_pattern = SimpleMLP(
            self.data_loader,
            self.dataTrain_patternBased,
            self.dataTest_patternBased,
            self.dataValidation_patternBased,
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

        self.mlp_vanilla = SimpleMLP(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent,
            self.dataTest_autoPatternExtractionAgent,
            self.dataValidation_autoPatternExtractionAgent,
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

        self.mlp_candle_rep = SimpleMLP(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent_candle_rep,
            self.dataTest_autoPatternExtractionAgent_candle_rep,
            self.dataValidation_autoPatternExtractionAgent_candle_rep,
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

        self.mlp_windowed = SimpleMLP(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent_windowed,
            self.dataTest_autoPatternExtractionAgent_windowed,
            self.dataValidation_autoPatternExtractionAgent_windowed,
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

        self.cnn1d = SimpleCNN(
            self.data_loader,
            self.dataTrain_autoPatternExtractionAgent,
            self.dataTest_autoPatternExtractionAgent,
            self.dataValidation_autoPatternExtractionAgent,
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

        self.cnn2d = CNN2d(
            self.data_loader,
            self.dataTrain_sequential,
            self.dataTest_sequential,
            self.dataValidation_sequential,
            self.dataset_name,
            self.feature_size,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step,
            window_size=self.window_size)

        self.gru = GRU(
            self.data_loader,
            self.dataTrain_sequential,
            self.dataTest_sequential,
            self.dataValidation_sequential,
            self.dataset_name,
            self.transaction_cost,
            self.feature_size,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step,
            window_size=self.window_size)

        self.deep_cnn = CNN(
            self.data_loader,
            self.dataTrain_sequential,
            self.dataTest_sequential,
            self.dataValidation_sequential,
            self.dataset_name,
            self.transaction_cost,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step,
            window_size=self.window_size)

        self.cnn_gru = CNN_GRU(
            self.data_loader,
            self.dataTrain_sequential,
            self.dataTest_sequential,
            self.dataValidation_sequential,
            self.dataset_name,
            self.transaction_cost,
            self.feature_size,
            BATCH_SIZE=self.batch_size,
            GAMMA=self.gamma,
            ReplayMemorySize=self.replay_memory_size,
            TARGET_UPDATE=self.target_update,
            n_step=self.n_step,
            window_size=self.window_size)

        self.cnn_attn = CNN_ATTN(
            self.data_loader,
            self.dataTrain_sequential,
            self.dataTest_sequential,
            self.dataValidation_sequential,
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

        self.test_portfolios['DQN-pattern'][key] = self.dqn_pattern.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-vanilla'][key] = self.dqn_vanilla.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-candlerep'][
            key] = self.dqn_candle_rep.test().get_daily_portfolio_value()
        self.test_portfolios['DQN-windowed'][key] = self.dqn_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-pattern'][key] = self.mlp_pattern.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-vanilla'][key] = self.mlp_vanilla.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-candlerep'][
            key] = self.mlp_candle_rep.test().get_daily_portfolio_value()
        self.test_portfolios['MLP-windowed'][key] = self.mlp_windowed.test().get_daily_portfolio_value()
        self.test_portfolios['CNN1d'][key] = self.cnn1d.test().get_daily_portfolio_value()
        self.test_portfolios['CNN2d'][key] = self.cnn2d.test().get_daily_portfolio_value()
        self.test_portfolios['GRU'][key] = self.gru.test().get_daily_portfolio_value()
        self.test_portfolios['Deep-CNN'][key] = self.deep_cnn.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-GRU'][key] = self.cnn_gru.test().get_daily_portfolio_value()
        self.test_portfolios['CNN-ATTN'][key] = self.cnn_attn.test().get_daily_portfolio_value()

        self.validation_portfolios['DQN-pattern'][key] = self.dqn_pattern.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['DQN-vanilla'][key] = self.dqn_vanilla.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['DQN-candlerep'][
            key] = self.dqn_candle_rep.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['DQN-windowed'][key] = self.dqn_windowed.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['MLP-pattern'][key] = self.mlp_pattern.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['MLP-vanilla'][key] = self.mlp_vanilla.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['MLP-candlerep'][
            key] = self.mlp_candle_rep.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['MLP-windowed'][key] = self.mlp_windowed.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['CNN1d'][key] = self.cnn1d.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['CNN2d'][key] = self.cnn2d.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['GRU'][key] = self.gru.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['Deep-CNN'][key] = self.deep_cnn.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['CNN-GRU'][key] = self.cnn_gru.test(test_type='validation').get_daily_portfolio_value()
        self.validation_portfolios['CNN-ATTN'][key] = self.cnn_attn.test(test_type='validation').get_daily_portfolio_value()

    def plot_and_save_sensitivity(self, data_set='test'):
        data = None
        if data_set == 'train':
            data = self.train_portfolios
        elif data_set == 'validation':
            data = self.validation_portfolios
        else:
            data = self.test_portfolios

        portfolio_plot_path = os.path.join(
            self.experiment_path, f'plots/portfolio/on_{data_set}')
        if not os.path.exists(portfolio_plot_path):
            os.makedirs(portfolio_plot_path)

        for model_in_question in data.keys():
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for gamma, color in zip(data[model_in_question], px.colors.qualitative.Plotly):

                profit_percentage = [
                    (data[model_in_question][gamma][i] - data[model_in_question][gamma][0]) /
                    data[model_in_question][gamma][0] * 100
                    for i in range(len(data[model_in_question][gamma]))]

                if data_set == 'test':
                    difference = len(data[model_in_question][gamma]) - \
                        len(self.data_loader.data_test_with_date)
                    prediction_df = pd.DataFrame({'date': self.data_loader.data_test_with_date.index,
                                                    'portfolio': profit_percentage[difference:]})
                elif data_set == 'train':
                    difference = len(data[model_in_question][gamma]) - \
                        len(self.data_loader.data_train_with_date)
                    prediction_df = pd.DataFrame({'date': self.data_loader.data_train_with_date.index,
                                                    'portfolio': profit_percentage[difference:]})

                elif data_set == 'validation':
                    difference = len(data[model_in_question][gamma]) - \
                        len(self.data_loader.data_validation_with_date)
                    prediction_df = pd.DataFrame({'date': self.data_loader.data_validation_with_date.index,
                                                    'portfolio': profit_percentage[difference:]})

                # Add a trace for each line
                fig.add_trace(go.Scatter(x=prediction_df['date'], y=prediction_df['portfolio'],
                                            mode='lines', name=gamma, line=dict(color=color)), secondary_y=False)

            # Update plot layout
            fig.update_layout(title=f'Tuning Hyperparameters of {model_in_question} using {self.evaluation_parameter} on {data_set} data',
                                xaxis_title='Time',
                                yaxis_title='% Rate of Return',
                                legend_title="Hyper-parameters",
                                font=dict(size=10))

            # Save plot as an file
            fig_file = os.path.join(portfolio_plot_path, f'{model_in_question}.html')
            fig.write_html(fig_file)
            fig_file = os.path.join(portfolio_plot_path, f'{model_in_question}.pdf')
            fig.write_image(fig_file)
            fig_file = os.path.join(portfolio_plot_path, f'{model_in_question}.svg')
            fig.write_image(fig_file)

    def save_portfolios(self):
        path = os.path.join(self.experiment_path, 'test_portfolios.pkl')
        save_pkl(path, self.test_portfolios)
        path = os.path.join(self.experiment_path, 'tain_portfolios.pkl')
        save_pkl(path, self.train_portfolios)
        path = os.path.join(self.experiment_path, 'validation_portfolios.pkl')
        save_pkl(path, self.validation_portfolios)

    def save_experiment(self):
        self.plot_and_save_sensitivity(data_set='validation')
        self.plot_and_save_sensitivity(data_set='test')
        self.plot_and_save_sensitivity(data_set='train')
        self.save_portfolios()


if __name__ == '__main__':
    gamma_list = [0.9, 0.8, 0.7]
    batch_size_list = [16, 64, 256]
    replay_memory_size_list = [16, 64, 256]
    n_step = 8
    window_size = 3
    dataset_name = "BTC-USD"
    n_episodes = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = 64
    target_update = 5

    gamma_default = 0.9
    batch_size_default = 16
    replay_memory_size_default = 32

    pbar = tqdm(len(gamma_list) + len(replay_memory_size_list) + len(batch_size_list))

    set_random_seed(42)
    # test gamma
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
        evaluation_parameter='gamma',
        transaction_cost=0)

    for gamma in gamma_list:
        run.gamma = gamma
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()

    # test batch-size
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
        evaluation_parameter='batch size',
        transaction_cost=0)

    for batch_size in batch_size_list:
        run.batch_size = batch_size
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()

    # test replay memory size
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
        evaluation_parameter='replay memory size',
        transaction_cost=0)

    for replay_memory_size in replay_memory_size_list:
        run.replay_memory_size = replay_memory_size
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()
    pbar.close()

path = os.path.join(os.getcwd(), f'run/BaseModel/')
if not os.path.exists(path):
    os.makedirs(path)
save_pkl(os.path.join(path, 'run.pkl'), run)