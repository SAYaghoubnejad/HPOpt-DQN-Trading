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

import pandas as pd
import torch
import os
import random
import numpy as np
from utils import save_pkl, load_pkl, set_random_seed, plot_hyperparams, plot_optimization_progress, plot_prob_accusition
import os
import pandas as pd
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
        split_point='2022-01-01',
        validation_split_point='2023-01-01',
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
    def __init__(
        self,
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
        self.test_data_first_price = self.data_loader.data_test_with_date.close[0]
        self.val_data_first_price = self.data_loader.data_validation_with_date.close[0]
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
        # window with k candles inside + the trend of those candles
        self.STATE_MODE_WINDOWED = 5

        self.dataTrain_autoPatternExtractionAgent = None
        self.dataTest_autoPatternExtractionAgent = None
        self.dataValidation_autoPatternExtractionAgent = None
        self.dataTrain_patternBased = None
        self.dataTest_patternBased = None
        self.dataValidation_patternBased = None
        self.dataTrain_autoPatternExtractionAgent_candle_rep = None
        self.dataTest_autoPatternExtractionAgent_candle_rep = None
        self.dataValidation_autoPatternExtractionAgent_candle_rep = None
        self.dataTrain_autoPatternExtractionAgent_windowed = None
        self.dataTest_autoPatternExtractionAgent_windowed = None
        self.dataValidation_autoPatternExtractionAgent_windowed = None
        self.dataTrain_sequential = None
        self.dataTest_sequential = None
        self.dataValidation_sequential = None
        self.model_in_question = None
        self.experiment_path = os.path.join(
            os.getcwd(), 'Results/' + self.evaluation_parameter + '/')
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        self.models = {
            'DQN-pattern': None,
            'DQN-vanilla': None,
            'DQN-candlerep': None,
            'DQN-windowed': None,
            'MLP-pattern': None,
            'MLP-vanilla': None,
            'MLP-candlerep': None,
            'MLP-windowed': None,
            'CNN1d': None,
            'CNN2d': None,
            'GRU': None,
            'Deep-CNN': None,
            'CNN-GRU': None,
            'CNN-ATTN': None}

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
        
        self.reset()

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
        self.models['DQN-pattern'] = DeepRL(
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

        self.models['DQN-vanilla'] = DeepRL(
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

        self.models['DQN-candlerep'] = DeepRL(
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

        self.models['DQN-windowed'] = DeepRL(
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

        self.models['MLP-pattern'] = SimpleMLP(
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

        self.models['MLP-vanilla'] = SimpleMLP(
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

        self.models['MLP-candlerep'] = SimpleMLP(
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

        self.models['MLP-windowed'] = SimpleMLP(
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

        self.models['CNN1d'] = SimpleCNN(
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

        self.models['CNN2d'] = CNN2d(
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

        self.models['GRU'] = GRU(
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

        self.models['Deep-CNN'] = CNN(
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

        self.models['CNN-GRU'] = CNN_GRU(
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

        self.models['CNN-ATTN'] = CNN_ATTN(
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
        self.models[self.model_in_question].train(self.n_episodes)

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

        # Train
        self.train_portfolios[self.model_in_question][key] = self.models[self.model_in_question].test(
            test_type='train').get_daily_portfolio_value()

        # Test
        self.test_portfolios[self.model_in_question][key] = self.models[self.model_in_question].test(
            initial_investment=self.test_data_first_price).get_daily_portfolio_value()

        # Validation
        self.validation_portfolios[self.model_in_question][key] = self.models[self.model_in_question].test(
            test_type='validation', initial_investment=self.val_data_first_price).get_daily_portfolio_value()

    def average_return(self):
        key = None
        if self.evaluation_parameter == 'gamma':
            key = self.gamma
        elif self.evaluation_parameter == 'batch size':
            key = self.batch_size
        elif self.evaluation_parameter == 'replay memory size':
            key = self.replay_memory_size
        else:
            key = f'G: {self.gamma}, BS: {self.batch_size}, RMS: {self.replay_memory_size}, n: {self.n_step}, episodes: {self.n_episodes}'

        self.avg_returns = (self.test_portfolios[self.model_in_question][key][-1] - self.test_portfolios[self.model_in_question][key][0]) \
            / self.test_portfolios[self.model_in_question][key][0] * 100
        return self.avg_returns

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

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for gamma, color in zip(data[self.model_in_question], px.colors.qualitative.Plotly):

            profit_percentage = [
                (data[self.model_in_question][gamma][i] - data[self.model_in_question][gamma][0]) /
                data[self.model_in_question][gamma][0] * 100
                for i in range(len(data[self.model_in_question][gamma]))]

            if data_set == 'test':
                difference = len(data[self.model_in_question][gamma]) - \
                    len(self.data_loader.data_test_with_date)
                prediction_df = pd.DataFrame({'date': self.data_loader.data_test_with_date.index,
                                                'portfolio': profit_percentage[difference:]})
            elif data_set == 'train':
                difference = len(data[self.model_in_question][gamma]) - \
                    len(self.data_loader.data_train_with_date)
                prediction_df = pd.DataFrame({'date': self.data_loader.data_train_with_date.index,
                                                'portfolio': profit_percentage[difference:]})

            elif data_set == 'validation':
                difference = len(data[self.model_in_question][gamma]) - \
                    len(self.data_loader.data_validation_with_date)
                prediction_df = pd.DataFrame({'date': self.data_loader.data_validation_with_date.index,
                                                'portfolio': profit_percentage[difference:]})

            # Add a trace for each line
            fig.add_trace(go.Scatter(x=prediction_df['date'], y=prediction_df['portfolio'],
                                        mode='lines', name=gamma, line=dict(color=color)), secondary_y=False)

        # Update plot layout
        fig.update_layout(title=f'Tuning Hyperparameters of {self.model_in_question} using {self.evaluation_parameter} on {data_set} data',
                            xaxis_title='Time',
                            yaxis_title='% Rate of Return',
                            legend_title="Hyper-parameters",
                            font=dict(size=10))

        # Save plot as an file
        fig_file = os.path.join(portfolio_plot_path, f'{self.model_in_question}.html')
        fig.write_html(fig_file)
        fig_file = os.path.join(portfolio_plot_path, f'{self.model_in_question}.pdf')
        fig.write_image(fig_file)
        fig_file = os.path.join(portfolio_plot_path, f'{self.model_in_question}.svg')
        fig.write_image(fig_file)

    def plot_and_save_return(self):
        prediction_plot_path = os.path.join(
            self.experiment_path, 'plots/prediction')
        if not os.path.exists(prediction_plot_path):
            os.makedirs(prediction_plot_path)

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        # Train data
        train_df = pd.DataFrame(
            self.data_loader.data_train_with_date.close, index=self.data_loader.data.index)
        fig.add_trace(go.Scatter(x=train_df.index,
                        y=train_df['close'], mode='lines', name='Train', line=dict(color=colors[0])))

        # Test data
        test_df = pd.Series(
            self.data_loader.data_test_with_date.close, index=self.data_loader.data.index)
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df,
                        mode='lines', name='Test', line=dict(color=colors[1])))

        # Validation data
        validation_df = pd.Series(
            self.data_loader.data_validation_with_date.close, index=self.data_loader.data.index)
        fig.add_trace(go.Scatter(x=validation_df.index, y=validation_df,
                        mode='lines', name='Validation', line=dict(color=colors[2])))

        # Predictions
        for gamma, color in zip(self.test_portfolios[self.model_in_question], colors[3:]):
            difference = len(
                self.test_portfolios[self.model_in_question][gamma]) - len(self.data_loader.data_test_with_date)
            prediction_series = pd.Series(self.test_portfolios[self.model_in_question][gamma][difference:],
                                            index=self.data_loader.data_test_with_date.index)
            prediction_series = prediction_series.reindex(
                self.data_loader.data.index, fill_value=np.nan)
            fig.add_trace(go.Scatter(x=prediction_series.index, y=prediction_series,
                            mode='lines', name=gamma, line=dict(color=color)))

        # Update plot layout
        fig.update_layout(title=f'Train, Test and Prediction of model {self.model_in_question} on dataset {self.dataset_name}',
                            xaxis_title='Time',
                            yaxis_title='Close Price',
                            legend_title="Legend",
                            font=dict(size=10))

        # Save plot
        fig_file = os.path.join(prediction_plot_path, f'{self.model_in_question}.html')
        fig.write_html(fig_file)
        fig_file = os.path.join(prediction_plot_path, f'{self.model_in_question}.pdf')
        fig.write_image(fig_file)
        fig_file = os.path.join(prediction_plot_path, f'{self.model_in_question}.svg')
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
        self.plot_and_save_return()
        self.save_portfolios()


iter = 2
init_set = 2

models = [
    'DQN-pattern',
    # 'DQN-vanilla',
    # 'DQN-candlerep',
    # 'DQN-windowed',
    # 'MLP-pattern',
    # 'MLP-vanilla',
    # 'MLP-candlerep',
    # 'MLP-windowed',
    # 'CNN1d',
    # 'CNN2d',
    # 'GRU',
    # 'Deep-CNN',
    # 'CNN-GRU',
    # 'CNN-ATTN'
]

params = ['Gamma', 'Batch Size', 'Reply Memory Size', 'Number of Steps', 'Number of Episodes']
# gamma, log2(batch_size), log2(replay_memory_size), log2(n_step), n_episodes / 10
bounds = torch.tensor([[0.4, 8.0, 8.0, 2.0, 10.0], [1.0, 512.0, 512.0, 64.0, 60.0]])
types = [torch.float64, torch.int64, torch.int64, torch.int64, torch.int64]

n_step = 8
window_size = 3
dataset_name = "S&P"
n_episodes = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on:', device)
feature_size = 64
target_update = 5

gamma_default = 0.9
batch_size_default = 16
replay_memory_size_default = 32



# # Imports Optimizer
# from Optimizer.SimpleBayesianOptimizer import SimpleBayesianOptimizer
# set_random_seed(42)

# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='Simple BO (UCB)',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     optimizer = SimpleBayesianOptimizer(objective_func, bounds, types, name='Simple BO (UCB)')
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set)

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)






# # Imports Optimizer
# from Optimizer.SimpleBayesianOptimizer import SimpleBayesianOptimizer
# set_random_seed(42)

# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='Simple BO (EI)',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     # optimizer = load_pkl('run/Simple BO/DQN-pattern/optimizer.pkl')

#     optimizer = SimpleBayesianOptimizer(objective_func, bounds, types, name='Simple BO (EI)')
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set, acquisition_func='EI')

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)





# # Imports Optimizer
# from Optimizer.SimpleBayesianOptimizer import SimpleBayesianOptimizer
# set_random_seed(42)
# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='Simple BO (PI)',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     # optimizer = load_pkl('run/Simple BO/DQN-pattern/optimizer.pkl')

#     optimizer = SimpleBayesianOptimizer(objective_func, bounds, types, name='Simple BO (PI)')
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set, acquisition_func='PI')

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)





# # Imports Optimizer
# from Optimizer.RandomSampling import RandomSearch
# set_random_seed(42)
# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='Random Search',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     optimizer = RandomSearch(objective_func, bounds, types, name='Random Search')
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set)

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)




# # Imports Optimizer
# from Optimizer.GPHedgeOptimizer import GP_HedgeBayesianOptimizer
# set_random_seed(42)
# eta=0.01
# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='GP Hedge',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     optimizer = GP_HedgeBayesianOptimizer(objective_func, bounds, types, name='GP Hedge', eta=eta)
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set)

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)



# # Imports Optimizer
# from Optimizer.NoPastBayesianOptimizer import NoPastBayesianOptimizer
# set_random_seed(42)
# eta = 0.1
# m = 0.8

# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='No-Past-BO',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     optimizer = NoPastBayesianOptimizer(objective_func, bounds, types, name='No-Past-BO', eta=eta, m=m)
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set)

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}_m:{m}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}_m:{m}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)




# # Imports Optimizer
# from Optimizer.NoPastBayesianOptimizerVarM import NoPastBoVarM
# set_random_seed(42)
# eta = 1.0
# m_start = 0.1
# m_end = 1.0
# run = SensitivityRun(
#     dataset_name,
#     gamma_default,
#     batch_size_default,
#     replay_memory_size_default,
#     feature_size,
#     target_update,
#     n_episodes,
#     n_step,
#     window_size,
#     device,
#     evaluation_parameter='No-Past-BO-VarM',
#     transaction_cost=0)

# for model_name  in models:

#     run.model_in_question = model_name

#     def objective_func(params):
#         print(params)
#         run.gamma = round(params[0].item(), 2)
#         run.batch_size = int(round(params[1].item()))
#         run.replay_memory_size = int(round(params[2].item()))
#         run.n_step = int(round(params[3].item()))
#         run.n_episodes = int(round(params[4].item()))
#         set_random_seed(42)
#         run.reset()
#         run.train()
#         set_random_seed()
#         run.evaluate_sensitivity()
#         eval = torch.tensor([run.average_return()])
#         X = torch.tensor([
#             round(params[0].item(), 2),
#             int(round(params[1].item())),
#             int(round(params[2].item())),
#             int(round(params[3].item())),
#             int(round(params[4].item()))
#             ])
#         return eval, X

#     optimizer = NoPastBoVarM(objective_func, bounds, types, name='No-Past-BO-VarM', eta=eta, m_start=m_start, m_end=m_end)
#     run.evaluation_parameter = optimizer.optimizer_name
#     optimizer.optimize(n_steps=iter, n_init_points=init_set)

#     run.save_experiment()

#     path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}_m:{m_start}-{m_end}/{run.model_in_question}/')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

#     optimizer.save_plots(path)

# path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}_eta:{eta}_m:{m_start}-{m_end}/')
# if not os.path.exists(path):
#     os.makedirs(path)
# save_pkl(os.path.join(path, 'run.pkl'), run)





# Imports Optimizer
from Optimizer.SETUPOptimaizer import SETUPBayesianOptimizer
set_random_seed(42)
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
    evaluation_parameter='SETUP-BO',
    transaction_cost=0)

for model_name  in models:

    run.model_in_question = model_name

    def objective_func(params):
        print(params)
        run.gamma = round(params[0].item(), 2)
        run.batch_size = int(round(params[1].item()))
        run.replay_memory_size = int(round(params[2].item()))
        run.n_step = int(round(params[3].item()))
        run.n_episodes = int(round(params[4].item()))
        set_random_seed(42)
        run.reset()
        run.train()
        set_random_seed()
        run.evaluate_sensitivity()
        eval = torch.tensor([run.average_return()])
        X = torch.tensor([
            round(params[0].item(), 2),
            int(round(params[1].item())),
            int(round(params[2].item())),
            int(round(params[3].item())),
            int(round(params[4].item()))
            ])
        return eval, X

    optimizer = SETUPBayesianOptimizer(objective_func, bounds, types, name='SETUP BO')
    run.evaluation_parameter = optimizer.optimizer_name
    optimizer.optimize(n_steps=iter, n_init_points=init_set)

    run.save_experiment()

    path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/{run.model_in_question}/')
    if not os.path.exists(path):
        os.makedirs(path)
    save_pkl(os.path.join(path, 'optimizer.pkl'), optimizer)

    optimizer.save_plots(path)

path = os.path.join(os.getcwd(), f'run/{optimizer.optimizer_name}/')
if not os.path.exists(path):
    os.makedirs(path)
save_pkl(os.path.join(path, 'run.pkl'), run)