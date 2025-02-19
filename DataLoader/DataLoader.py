import warnings
import pandas as pd
import pickle
from PatternDetectionInCandleStick.LabelPatterns import label_candles
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import os
import ast
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class YahooFinanceDataLoader:
    """ Dataset form GOOGLE"""

    def __init__(self, dataset_name, split_point, validation_split_point, begin_date=None, end_date=None, load_from_file=False):
        """
        :param dataset_name
            folder name in './Data' directory
        :param file_name
            csv file name in the Data directory
        :param load_from_file
            if False, it would load and process the data from the beginning
            and save it again in a file named 'data_processed.csv'
            else, it has already processed the data and saved in 'data_processed.csv', so it can load
            from file. If you have changed the original .csv file in the Data directory, you should set it to False
            so that it will rerun the preprocessing process on the new data.
        :param begin_date
            This is the beginning date in the .csv file that you want to consider for the whole train and test
            processes
        :param end_date
            This is the end date in the .csv file of the original data to to consider for the whole train and test
            processes
        :param split_point
            The point (date) between begin_date and end_date that you want to split the train and test sets.
        :param validation_split_point
            The point (date) between split_point and end_date that you want to split the test and validation sets.
        """
        warnings.filterwarnings('ignore')
        self.DATA_NAME = dataset_name
        self.DATA_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                      f'Data/{dataset_name}') + '/'
        self.OBJECT_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, 'Objects') + '/'

        self.DATA_FILE = dataset_name + '.csv'

        self.split_point = split_point
        self.begin_date = begin_date
        self.end_date = end_date

        if not load_from_file:
            self.data, self.patterns = self.load_data()
            self.save_pattern()
            self.normalize_data()
            self.data.to_csv(f'{self.DATA_PATH}data_processed.csv', index=True)

            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]

            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]

            if type(split_point) == str:
                self.data_train = self.data[self.data.index < split_point]
                if validation_split_point is None:
                    self.data_test = self.data[self.data.index >= split_point]
                else:
                    self.data_test = self.data[(self.data.index >= split_point) & (self.data.index < validation_split_point)]
                    self.data_validation = self.data[self.data.index >= validation_split_point]
            elif type(split_point) == int:
                self.data_train = self.data[:split_point]
                if validation_split_point is None:
                    self.data_test = self.data[split_point:]
                else:
                    self.data_test = self.data[split_point:validation_split_point]
                    self.data_validation = self.data[validation_split_point:]
            else:
                raise ValueError('Split point should be either int or date!')

            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()
            self.data_validation_with_date = self.data_validation.copy()

            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)
            self.data_validation.reset_index(drop=True, inplace=True)
            # self.data.reset_index(drop=True, inplace=True)
        else:
            self.data = pd.read_csv(f'{self.DATA_PATH}data_processed.csv')
            self.data.set_index('Date', inplace=True)
            labels = list(self.data.label)
            labels = [ast.literal_eval(l) for l in labels]
            self.data['label'] = labels
            self.load_pattern()
            self.normalize_data()

            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]

            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]

            if type(split_point) == str:
                self.data_train = self.data[self.data.index < split_point]
                if validation_split_point is None:
                    self.data_test = self.data[self.data.index >= split_point]
                else:
                    self.data_test = self.data[(self.data.index >= split_point) & (self.data.index < validation_split_point)]
                    self.data_validation = self.data[self.data.index >= validation_split_point]
            elif type(split_point) == int:
                self.data_train = self.data[:split_point]
                if validation_split_point is None:
                    self.data_test = self.data[split_point:]
                else:
                    self.data_test = self.data[split_point:validation_split_point]
                    self.data_validation = self.data[validation_split_point:]
            else:
                raise ValueError('Split point should be either int or date!')

            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()
            self.data_validation_with_date = self.data_validation.copy()

            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)
            self.data_validation.reset_index(drop=True, inplace=True)
            # self.data.reset_index(drop=True, inplace=True)

    def load_data(self):
        """
        This function is used to read and clean data from .csv file.
        @return:
        """
        data = pd.read_csv(f'{self.DATA_PATH}{self.DATA_FILE}')
        data.dropna(inplace=True)
        data.set_index('Date', inplace=True)
        data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        data = data.drop(['Adj Close', 'Volume'], axis=1)
        data['mean_candle'] = data.close
        patterns = label_candles(data)
        return data, list(patterns.keys())

    def plot_data(self):
        """
        This function is used to plot the dataset (train and test in different colors).
        @return:
        """
        # Assuming self.data and other properties are already defined
        df1 = pd.Series(self.data_train_with_date.close, index=self.data.index)
        df2 = pd.Series(self.data_test_with_date.close, index=self.data.index)
        df3 = pd.Series(self.data_validation_with_date.close, index=self.data.index)

        # Create a subplot
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Add traces
        fig.add_trace(go.Scatter(x=df1.index, y=df1, mode='lines', name='Train', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df2.index, y=df2, mode='lines', name='Test', line=dict(color='red')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df3.index, y=df3, mode='lines', name='Validation', line=dict(color='green')), secondary_y=False)

        # Set plot titles and labels
        fig.update_layout(title_text=f'Train, Test and Validation Sections of Dataset {self.DATA_NAME}',
                        xaxis_title='Time',
                        yaxis_title='Close Price',
                        legend_title="Sections")

        # Define the file path prefix for saving the images
        file_path_prefix = f'{Path(self.DATA_PATH).parent}/DatasetImages/{self.DATA_NAME}'

        # Calculate the width and height in pixels (300 DPI for high-quality print)
        dpi = 300
        width_inches = 3.5  # Single column width in inches
        height_inches = 2.625  # Adjusted height in inches for a 4:3 aspect ratio

        # Convert inches to pixels
        width_pixels = int(width_inches * dpi)
        height_pixels = int(height_inches * dpi)

        # Save the plot to files in different formats with dimensions suitable for single column figures
        fig.write_image(f'{file_path_prefix}.jpg', format='jpg', width=width_pixels, height=height_pixels)
        fig.write_image(f'{file_path_prefix}.pdf', format='pdf', width=width_pixels, height=height_pixels)
        fig.write_image(f'{file_path_prefix}.svg', format='svg', width=width_pixels, height=height_pixels)

    def save_pattern(self):
        with open(
                f'{self.OBJECT_PATH}pattern.pkl', 'wb') as output:
            pickle.dump(self.patterns, output, pickle.HIGHEST_PROTOCOL)

    def load_pattern(self):
        with open(self.OBJECT_PATH + 'pattern.pkl', 'rb') as input:
            self.patterns = pickle.load(input)

    def normalize_data(self):
        """
        This function normalizes the input data
        @return:
        """
        min_max_scaler = MinMaxScaler()
        self.data['open_norm'] = min_max_scaler.fit_transform(self.data.open.values.reshape(-1, 1))
        self.data['high_norm'] = min_max_scaler.fit_transform(self.data.high.values.reshape(-1, 1))
        self.data['low_norm'] = min_max_scaler.fit_transform(self.data.low.values.reshape(-1, 1))
        self.data['close_norm'] = min_max_scaler.fit_transform(self.data.close.values.reshape(-1, 1))

    def calculate_max_return(self, data_set='Test'):
        if data_set == 'Train':
            data = self.data_train_with_date
        elif data_set == 'Validation':
            data = self.data_validation_with_date
        else:
            data = self.data_test_with_date

        daily_returns = []
        for index, data in data.iterrows():
            daily_returns.append((data['close'] - data['open']) / data['open'])

        res = 100
        for ret in daily_returns:
            # if ret > 0:
            res *= (1 + ret)

        return daily_returns