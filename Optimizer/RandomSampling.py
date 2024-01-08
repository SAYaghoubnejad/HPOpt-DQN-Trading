import torch
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm

class RandomSearch:
    def __init__(self, objective_function, bounds, types, X_init=None, Y_init=None, name='Simple RS'):
        self.function = objective_function
        self.bounds = bounds
        self.types = types
        if (X_init is None) ^ (Y_init is None):
            print('Warning: X_init and Y_init should be given.')
        self.X = X_init
        self.Y = Y_init
        self.x_dim = bounds.shape[1]
        self.best = None
        self.max_obj_value = -1 * torch.inf
        self.history = {'X': [], 'Y': []}
        self.optimizer_name = name

    def generate_random_tensor(self, n_sample=1):
        tensor = torch.rand((n_sample, self.x_dim), dtype=torch.float64)
        for dim in range(self.x_dim):
            lower, upper = self.bounds.T[dim]
            tensor[:, dim] = tensor[:, dim] * (upper - lower) + lower
            tensor[:, dim] = tensor[:, dim].to(self.types[dim])
        return tensor

    def initialize_data(self, n_init_points=5):
        X_init = self.generate_random_tensor(n_init_points)
        Y_init = torch.empty((n_init_points, 1), dtype=torch.float64)
        for index in range(n_init_points):
            Y_init[index][0], X_init[index] = self.function(X_init[index].flatten())
            if Y_init[index][0].item() > self.max_obj_value:
                self.max_obj_value = Y_init[index][0].item()
                self.best = X_init[index]
            self.history['X'].append(self.best)
            self.history['Y'].append(self.max_obj_value)
        return X_init, Y_init

    def select_next_point(self):
        return self.generate_random_tensor(1).flatten()

    def optimize(self, n_steps=10, n_init_points=5):
        if self.X is None or self.Y is None:
            self.X, self.Y = self.initialize_data(n_init_points)
        self.max_obj_value = torch.max(self.Y)
        self.best = self.X[torch.where(self.Y == self.max_obj_value, 1.0, 0.0).to(torch.int)]
        self.history['X'].append(self.best)
        self.history['Y'].append(self.max_obj_value.item())
        pbar = tqdm(range(n_steps))
        for _ in pbar:
            next_point = self.select_next_point()
            Y_next, X_next = self.function(next_point)
            self.X = torch.vstack((self.X, X_next))
            self.Y = torch.vstack((self.Y, Y_next))
            if Y_next.item() > self.max_obj_value:
                self.max_obj_value = Y_next.item()
                self.best = X_next
            self.history['X'].append(self.best)
            self.history['Y'].append(self.max_obj_value)
            pbar.update(1)
        pbar.close()
        return self.best

    def save_plots(self, path):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(self.Y.flatten().shape[-1]), y=self.Y.flatten(), mode='lines'))

        fig.update_layout(
            xaxis_title="Iteration No.",
            yaxis_title="Loss",
            title="Progress of Bayesian Optimization",
            font=dict(size=10),
        )

        # Save plot as an file
        fig_file = os.path.join(path, f'{self.optimizer_name} progress.html')
        fig.write_html(fig_file)
        fig_file = os.path.join(path, f'{self.optimizer_name} progress.pdf')
        fig.write_image(fig_file)
        fig_file = os.path.join(path, f'{self.optimizer_name} progress.svg')
        fig.write_image(fig_file)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(self.history['Y'])), y=self.history['Y'], mode='lines'))

        fig.update_layout(
            xaxis_title="Iteration No.",
            yaxis_title="Best Loss",
            title="Best Loss Found Till Each Iteration",
            font=dict(size=10)
        )

        # Save plot as an file
        fig_file = os.path.join(path, f'{self.optimizer_name} best loss.html')
        fig.write_html(fig_file)  
        fig_file = os.path.join(path, f'{self.optimizer_name} best loss.pdf')
        fig.write_image(fig_file)
        fig_file = os.path.join(path, f'{self.optimizer_name} best loss.svg')
        fig.write_image(fig_file)
