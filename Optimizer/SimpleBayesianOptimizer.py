from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement

import torch
from tqdm import tqdm

class SimpleBayesianOptimizer:
    def __init__(self, objective_function, bounds, types, X_init=None, Y_init=None):
        self.function: function = objective_function
        self.bounds = bounds
        self.types = types
        if (X_init == None) ^ (Y_init == None):
            print('Warning: X_init and Y_init should be given.')
        self.X = X_init
        self.Y = Y_init
        self.x_dim = bounds.shape[1]
        self.best = None
        self.min_lose = torch.inf
        self.history = {'X': [], 'Y': []}

    def generate_random_tensor(self, n_sample=1):
        tensor = torch.rand((n_sample, self.x_dim), dtype=torch.float64)

        for dim in range(self.x_dim):
            lower, upper = self.bounds.T[dim]
            tensor[:, dim] = tensor[:, dim] * (upper - lower) + lower
            tensor[:, dim] = tensor[:, dim].to(self.types[dim])
        return tensor

    def initialize_data(self, n_init_points=5):
        # Generate initial data
        X_init = self.generate_random_tensor(n_init_points)
        Y_init = torch.empty((n_init_points, 1), dtype=torch.float64)
        for index in range(n_init_points):
            Y_init[index][0] = self.function(X_init[index].flatten())
        return X_init, Y_init

    def fit_model(self, X, Y):
        self.model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def select_next_point(self, acquisition_func):
        # Placeholder for the acquisition function selection logic
        # This is where GP-Hedge, No-Past-BO, and SETUP-BO would be implemented
        if acquisition_func == "UCB":
            acq_func = UpperConfidenceBound(self.model, beta=0.1)
        elif acquisition_func == "EI":
            acq_func = qExpectedImprovement(self.model, best_f=self.function.optimal_value)
        # Add other acquisition functions as needed
        next_point, _ = optimize_acqf(
            acq_func, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20
        )
        return next_point

    def simple_BO(self, n_steps=10, acquisition_func="UCB", n_init_points=5):
        if self.X == None or self.Y == None:
            self.X, self.Y = self.initialize_data(n_init_points)
        self.min_lose = torch.max(self.Y)
        self.best = self.X[torch.where(self.Y == self.min_lose, 1.0, 0.0).to(torch.int)]
        self.history['X'].append(self.best)
        self.history['Y'].append(self.min_lose.item())
        pbar = tqdm(n_steps)
        for _ in range(n_steps):
            self.fit_model(self.X, self.Y)
            next_point = self.select_next_point(acquisition_func).flatten()
            print('next', next_point)
            Y_next = torch.tensor([self.function(next_point)])
            self.X = torch.vstack((self.X, next_point))
            self.Y = torch.vstack((self.Y, Y_next))
            if Y_next.item() > self.min_lose:
                self.min_lose = Y_next.item()
                self.best = next_point
            self.history['X'].append(self.best)
            self.history['Y'].append(self.min_lose)
            pbar.update(1)
        pbar.close()
        return self.best