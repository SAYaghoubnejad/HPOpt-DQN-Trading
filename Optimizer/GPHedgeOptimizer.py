import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from .SimpleBayesianOptimizer import SimpleBayesianOptimizer
from tqdm import tqdm

class GP_HedgeBayesianOptimizer(SimpleBayesianOptimizer):
    def __init__(self, objective_function, bounds, types, eta=1.0, X_init=None, Y_init=None, name='GP-Hedge BO'):
        super().__init__(objective_function, bounds, types, X_init, Y_init, name)
        self.eta = eta
        self.acquisition_functions = {
            "PI": ProbabilityOfImprovement,
            "EI": qExpectedImprovement,
            "UCB": UpperConfidenceBound
        }
        self.acquisition_rewards = {key: 0 for key in self.acquisition_functions}
        self.acq_history = []

    def select_next_point(self):

        # Calculate probabilities for each acquisition function
        total_reward = sum(np.exp(self.eta * np.array(list(self.acquisition_rewards.values()))))
        probabilities = {key: np.exp(self.eta * reward) / total_reward for key, reward in self.acquisition_rewards.items()}

        # Select acquisition function based on probabilities
        acq_func_name = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        self.acq_history.append([probabilities, acq_func_name])
        if acq_func_name == "EI":
            acq_func = self.acquisition_functions[acq_func_name](self.model, best_f=self.max_obj_value)
        elif acq_func_name == "PI":
            acq_func = self.acquisition_functions[acq_func_name](self.model, best_f=self.max_obj_value)
        elif acq_func_name == "UCB":
            acq_func = self.acquisition_functions[acq_func_name](self.model, beta=10)

        next_point, _ = optimize_acqf(
            acq_func, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20
        )
        return next_point, acq_func_name

    def update_acquisition_rewards(self, acq_func_name, x):
        mu = self.model.posterior(x).mean.view(-1).item()
        self.acquisition_rewards[acq_func_name] -= mu

    def optimize(self, n_steps=10, n_init_points=5):
        if self.X is None or self.Y is None:
            self.X, self.Y = self.initialize_data(n_init_points)
        self.max_obj_value = torch.max(self.Y).item()
        self.best = self.X[torch.where(self.Y == self.max_obj_value, 1.0, 0.0).to(torch.int)]
        self.history['X'].append(self.best)
        self.history['Y'].append(self.max_obj_value)
        pbar = tqdm(range(n_steps))
        for _ in pbar:
            self.fit_model(self.X, self.Y)
            next_point, acq_func_name = self.select_next_point()
            Y_next, X_next = self.function(next_point.squeeze())
            self.X = torch.vstack((self.X, X_next))
            self.Y = torch.vstack((self.Y, Y_next))
            if Y_next.item() > self.max_obj_value:
                self.max_obj_value = Y_next.item()
                self.best = X_next
            self.history['X'].append(self.best)
            self.history['Y'].append(self.max_obj_value)
            self.update_acquisition_rewards(acq_func_name, next_point)
        pbar.close()
        return self.best
