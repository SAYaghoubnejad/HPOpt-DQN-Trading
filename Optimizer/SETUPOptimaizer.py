import torch
import numpy as np
from scipy.stats import gamma, beta
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement, ProbabilityOfImprovement
from .GPHedgeOptimizer import GP_HedgeBayesianOptimizer
from botorch.optim import optimize_acqf
from tqdm import tqdm

class SETUPBayesianOptimizer(GP_HedgeBayesianOptimizer):
    def __init__(self, objective_function, bounds, types, alpha=40.0, beta_param=0.1, a=17.0, b=3.0, X_init=None, Y_init=None, name='SETUP BO'):
        super().__init__(objective_function, bounds, types, X_init=X_init, Y_init=Y_init, name=name)
        self.alpha = alpha
        self.beta_param = beta_param
        self.a = a
        self.b = b
        self.acquisition_functions = {
            "PI": ProbabilityOfImprovement,
            "EI": qExpectedImprovement,
            "UCB": UpperConfidenceBound
        }
        self.acquisition_rewards = {key: 0 for key in self.acquisition_functions}
        self.m_history = []
        self.eta_history = []

    def select_next_point(self):
        # Calculate normalized rewards
        r_min = min(self.acquisition_rewards.values())
        r_max = max(self.acquisition_rewards.values())
        normalized_rewards = {key: (reward - r_max) / (r_max - r_min) if r_max != r_min else 0 for key, reward in self.acquisition_rewards.items()}

        # Calculate probabilities for each acquisition function
        total_reward = sum(np.exp(self.eta * np.array(list(normalized_rewards.values()))))
        probabilities = {key: np.exp(self.eta * reward) / total_reward for key, reward in normalized_rewards.items()}

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

    def sample_hyperparameters(self):
        self.eta = gamma.rvs(self.alpha, scale=1/self.beta_param)
        self.m = beta.rvs(self.a, self.b)
        self.m_history.append(self.m)
        self.eta_history.append(self.eta)

    def update_distributions(self, y_t, y_best, acq_func_name):
        if y_t > y_best:
            self.a += 1
        else:
            self.b += 1
        self.alpha += 1

        # Calculate normalized rewards
        r_min = min(self.acquisition_rewards.values())
        r_max = max(self.acquisition_rewards.values())
        normalized_rewards = {key: (reward - r_max) / (r_max - r_min) if r_max != r_min else 0 for key, reward in self.acquisition_rewards.items()}
        self.beta_param += normalized_rewards[acq_func_name]

    def update_acquisition_rewards(self, acq_func_name, x):
        mu, _ = self.model.posterior(x).mean.view(-1).item(), self.model.posterior(x).variance.view(-1).item()
        self.acquisition_rewards[acq_func_name] = self.m * self.acquisition_rewards[acq_func_name] - mu

    def optimize(self, n_steps=10, n_init_points=5):
        if self.X is None or self.Y is None:
            self.X, self.Y = self.initialize_data(n_init_points)
        self.max_obj_value = torch.max(self.Y).item()
        self.best = self.X[torch.where(self.Y == self.max_obj_value, 1.0, 0.0).to(torch.int)]
        self.history['X'].append(self.best)
        self.history['Y'].append(self.max_obj_value)
        pbar = tqdm(range(n_steps))
        for _ in pbar:
            self.sample_hyperparameters()
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
            self.update_distributions(Y_next.item(), self.max_obj_value, acq_func_name)

        pbar.close()
        return self.best