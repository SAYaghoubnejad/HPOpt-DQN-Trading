import numpy as np
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from .GPHedgeOptimizer import GP_HedgeBayesianOptimizer

class NoPastBayesianOptimizer(GP_HedgeBayesianOptimizer):
    def __init__(self, objective_function, bounds, types, eta=1.0, m=0.5, X_init=None, Y_init=None, name='No-PASt BO'):
        super().__init__(objective_function, bounds, types, eta, X_init, Y_init, name)
        self.eta = eta
        self.m = m  # m parameter for updating rewards
        self.acquisition_functions = {
            "PI": ProbabilityOfImprovement,
            "EI": qExpectedImprovement,
            "UCB": UpperConfidenceBound
        }
        self.acquisition_rewards = {key: 0 for key in self.acquisition_functions}
        self.acq_history = []

    def select_next_point(self):
        best_x = None
        best_acq = None

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

    def update_acquisition_rewards(self, acq_func_name, x):
        mu, _ = self.model.posterior(x).mean.view(-1).item(), self.model.posterior(x).variance.view(-1).item()
        self.acquisition_rewards[acq_func_name] = self.m * self.acquisition_rewards[acq_func_name] - mu