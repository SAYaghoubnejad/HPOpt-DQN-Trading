import torch
from .NoPastBayesianOptimizer import NoPastBayesianOptimizer
from tqdm import tqdm

class NoPastBoVarM(NoPastBayesianOptimizer):
    def __init__(self, objective_function, bounds, types, eta=1.0, m_start=0.1, m_end=0.9, X_init=None, Y_init=None, name='No-PASt BO Variable M'):
        super().__init__(objective_function, bounds, types, eta, m_start, X_init, Y_init, name)
        self.eta = eta
        self.m = m_start
        self.m_start = m_start
        self.m_end = m_end
        self.acquisition_rewards = {key: 0 for key in self.acquisition_functions}
        self.acq_history = []
        self.m_history = []

    def update_m(self, current_step, total_steps):
        stimilate_rate = (self.m_end - self.m_start) / total_steps
        self.m = max(self.m_start + stimilate_rate * current_step, self.m_end)
        self.m_history.append(self.m)

    def optimize(self, n_steps=10, n_init_points=5):
        if self.X is None or self.Y is None:
            self.X, self.Y = self.initialize_data(n_init_points)
        self.max_obj_value = torch.max(self.Y).item()
        self.best = self.X[torch.where(self.Y == self.max_obj_value, 1.0, 0.0).to(torch.int)]
        self.history['X'].append(self.best)
        self.history['Y'].append(self.max_obj_value)
        pbar = tqdm(range(n_steps))
        for step in pbar:
            self.update_m(step, n_steps)
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