import numpy as np
import astroabc

from typing import TypeVar

Tensor = TypeVar('torch.tensor')
Array = TypeVar('numpy.ndarray')
function = TypeVar('function')


def dist(d, x):
    return np.sum(np.mean(np.square(d-x), axis=0))


class tABC:
    def __init__(self,
                 X: Array = [],
                 dist_metric: function = dist,
                 L: int = 1):
        self.X = X
        self.T = X.shape[0]
        self.prop = {'dfunc': dist_metric,
                     'outfile': "t_ABC_log.txt",
                     'verbose': 1,
                     'adapt_t': True,
                     'variance_method': 4,
                     'k_near': 10}
        self.N = X.shape[1]
        self.L = L
        self.num_params = 4*self.L+2*self.N+self.N*self.L
        self.priors = [('normal', [0, 1]) for i in range(self.num_params)]

    # Define a method for simulating the data given input parameters
    def simulation(self, param: Array):
        param = np.array(param)
        mu_z = param[:self.L]
        log_var_z = param[self.L:2*self.L]
        mu_s = param[2*self.L:2*self.L+self.N]
        beta = param[2*self.L+self.N:3*self.L+self.N]
        beta0 = param[3*self.L+self.N:4*self.L+self.N]
        gamma = param[4*self.L+self.N:4*self.L+2*self.N]
        phi = param[4*self.L+2*self.N:4*self.L+2*self.N+self.N*self.L]
        Z = np.zeros((self.T, self.L))
        Z[0, :] = np.random.normal(mu_z, 1)
        S = np.zeros((self.T, self.N))
        S[0, :] = np.random.normal(mu_s, 1)
        for t in range(1, self.T):
            Z[t, :] = beta*Z[t-1, :] + beta0 + np.random.randn(self.L)*np.exp(log_var_z/2)
        for t in range(1, self.T):
            shape = (self.N, self.L)
            S[t, :] = (gamma)*S[t-1, :] + np.matmul(phi.reshape(shape), Z[t, :])
        return np.array(S)

    # Define a method that initiates a sampler
    def sampler(self,
                num_particles: int = 100,
                tol_up: float = 0.5,
                tol_low: float = 0.002,
                itr: int = 20):
        self.sampler = astroabc.ABC_class(self.num_params, num_particles, self.X,
                                          [tol_up, tol_low], itr, self.priors, **self.prop)

    # Define a method that samples
    def sample(self):
        samples = self.sampler.sample(self.simulation)
        return samples
