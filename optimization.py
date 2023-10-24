import random
import math
import copy
import numpy as np
import pyswarms
from scipy.optimize import dual_annealing

class SimulatedAnnealing:
    def __init__(self, T_start: float, T_stop: float, iterations: int, step_decrease_rate=0.0, step=0.01):
        if (T_start < T_stop):
            raise ValueError("T_start must be greater than T_stop")
        
        self.T_start = T_start
        self.T_stop = T_stop
        self.iterations = iterations
        self.alpha = (T_start - T_stop) / iterations
        self.step_decrease_rate = step_decrease_rate
        self.step = step
    
    #def minimize(self, func, args, x0, step, bounds):
    def minimize(self, func, bounds, **kwargs):
        """Performs minimization procedure using simulated annealing
        
        Arguments:
            func: function which is minimized in form of func(x, args)
            args: arguments of the function in form of tuple
            x0: starting point in form of [x01, x02, ..., x0n]
            steps: step sizes for each feature in form of array of floats [s1, s2, ..., sn]
            bounds: lower and upper bounds of each feature in form of array of tuples [(lb1, ub1), (lb2, ub2), ..., (lbn, ubn)]
        
        Returns:
            xo: optimal point where the function is minimum in form of [xo1, xo2, ..., xon]
            
        """
        args = tuple(kwargs.values())
        x0 = kwargs["x_original"]
        T = self.T_start
        x = copy.deepcopy(x0)
        x_best = copy.deepcopy(x0)
        E = func(x, *args)
        E_best = func(x_best, *args)
        
        while T > self.T_stop:
            x_next = self.make_step(x, self.step, bounds)
            E_next = func(x_next, *args)
            dE = E_next - E         
            
#             print("x:", x, "E:", E)
#             print("x_next:", x_next, "E_next:", E_next)
            
#             print("dE:", dE)
            
            if (dE < 0):
                x = x_next
                E = E_next
            else:
                # print("Replacement probability: %.3f" % (math.exp(-dE / T)))
                if (random.uniform(0, 1) < math.exp(-dE / T)):
                    x = x_next
                    E = E_next
            
            if (E < E_best):
                x_best = x
                E_best = E
            
            T = self.decrease_temperature(T)
            # step = step * (1 - self.step_decrease_rate)

        return x_best
            
    def make_step(self, x, step, bounds, discrete_step=True):
        """ Randomly finds next step """

        # dx = np.random.uniform(-step, step, size=x.shape[-1])
        if discrete_step:
            dx = np.random.choice([-step, 0, step], size=x.shape[-1])
        else:
            dx = np.random.uniform(-step, step, size=x.shape[-1])
        #print(dx)
        x_next = x + dx
        
#         i = 0
#         for xi, s, bs in zip(x, steps, bounds):
#             dxi = random.choice([-s, 0, s])
            
#             if (xi - dxi < bs[0]) or (xi + dxi > bs[1]):
#                 dxi = 0
            
#             x_next[i] += dxi
#             i+=1
            
        return x_next
        
        
    def decrease_temperature(self, T):
        """ Routine which decreases the temperature in the model """
        return T - self.alpha
        
        
class ParticleSwarmOptimization:
    def __init__(self, n_particles, iterations, patience, options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}, ftol=1e-3):
        self.n_particles = n_particles
        self.iterations = iterations
        self.patience = patience
        self.options = options
        self.ftol = ftol
        
    def minimize(self, func, bounds, **kwargs):
        min_bounds = tuple([b[0] for b in bounds])
        max_bounds = tuple([b[1] for b in bounds])
        bounds = (min_bounds, max_bounds)

        pso_opt = pyswarms.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=len(min_bounds), bounds=bounds, options=self.options, ftol_iter=self.patience, ftol=self.ftol)
        best_cost, best = pso_opt.optimize(func, iters=self.iterations, **kwargs)
        
        return best
    
class DualAnnealing:
    def __init__(self, initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxiter=1000, maxfun=1e7) -> None:
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept 
        self.maxiter = maxiter
        self.maxfun = maxfun

    def minimize(self, func, bounds, **kwargs):

        args = tuple(kwargs.values())
        x0 = kwargs["x_original"]
        result = dual_annealing(func, bounds, args=args, 
                                initial_temp=self.initial_temp, 
                                restart_temp_ratio=self.restart_temp_ratio, 
                                visit = self.visit,
                                accept = self.accept,
                                maxiter = self.maxiter,
                                maxfun = self.maxfun,
                                x0=x0)
        
        best = result.x
        return best