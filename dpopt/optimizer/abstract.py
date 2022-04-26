from abc import ABC, abstractmethod
import numpy as np
#TODO better documentary and type indication

# https://docs.scipy.org/doc/scipy/reference/optimize.html
from scipy.optimize import Bounds, minimize


class Optimizer(ABC):
    """
    Base class for optimizers for best threshold attack.
    """
    def __init__(self, n_variables, t0=None, q0=None):
        assert n_variables in [1, 2], 'We only support optimization on at most 2 variabls.'
        self.n_variables = n_variables
        self.objective = None
        self.bounds = None
        if self.n_variables == 1:
            self.x0 = np.asarray([t0])
        else:
            self.x0 = np.asarray([t0, q0])

    def _set_objective(self, func, *fargs):
        if self.n_variables == 1:
            self.objective = lambda x: -func(x[0], 0, *fargs)
        else:
            self.objective = lambda x: -func(x[0], x[1], *fargs)

    @abstractmethod
    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0): pass

    @abstractmethod
    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        '''
        Returns:
            pair (t, q, lb) of threshold t, tie-breaker q and corresponding lower bound on power lb.
        '''
        pass

    def __str__(self):
        return type(self).__name__.split(".")[-1] + ("_uni" if self.n_variables == 1 else "_bi")


class LocalOptimizer(Optimizer):
    def __init__(self, method, n_variables, t0=None, q0=None):
        self.method = method
        super(LocalOptimizer, self).__init__(n_variables, t0, q0)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = Bounds([tmin], [tmax])
        else:
            self.bounds = Bounds([tmin, qmin], [tmax, qmax])

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = minimize(self.objective, self.x0, method=self.method, bounds=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb