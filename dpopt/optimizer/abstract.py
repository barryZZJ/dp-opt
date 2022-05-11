from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """
    An optimizer to search for best combination of t,q that produces the highest lower bound on power.
    """

    def __init__(self, n_variables, t0=None, q0=None):
        """
        Args:
             t0: initial guess on t
             q0: initial guess on q
        """
        assert n_variables in [1, 2], 'We only support optimization on at most 2 variables.'
        self.n_variables = n_variables
        self.objective = None
        self.bounds = None
        if self.n_variables == 1:
            self.x0 = np.asarray([t0])
        else:
            self.x0 = np.asarray([t0, q0])

    def _set_objective(self, func, *fargs):
        """
        set the optimization objective to be pass to each optimizer.
        note the objective should be a minimization problem.
        """
        if self.n_variables == 1:
            self.objective = lambda x: -func(x[0], 0, *fargs)
        else:
            self.objective = lambda x: -func(x[0], x[1], *fargs)

    @abstractmethod
    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        """
        set the constraints on t and q
        """
        pass

    @abstractmethod
    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        '''
        Maximize func with additional fargs, with dynamic constraints on t. Additional args and kwargs are passed to optimizer

        Args:
            func: function to be maximized
            tmin: lower constraint on t
            tmax: upper constraint on t
            args: additional args for optimizer
            kwargs: additional args for optimizer
        Returns:
            pair (t, q, lcb) of threshold t, tie-breaker q and corresponding lower bound on power lcb.
        '''
        pass

    def __str__(self):
        return type(self).__name__.split(".")[-1] + ("_uni" if self.n_variables == 1 else "_bi")
