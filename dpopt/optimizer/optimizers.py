"""
An implementation of all optimizers listed in the paper
according to https://docs.scipy.org/doc/scipy/reference/optimize.html
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize, Bounds, differential_evolution, dual_annealing, brute, \
    basinhopping, shgo

from dpopt.optimizer.abstract import Optimizer


class Bounded(Optimizer):
    def __init__(self):
        super(Bounded, self).__init__(n_variables=1)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        self.bounds = (tmin, tmax)

    def _set_objective(self, func, *fargs):
        self.objective = lambda x: -func(x, 0, *fargs)

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = minimize_scalar(self.objective, bounds=self.bounds, method='bounded')
        t, q = np.round(res.x, 3), 0.0
        lcb = -res.fun
        return t, q, lcb


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


class NelderMead(LocalOptimizer):
    def __init__(self, n_variables, t0, q0=None):
        super(NelderMead, self).__init__('Nelder-Mead', n_variables, t0, q0)


class Powell(LocalOptimizer):
    def __init__(self, n_variables, t0, q0=None):
        super(Powell, self).__init__('Powell', n_variables, t0, q0)


class L_BFGS_B(LocalOptimizer):
    def __init__(self, n_variables, t0, q0):
        super(L_BFGS_B, self).__init__('L-BFGS-B', n_variables, t0, q0)


class TNC(LocalOptimizer):
    def __init__(self, n_variables, t0, q0):
        super(TNC, self).__init__('TNC', n_variables, t0, q0)


class COBYLA(Optimizer):
    def __init__(self, n_variables, t0, q0=None):
        super(COBYLA, self).__init__(n_variables, t0, q0)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([x[0] - tmin,
                                           tmax - x[0]])}
        else:
            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([x[0] - tmin,
                                           tmax - x[0],
                                           x[1] - qmin,
                                           qmax - x[1]])}
        self.bounds = ineq_cons

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = minimize(self.objective, self.x0, method='COBYLA', constraints=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb


class SLSQP(LocalOptimizer):
    def __init__(self, n_variables, t0, q0):
        super(SLSQP, self).__init__('SLSQP', n_variables, t0, q0)


class Trust_constr(LocalOptimizer):
    def __init__(self, n_variables, t0, q0):
        super(Trust_constr, self).__init__('trust-constr', n_variables, t0, q0)


class Basinhopping(Optimizer):
    class MyBounds:
        def __init__(self, xmin, xmax):
            self.xmin = np.array(xmin)
            self.xmax = np.array(xmax)

        def __call__(self, **kwargs):
            x = kwargs['x_new']
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = self.MyBounds([tmin], [tmax])
        else:
            self.bounds = self.MyBounds([tmin, qmin], [tmax, qmax])

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = basinhopping(self.objective, self.x0, accept_test=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb


class BruteForce(Optimizer):
    def __init__(self, n_variables):
        super(BruteForce, self).__init__(n_variables)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = [[tmin, tmax]]
        else:
            self.bounds = [[tmin, tmax], [qmin, qmax]]

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        if self.n_variables == 1:
            (t), res, *_ = brute(self.objective, self.bounds, finish=None, full_output=True)
            q = 0.0
        else:
            (t, q), res, *_ = brute(self.objective, self.bounds, finish=None, full_output=True)
        lcb = -res
        t = np.round(t, 3)
        return t, q, lcb


class DifferentialEvolution(Optimizer):
    def __init__(self, n_variables):
        super(DifferentialEvolution, self).__init__(n_variables)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = Bounds([tmin], [tmax])
        else:
            self.bounds = Bounds([tmin, qmin], [tmax, qmax])

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = differential_evolution(self.objective, bounds=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb


class SHG(Optimizer):
    def __init__(self, n_variables):
        super(SHG, self).__init__(n_variables)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = [[tmin, tmax]]
        else:
            self.bounds = [[tmin, tmax], [qmin, qmax]]

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = shgo(self.objective, bounds=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb


class DualAnnealing(Optimizer):
    def __init__(self, n_variables):
        super(DualAnnealing, self).__init__(n_variables)

    def _set_constraints(self, tmin, tmax, qmin=0.0, qmax=1.0):
        if self.n_variables == 1:
            self.bounds = [[tmin, tmax]]
        else:
            self.bounds = [[tmin, tmax], [qmin, qmax]]

    def maximize(self, func, fargs, tmin, tmax, *args, **kwargs) -> (float, float, float):
        self._set_objective(func, *fargs)
        self._set_constraints(tmin, tmax)
        res = dual_annealing(self.objective, bounds=self.bounds)
        t = np.round(res.x[0], 3)
        if self.n_variables == 1:
            q = 0.0
        else:
            q = res.x[1]
        lcb = -res.fun
        return t, q, lcb
