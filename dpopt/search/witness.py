import numpy as np

from dpopt.mechanisms.abstract import Mechanism
from dpopt.probability.estimators import PrEstimator, EpsEstimator
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log


class Witness:
    """
    A representation of a witness.
    """

    def __init__(self, a1, a2, attack: 'Attack', optmeth):
        """
        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input
            attack: the attack
            optmeth: the name of optimizer (for logging)
        """
        self.a1 = a1
        self.a2 = a2
        self.attack = attack
        self.optmeth = optmeth
        # low precision
        self.low_lcb = None
        self.low_eps = None
        self.low_p1 = None
        self.low_p2 = None
        # high precision
        self.lcb = None
        self.p1_lcb = None
        self.p2_ucb = None
        self.eps = None
        self.p1 = None
        self.p2 = None

    def set_lcb(self, low_lcb, low_p1=None, low_p2=None, low_eps=None):
        self.low_lcb = low_lcb
        self.low_p1 = low_p1
        self.low_p2 = low_p2
        self.low_eps = low_eps

    def compute_lcb_high_precision(self, mechanism: Mechanism, config: DPConfig):
        """
        Computes epsilon and its lower bound using high precision as specified by config.n_final.
        Sets corresponding values.
        """
        eps_estimator = EpsEstimator(PrEstimator(mechanism, config.n_final, config))
        self.lcb, self.eps, self.p1, self.p2, self.p1_lcb, self.p2_ucb = eps_estimator.compute_lcb_estimates(self.a1,
                                                                                                             self.a2,
                                                                                                             self.attack)
        if self.p1 < self.p2:
            log.warning("probability p1 < p2 for high estimation")

    def __lt__(self, other):
        return self.low_lcb < other.low_lcb

    def __eq__(self, other):
        return self.low_lcb == other.low_lcb

    def __str__(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                d[str(k)] = '%.3f' % v
            else:
                d[str(k)] = str(v)
        return str(d)

    def to_json(self):
        d = {}
        for k, v in self.__dict__.items():
            if k == "attack":
                d[k] = v.to_json()
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d
