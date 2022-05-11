"""
MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich
"""
import numpy as np

from dpopt.attack.ml_attack_origin import MlAttackOrigin
from dpopt.mechanisms.abstract import Mechanism
from dpopt.probability.estimators_origin import EpsEstOrigin
from dpopt.probability.estimators_origin import PrEstOrigin
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log


class DDWitness:
    """
    A representation of a DD witness.
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
        self.low_eps = None  # estimate of epsilon
        self.low_lcb = None
        self.low_p1 = None
        self.low_p2 = None
        self.lcb = None  # lower bound on epsilon
        self.eps = None
        self.p1 = None
        self.p2 = None
        self.p1_lcb = None
        self.p2_ucb = None
        self.optmeth = optmeth

    def set_lcb(self, low_lcb, low_p1, low_p2, low_eps):
        self.low_lcb = low_lcb
        self.low_p1 = low_p1
        self.low_p2 = low_p2
        self.low_eps = low_eps

    def compute_lcb_high_precision_origin(self, mechanism: Mechanism, config: DPConfig):
        """
        Computes epsilon and its lower bound using high precision as specified by config.n_final.
        Sets corresponding values.
        """
        eps_estimator = EpsEstOrigin(PrEstOrigin(mechanism, config.n, config))
        self.lcb, self.eps, self.p1, self.p2, self.p1_lcb, self.p2_ucb = eps_estimator.compute_eps_estimate(self.a1,
                                                                                                            self.a2,
                                                                                                            self.attack,
                                                                                                            True)
        if self.p1 < self.p2:
            log.warning("probability p1 < p2 for high estimation")

    def comput_lcb_low_precision_origin(self, mechanism, config, att_ori: MlAttackOrigin):
        """
        Computes epsilon and its lower bound using low precision as specified by config.n.
        Sets corresponding values.
        """
        eps_estimator = EpsEstOrigin(PrEstOrigin(mechanism, config.n, config))
        self.low_eps, self.low_lcb, self.low_p1, self.low_p2 = eps_estimator.compute_eps_estimate(self.a1, self.a2,
                                                                                                  att_ori, False)

    def __lt__(self, other):
        return self.low_eps < other.low_eps

    def __eq__(self, other):
        return self.low_eps == other.low_eps

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
