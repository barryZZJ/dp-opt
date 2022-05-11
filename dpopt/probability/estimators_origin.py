"""
The estimation method of original dpsniper, used for baseline.
MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich
"""
import math

import numpy as np

from dpopt.attack.attack import Attack
from dpopt.mechanisms.abstract import Mechanism
from dpopt.probability.binomial_cdf import lcb, ucb
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log
from dpopt.utils.my_multiprocessing import the_parallel_executor, split_by_batch_size, split_into_parts


class PrEstOrigin:
    """
    Class for computing an estimate of Pr[M(a) in S].
    """

    def __init__(self, mechanism: Mechanism, n_samples: int, config: DPConfig, use_parallel_executor: bool = False):
        """
        Creates an estimator.

        Args:
            mechanism: mechanism
            n_samples: number of samples used to estimate the probability
            use_parallel_executor: whether to use the global parallel executor for probability estimation.
        """
        self.mechanism = mechanism
        self.n_samples = n_samples
        self.use_parallel_executor = use_parallel_executor
        self.config = config

    def compute_pr_estimate(self, a, attack: Attack) -> float:
        """
        Returns:
             An estimate of Pr[M(a) in S]
        """
        if not self.use_parallel_executor:
            frac_cnt = PrEstOrigin._compute_frac_cnt((self, attack, a, self.n_samples))
        else:
            inputs = [(self, attack, a, batch) for batch in split_into_parts(self.n_samples, self.config.n_processes)]
            res = the_parallel_executor.execute(PrEstOrigin._compute_frac_cnt, inputs)
            frac_cnt = math.fsum(res)
        pr = frac_cnt / self.n_samples
        log.debug(f'{frac_cnt} / {self.n_samples} = {pr}')
        return pr

    def _get_samples(self, a, n_samples):
        return self.mechanism.m(a, n_samples=n_samples)

    def _check_attack(self, bs, attack):
        return attack.check(bs)

    @staticmethod
    def _compute_frac_cnt(args):
        pr_estimator, attack, a, n_samples = args

        frac_counts = []
        for sequential_size in split_by_batch_size(n_samples, pr_estimator.config.prediction_batch_size):
            bs = pr_estimator._get_samples(a, sequential_size)
            res = pr_estimator._check_attack(bs, attack)
            frac_counts += [math.fsum(res)]

        return math.fsum(frac_counts)

    def get_variance(self):
        """
        Returns the variance of estimations
        """
        return 1.0 / (4.0 * self.n_samples)


class EpsEstOrigin:
    """
    Class for computing an estimate of
        eps(a, a', S) = log(Pr[M(a) in S]) - log(Pr[M(a') in S])
    """

    def __init__(self, pr_estimator: PrEstOrigin):
        """
        Creates an estimator.

        Args:
            pr_estimator: the PrEstOrigin used to estimate probabilities based on samples
        """
        self.pr_estimator = pr_estimator

    def compute_eps_estimate(self, a1, a2, attack: Attack, return_prob_bounds=False) -> (float, float, float, float):
        """
        Estimates eps(a2, a2, attack) using samples.

        Returns:
            eps: the eps estimate
            lcb: a lower confidence bound for eps
            p1: estimated probability of Pr[M(a1) in S]
            p2: estimated probability of Pr[M(a2) in S]
            p1_lcb: lower confidence bound for p1
            p2_ucb: upper confidence bound for p2
        """
        p1 = self.pr_estimator.compute_pr_estimate(a1, attack)
        p2 = self.pr_estimator.compute_pr_estimate(a2, attack)
        log.debug("ori p1=%f, p2=%f", p1, p2)
        log.data("p1", p1)
        log.data("p2", p2)

        if p1 < p2:
            log.warning("probability p1 < p2 for eps estimation")

        eps = self._compute_eps(p1, p2)
        if return_prob_bounds:
            lcb, p1_lcb, p2_ucb = self._compute_lcb(p1, p2, return_prob_bounds)
            return eps, lcb, p1, p2, p1_lcb, p2_ucb
        else:
            lcb = self._compute_lcb(p1, p2, return_prob_bounds)
            return eps, lcb, p1, p2

    @staticmethod
    def _compute_eps(p1, p2):
        if p1 > 0 and p2 == 0:
            eps = float("infinity")
        elif p1 <= 0:
            eps = 0
        else:
            eps = np.log(p1) - np.log(p2)
        return eps

    def _compute_lcb(self, p1, p2, return_probs=False):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1 - self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1 - confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1 - confidence)
        if return_probs:
            return self._compute_eps(p1_lcb, p2_ucb), p1_lcb, p2_ucb
        else:
            return self._compute_eps(p1_lcb, p2_ucb)
