"""
Improve estimation methods by implementing binary search
"""
import math

import numpy as np

from dpopt.attack.ml_attack import MlAttack
from dpopt.mechanisms.abstract import Mechanism
from dpopt.probability.binomial_cdf import lcb, ucb
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log
from dpopt.utils.my_multiprocessing import split_by_batch_size


class PrEstimator:
    """
    Class for computing an estimate of Pr[M(a) in S].
    """

    def __init__(self, mechanism: Mechanism, n_samples: int, config: DPConfig):
        """
        Creates an estimator.

        Args:
            mechanism: mechanism
            n_samples: number of samples used to estimate the probability
        """
        self.mechanism = mechanism
        self.n_samples = n_samples
        self.config = config

    def compute_pr_estimate_by_pp(self, t, q, sorted_post_probs):
        """
        Returns:
             An estimate of Pr[M(a) in S] with provided post probability samples.
        """
        cnt = MlAttack.compute_above_thresh_number_by_pp(t, q, sorted_post_probs)
        p = cnt / self.n_samples
        return p

    def compute_pr_estimate(self, a, attack):
        """
        Returns:
             An estimate of Pr[M(a) in S] with fresh samples
        """
        frac_count = self._compute_frac_cnt((self, attack, a, self.n_samples))
        pr = frac_count / self.n_samples
        log.debug(f'{frac_count} / {self.n_samples} = {pr}')
        return pr

    def _get_samples(self, a, n_samples):
        """
        Returns:
             1d array of shape (n_samples,)
        """
        return self.mechanism.m(a, n_samples=n_samples)

    def _check_attack(self, bs, attack: MlAttack):
        return attack.check(bs)

    def _compute_frac_cnt(self, args):
        pr_estimator, attack, a, n_samples = args

        frac_counts = []
        for sequential_size in split_by_batch_size(n_samples, pr_estimator.config.prediction_batch_size):
            bs = pr_estimator._get_samples(a, sequential_size)
            temp_cnt = pr_estimator._check_attack(bs, attack)
            frac_counts.append(temp_cnt)
        frac_count = math.fsum(frac_counts)
        return frac_count


class EpsEstimator:
    """
    Class for computing an estimate of
        eps(a, a', S) = log(Pr[M(a1) in S]) - log(Pr[M(a2) in S])
    """

    def __init__(self, pr_estimator: PrEstimator):
        """
        Creates an estimator.

        Args:
            pr_estimator: the PrEstimator used to estimate probabilities based on samples
        """
        self.pr_estimator = pr_estimator

    def compute_lcb_estimates(self, a1, a2, attack) -> (float, float, float, float, float, float):
        """
        Estimates eps(a1, a2, attack) with fresh samples.

        Returns:
            eps: the eps estimate
            lcb: a lower confidence bound for eps
            p1: estimated probability of Pr[M(a1) in S]
            p2: estimated probability of Pr[M(a2) in S]
            p1_lcb: lower confidence bound for p1
            p2_ucb: upper confidence bound for p2
        """
        """split samples into small batches"""
        p1 = self.pr_estimator.compute_pr_estimate(a1, attack)
        p2 = self.pr_estimator.compute_pr_estimate(a2, attack)
        log.debug("my  p1=%f, p2=%f", p1, p2)
        log.data("p1", p1)
        log.data("p2", p2)

        if p1 < p2:
            log.warning("probability p1 < p2 for eps estimation")

        eps = self._compute_eps(p1, p2)
        lcb, p1_lcb, p2_ucb = self._compute_lcb(p1, p2, return_probs=True)
        return lcb, eps, p1, p2, p1_lcb, p2_ucb

    def compute_lcb_by_pp(self, t, q, post_probs1, post_probs2) -> float:
        """
        Estimates lower bound on eps(a1, a2, attack) with provided post probability samples.

        Returns:
            lcb: a lower confidence bound for eps
        """
        t = np.around(t, 3)
        p1 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs1)
        p2 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs2)
        lcb, p1_lcb, p2_ucb = self._compute_lcb(p1, p2)
        return lcb

    @staticmethod
    def _compute_eps(p1, p2):
        if p1 > 0 and p2 == 0:
            eps = float("infinity")
        elif p1 <= 0:
            eps = 0
        else:
            eps = np.log(p1) - np.log(p2)
        return eps

    def _compute_lcb(self, p1, p2, return_probs=True):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1 - self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1 - confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1 - confidence)
        if return_probs:
            return self._compute_eps(p1_lcb, p2_ucb), p1_lcb, p2_ucb
        else:
            return self._compute_eps(p1_lcb, p2_ucb)
