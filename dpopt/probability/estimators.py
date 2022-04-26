from dpopt.attack.ml_attack import MlAttack
from dpopt.mechanisms.abstract import Mechanism
from dpopt.utils.my_logging import log
from dpopt.utils.my_multiprocessing import the_parallel_executor, split_by_batch_size, split_into_parts
from dpopt.probability.binomial_cdf import lcb, ucb
from dpopt.search.dpconfig import DPConfig

import numpy as np
import math


class PrEstimator:
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

    def compute_pr_estimate_by_pp(self, t, q, sorted_post_probs):
        cnt = MlAttack.compute_above_thresh_number_by_pp(t, q, sorted_post_probs)
        p = cnt / self.n_samples
        return p

    def compute_pr_estimate(self, a, attack):
        """
        Returns:
             An estimate of Pr[M(a) in S]
        """
        if not self.use_parallel_executor:
            frac_count = self._compute_frac_cnt_parallel((self, attack, a, self.n_samples))
        else:
            inputs = [(self, attack, a, batch) for batch in split_into_parts(self.n_samples, self.config.n_processes)]
            frac_counts = the_parallel_executor.execute(self._compute_frac_cnt_parallel, inputs)
            frac_count = math.fsum(frac_counts)

        pr = frac_count / self.n_samples
        return pr

    def _get_samples(self, a, n_samples):
        return self.mechanism.m(a, n_samples=n_samples)

    def _check_attack(self, bs, attack):
        return attack.check(bs)

    def _compute_frac_cnt_parallel(self, args):
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
        eps(a, a', S) = log(Pr[M(a) in S]) - log(Pr[M(a') in S])
    """

    def __init__(self, pr_estimator: PrEstimator):
        """
        Creates an estimator.

        Args:
            pr_estimator: the PrEstimator used to estimate probabilities based on samples
        """
        self.pr_estimator = pr_estimator

    def compute_lcb_estimates(self, a1, a2, attack):
        '''split samples into small batches'''
        p1 = self.pr_estimator.compute_pr_estimate(a1, attack)
        p2 = self.pr_estimator.compute_pr_estimate(a2, attack)
        log.debug("p1=%f, p2=%f", p1, p2)
        log.data("p1", p1)
        log.data("p2", p2)

        if p1 < p2:
            log.warning("probability p1 < p2 for eps estimation")

        # eps = self._compute_eps(p1, p2)
        lcb = self._compute_lcb(p1, p2)
        return lcb

    def compute_lcb_by_pp(self, t, q, post_probs1, post_probs2):
        '''Optimization objective, bivariate'''
        t = np.around(t, 3)
        p1 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs1)
        p2 = self.pr_estimator.compute_pr_estimate_by_pp(t, q, post_probs2)
        return self._compute_lcb(p1, p2)

    @staticmethod
    def _compute_eps(p1, p2):
        if p1 > 0 and p2 == 0:
            eps = float("infinity")
        elif p1 <= 0:
            eps = 0
        else:
            eps = np.log(p1) - np.log(p2)
        return eps

    def _compute_lcb(self, p1, p2):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1-self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1-confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1-confidence)
        return self._compute_eps(p1_lcb, p2_ucb)
