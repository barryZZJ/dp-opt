
import math
from typing import Tuple

import numpy as np

from dpopt.attack.ml_attack import MlAttack
from dpopt.classifiers.classifier_factory import ClassifierFactory
from dpopt.classifiers.stable_classifier import StableClassifier
from dpopt.mechanisms.abstract import Mechanism
from dpopt.optimizer.optimizer_generator import OptimizerGenerator
from dpopt.probability.estimators import PrEstimator, EpsEstimator
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log, time_measure
from dpopt.utils.my_multiprocessing import split_into_parts, split_by_batch_size

# change the process of best_attack
class DPOpt:
    """
    The main DP-Sniper algorithm. This optimizer finds an MlAttack for a given input pair by
    training a classifier and deriving attack thresholds.
    """

    def __init__(self, mechanism: Mechanism, classifier_factory: ClassifierFactory, optimizer_generator: OptimizerGenerator, config: DPConfig):
        """
        Create an optimizer.

        Args:
            mechanism: mechanism to attack
            classifier_factory: factory creating instances the classifier to be used for the attack
            config: configuration
        """
        self.mechanism = mechanism
        self.classifier_factory = classifier_factory
        self.optimizer_generator = optimizer_generator
        self.config = config
        self.pr_estimator = PrEstimator(mechanism, self.config.n, self.config)
        self.eps_estimator = EpsEstimator(self.pr_estimator)

    def best_attack(self, a1, a2) -> (MlAttack, float):
        """
        Runs the optimizer to construct an attack for given input pair a1, a2, and corresponding lower bound.

        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input

        Returns:
            The constructed MlAttack, and its lower bound
        """
        log.debug("Searching best attack for mechanism %s, classifier %s...",
                  type(self.mechanism).__name__,
                  type(self.classifier_factory).__name__)

        classifier = self._train_classifier(a1, a2)

        with time_measure("time_sample_for_optimizer"):
            log.debug("Determining optimizer samples...")
            b1s = self.mechanism.m(a1, self.pr_estimator.n_samples)
            b2s = self.mechanism.m(a2, self.pr_estimator.n_samples)
            if len(b1s.shape) == 1:
                # make sure b1s and b2s have shape (n_samples, 1)
                b1s = np.atleast_2d(b1s).T
                b2s = np.atleast_2d(b2s).T

            # p(a1|b1), p(a1|b2)
            post_probs1 = classifier.predict_probabilities(b1s)
            post_probs2 = classifier.predict_probabilities(b2s)

            assert (post_probs1.shape[0] == self.pr_estimator.n_samples)
            assert (post_probs2.shape[0] == self.pr_estimator.n_samples)

            post_probs1 = np.sort(post_probs1)
            post_probs2 = np.sort(post_probs2)

        # find optimal threshold
        log.debug("Finding optimal threshold...")
        with time_measure("time_find_optimal_threshold"):
            t, q, lcb = self._find_optimal_threshold(post_probs1, post_probs2)
        log.debug("Selected t = %f, q = %f, lcb = %f", t, q, lcb)

        # TODO specially log result of dpsniper result (must use same post probs)

        # record dpsniper runtime baseline
        with time_measure("time_sample_for_dpsniper"):
            log.debug("Sampling for dpsniper")
            probabilities = []
            for parallel_size in split_into_parts(self.config.n, self.config.n_processes):
                sequential_probabilities = []
                for sequential_size in split_by_batch_size(parallel_size, self.config.prediction_batch_size):
                    # generate samples from a2
                    b_new = self.mechanism.m(a2, sequential_size)
                    if len(b_new.shape) == 1:
                        # make sure b1 and b2 have shape (n_samples, 1)
                        b_new = np.atleast_2d(b_new).T

                    # compute Pr[a1 | M(a1) = b_new]
                    probabilities_new = classifier.predict_probabilities(b_new)

                    # wrap up
                    sequential_probabilities.append(probabilities_new)

                sequential_probabilities = np.concatenate(sequential_probabilities)
                probabilities.append(sequential_probabilities)

            probabilities = np.concatenate(probabilities)
            probabilities[::-1].sort()  # sorts descending, in-place

            assert (probabilities.shape[0] == self.config.n)

        log.debug("Finding dpsniper optimal threshold...")
        with time_measure("time_dpsniper_find_threshold"):
            thresh, q = DPOpt._dpsniper_find_threshold(probabilities, self.config.c * probabilities.shape[0])
        log.debug("DPsniper selected t = %f, q = %f", t, q)


        return MlAttack(classifier, t, q), lcb

    def _train_classifier(self, a1, a2) -> StableClassifier:
        """
        Trains the classifier for inputs a1, a2.
        """
        def generate_batches():
            for size in split_by_batch_size(self.config.n_train, self.config.training_batch_size):
                yield self._generate_data_batch(a1, a2, size)

        log.debug("Creating classifier...")
        classifier = self.classifier_factory.create()

        log.debug("Training classifier...")
        with time_measure("time_dp_distinguisher_train"):
            classifier.train(generate_batches())
        log.debug("Done training")

        return classifier

    def _generate_data_batch(self, a1, a2, n) -> Tuple:
        """
        Generates a training data batch of size 2n (n samples for each input a1 and a2).
        """
        log.debug("Generating training data batch of size 2*%d...", n)

        b1 = self.mechanism.m(a1, n)
        b2 = self.mechanism.m(a2, n)
        if len(b1.shape) == 1:
            # make sure b1 and b2 have shape (n_samples, 1)
            b1 = np.atleast_2d(b1).T
            b2 = np.atleast_2d(b2).T

        # rows = samples, columns = features
        x = np.concatenate((b1, b2), axis=0)

        # 1d array of labels
        y = np.zeros(2 * n)
        y[n:] = 1

        return x, y

    def _find_optimal_threshold(self, sorted_post_probs1, sorted_post_probs2) -> (float, float, float):
        """
        Finds threshold t and tie-breaker q such that a given target fraction lies above t.
        Args:
            sorted_post_probs1: 1d array of probabilities, sorted ascending
            sorted_post_probs2: 1d array of probabilities, sorted ascending

        Returns:
            pair (t, q, lcb) of threshold t, tie-breaker q and corresponding lower bound on power lcb.
        """
        tmin = min(sorted_post_probs1[0], sorted_post_probs2[0])  # minimum postprob possible
        tmax = sorted_post_probs1[-1]  # maximum postprob possible
        best_t = tmin
        best_q = 0
        best_lcb = -np.inf
        for optimizer in self.optimizer_generator.get_optimizers():
            with time_measure("time_find_optimal_threshold_method_" + str(optimizer)):
                t, q, lcb = optimizer.maximize(self.eps_estimator.compute_lcb_by_pp, (sorted_post_probs1, sorted_post_probs2), tmin, tmax)
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_t = np.round(t, 3)  # round to 3 decimals for numerical stability, same as DPSniper.
                    best_q = q
        return best_t, best_q, best_lcb

    @staticmethod
    def _dpsniper_find_threshold(sorted_probs, target: float) -> (float, float):
        """
        Finds threshold t and tie-breaker q such that a given target fraction lies above t.

        Args:
            sorted_probs: 1d array of probabilities, sorted descending
            target: target fraction

        Returns:
            pair (t, q) of threshold t and tie-breaker q
        """
        thresh = sorted_probs[min(math.floor(target), sorted_probs.shape[0] - 1)]

        # find number of samples strictly above thresh
        n_above = np.sum(sorted_probs > thresh)

        # find number of samples equal to thresh
        n_equal = np.sum(sorted_probs == thresh)

        # split remaining weight
        q = (target - n_above) / n_equal

        return thresh, q