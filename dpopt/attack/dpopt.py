import warnings
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
from dpopt.utils.my_multiprocessing import split_by_batch_size


class DPOpt:
    """
    The core algorithm that finds an MlAttack for a given input pair by
    training a classifier and utilizing off-the-shelf optimizers to derive attack thresholds.
    """

    def __init__(self, mechanism: Mechanism, classifier_factory: ClassifierFactory,
                 optimizer_generator: OptimizerGenerator, config: DPConfig):
        """
        Args:
            mechanism: mechanism to attack
            classifier_factory: factory creating instances the classifier to be used for the attack
            optimizer_generator: generator for optimizers
            config: configuration
        """
        self.mechanism = mechanism
        self.classifier_factory = classifier_factory
        self.optimizer_generator = optimizer_generator
        self.config = config
        self.pr_estimator = PrEstimator(mechanism, self.config.n, self.config)
        self.eps_estimator = EpsEstimator(self.pr_estimator)

    def best_attack(self, a1, a2) -> (MlAttack, float, str):
        """
        Construct an attack for given input pair a1, a2.

        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input

        Returns:
            The constructed MlAttack, its lower bound and the name of optimizer that derives this bound
        """
        log.info("Searching best attack for mechanism %s, a1 %s, a2 %s",
                 type(self.mechanism).__name__,
                 str(a1),
                 str(a2))

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

        # Reuse the same sample to find optimal threshold.
        # Only use provided optimizers, then select the optimal one.
        log.debug("Finding optimal threshold...")
        with time_measure("time_find_optimal_threshold_total"):
            t, q, lcb, optmeth = self._find_optimal_threshold(post_probs1, post_probs2)
        log.info("DPOpt selected t = %f, q = %f, (low sample)lcb = %f, produced by %s", t, q, lcb, optmeth)

        return MlAttack(classifier, t, q), lcb, optmeth

    def _find_optimal_threshold(self, sorted_post_probs1, sorted_post_probs2) -> (float, float, float, str):
        """
        Finds threshold t and tie-breaker q such that a given target fraction lies above t.
        Args:
            sorted_post_probs1: 1d array of probabilities, sorted ascending
            sorted_post_probs2: 1d array of probabilities, sorted ascending

        Returns:
            pair (t, q, lcb) of threshold t, tie-breaker q and corresponding lower bound on power lcb,
            and the name of the best optimizer for logging
        """
        # t relies on post_probs1, 2, so we limit t_bound for better performance.
        tmin = min(sorted_post_probs1[0], sorted_post_probs2[0])  # minimum postprob possible
        tmax = sorted_post_probs1[-1]  # maximum postprob possible
        if tmin >= tmax:
            log.error("wrong t_bound: %s, using (0,1)", str((tmin, tmax)))
            tmin = 0
            tmax = 1
        best_t = tmin
        best_q = 0
        best_lcb = -np.inf
        best_optmeth = ''
        log.debug('t_bound %s', str((tmin, tmax)))
        # log.data('t_bound', str((tmin, tmax)))
        if tmin > 0.5 or tmax < 0.5:
            log.warning('Initial guess out of bounds! t_bound %s', str((tmin, tmax)))
        for optimizer in self.optimizer_generator.get_optimizers():
            with time_measure("time_find_optimal_threshold_method_" + str(optimizer)):
                log.debug('Method %s', str(optimizer))
                warnings.filterwarnings("ignore")
                t, q, lcb = optimizer.maximize(self.eps_estimator.compute_lcb_by_pp,
                                               (sorted_post_probs1, sorted_post_probs2), tmin, tmax)
                warnings.filterwarnings("default")
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_t = np.round(t, 3)  # round to 3 decimals for numerical stability, same as DPSniper.
                    best_q = q
                    best_optmeth = str(optimizer)
        return best_t, best_q, best_lcb, best_optmeth

    def _train_classifier(self, a1, a2) -> StableClassifier:
        """
        Trains the classifier for inputs a1, a2.
        """

        def generate_batches():
            for size in split_by_batch_size(self.config.n_train, self.config.training_batch_size):
                yield self._generate_data_batch(a1, a2, size)

        classifier = self.classifier_factory.create()

        log.debug("Training classifier...")
        with time_measure("time_classifier_train"):
            classifier.train(generate_batches())
        log.debug("Done training")

        return classifier

    def _generate_data_batch(self, a1, a2, n) -> Tuple:
        """
        Generates a training data batch of size 2n (n samples for each input a1 and a2).
        """
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
