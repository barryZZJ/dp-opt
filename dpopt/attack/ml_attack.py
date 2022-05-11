import os

import numpy as np

from dpopt.attack.attack import Attack
from dpopt.classifiers.stable_classifier import StableClassifier


class MlAttack(Attack):
    """
    An threshold attack based on membership inference.
    Estimation of membership probability is improved with binary search.
    """

    def __init__(self, classifier: StableClassifier, thresh: float, q: float):
        """
        Create an threshold attack.

        Args:
            classifier: trained membership inference classifier
            thresh: probability threshold
            q: threshold tie-break probability
        """
        self.classifier = classifier
        self.thresh = thresh
        self.q = q

    def check(self, b):
        """
        Compute the number of vectorized outputs b that is included in the threshold attack.
        Note this returns the number, not probability.

        Args:
            b:  1d array of shape (n_samples,) if mechanism output is 1-dimensional;
                nd array of shape (n_samples, d) if mechanism output is d-dimensional

        Returns:
            float 1d array of shape (n_samples,) containing numbers <= n_samples
        """
        post_probs = self.predict_probabilities(b)
        post_probs = np.sort(post_probs)
        above_thresh_number = self.compute_above_thresh_number_by_pp(self.thresh, self.q, post_probs)
        return above_thresh_number

    def predict_probabilities(self, b):
        if len(b.shape) == 1:
            # make sure b has shape (n_samples, 1)
            b = np.atleast_2d(b).T

        post_probs = self.classifier.predict_probabilities(b)
        return post_probs

    @staticmethod
    def compute_above_thresh_number_by_pp(thresh, q, sorted_post_probs) -> float:
        """
        Use binary search to compute the number of b included, with provided post probability samples.
        """
        n_total = len(sorted_post_probs)
        inde, indg = MlAttack._bin_search(sorted_post_probs, thresh, 0, n_total - 1)
        number = MlAttack._compute_above_thresh_number(n_total, q, inde, indg)
        return number

    @staticmethod
    def _compute_above_thresh_number(n_total, q, inde, indg):
        """
        Count number of samples above and equal to self.t by respective indices

        Args:
            n_total: Total number of samples.
            q: tie-breaker probability
            inde: Index of the first postprob == thresh.
            indg: Index of the first postprob > thresh.
        """
        if inde >= n_total:
            return 0.0
        if indg >= n_total:
            above = 0.0
            equal = n_total - inde
        else:
            above = n_total - indg
            equal = indg - inde
        return above + q * equal

    @staticmethod
    def _bin_search(postprobs, thresh, li, ri) -> (int, int):
        """Binary search the sorted postprobs.

        Returns:
            inde: Index of the first postprob == thresh. Return len(postprobs) if not exist.
            indg: Index of the first postprob > thresh. Return len(postprobs) if not exist.
        """
        inde = MlAttack._bin_search_eq(postprobs, thresh, li, ri)
        indg = MlAttack._bin_search_gt(postprobs, thresh, inde, ri)
        return inde, indg

    @staticmethod
    def _bin_search_eq(postprobs, thresh, li, ri) -> int:
        if li > ri or postprobs[li] == thresh: return li
        l, r = li, ri
        while l <= r:
            mid = l + ((r - l) >> 1)
            if postprobs[mid] < thresh:
                l = mid + 1
            else:
                r = mid - 1
        return l

    @staticmethod
    def _bin_search_gt(postprobs, thresh, li, ri) -> int:
        if li > ri or postprobs[li] > thresh: return li
        l, r = li, ri
        while l <= r:
            mid = l + ((r - l) >> 1)
            if postprobs[mid] <= thresh:
                l = mid + 1
            else:
                r = mid - 1
        return l

    def __str__(self):
        return "t = {}, q = {}, CLASSIFIER = {}".format(self.thresh, self.q, str(self.classifier))

    def to_json(self):
        d = {
            't': self.thresh,
            'q': self.q,
            'CLSSIFIER': str(self.classifier),
            'file': os.path.basename(self.classifier.state_dict_file or '')
        }
        return d

    def save_model(self):
        self.classifier.to_tmp_file()