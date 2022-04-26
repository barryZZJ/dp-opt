from dpopt.attack.attack import Attack
from dpopt.classifiers.stable_classifier import StableClassifier

import numpy as np
import os.path


class MlAttack(Attack):
    """
    An attack based on membership inference.
    """

    def __init__(self, classifier: StableClassifier, thresh: float, q: float):
        """
        Create an attack.

        Args:
            classifier: trained membership inference classifier
            thresh: probability threshold
            q: threshold tie-break probability
        """
        self.classifier = classifier
        self.thresh = thresh
        self.q = q

    def check(self, b):
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
    def compute_above_thresh_number_by_pp(thresh, q, sorted_post_probs):
        n_total = len(sorted_post_probs)
        inde, indg = MlAttack._bin_search(sorted_post_probs, thresh, 0, n_total - 1)
        number = MlAttack._compute_above_thresh_number(n_total, q, inde, indg)
        return number

    @staticmethod
    def _compute_above_thresh_number(n_total, q, inde, indg):
        """Count number of samples above and equal to self.t by respective indices."""
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
    def _bin_search(postprobs, thresh, li, ri):
        """Binary search the sorted postprobs.

        Returns:
                inde: Index of the first postprob == thresh. Return len(postprobs) if not exist.
                indg: Index of the first postprob > thresh. Return len(postprobs) if not exist.
        """
        inde = MlAttack._bin_search_eq(postprobs, thresh, li, ri)
        indg = MlAttack._bin_search_gt(postprobs, thresh, inde, ri)
        return inde, indg

    @staticmethod
    def _bin_search_eq(postprobs, thresh, li, ri):
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
    def _bin_search_gt(postprobs, thresh, li, ri):
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
        if self.classifier.state_dict_file is None:
            return "t = {}, q = {}, CLASSIFIER = {}".format(self.thresh, self.q, str(self.classifier))
        return "t = {}, q = {}, CLASSIFIER = {}, file = {}".format(self.thresh, self.q, str(self.classifier), os.path.basename(self.classifier.state_dict_file))

    def to_json(self):
        d = {
            't': self.thresh,
            'q': self.q,
            'CLSSIFIER': str(self.classifier),
            'file': os.path.basename(self.classifier.state_dict_file) if self.classifier.state_dict_file is not None else None
        }
        return d