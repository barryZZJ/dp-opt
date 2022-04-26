import pickle
import tempfile
import os
from typing import Tuple

import numpy as np

from dpopt.search.dpconfig import DPConfig
from dpopt.probability.estimators import PrEstimator, EpsEstimator
from dpopt.mechanisms.abstract import Mechanism
from dpopt.utils.my_logging import log
from dpopt.utils.paths import get_output_directory


class Witness:
    """
    A representation of a DD witness.
    """

    def __init__(self, a1, a2, attack: 'Attack'):
        """
        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input
            attack: the attack
        """
        self.a1 = a1
        self.a2 = a2
        self.attack = attack
        self.lcb = None     # lower bound on epsilon

    def set_lcb(self, lcb):
        self.lcb = lcb

    def compute_lcb_high_precision(self, mechanism: Mechanism, config: DPConfig):
        """
        Computes epsilon and its lower bound using high precision as specified by config.n_final.
        Sets self.eps and self.lcb.
        """
        eps_estimator = EpsEstimator(PrEstimator(mechanism, config.n_final, config, use_parallel_executor=True))
        self.lcb = eps_estimator.compute_lcb_estimates(self.a1, self.a2, self.attack)

    def to_tmp_file(self) -> str:
        """
        Stores the result to a temporary file.

        Returns:
            The path of the created temporary file.
        """
        tmp_dir = get_output_directory("tmp")
        fd, filename = tempfile.mkstemp(dir=tmp_dir)
        log.debug("Storing result to file '%s'", filename)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(self, f)
        return filename

    @staticmethod
    def from_file(filename):
        """
        Loads a DDWitness object from a file with given name.
        """
        log.debug("Loading result from file '%s'", filename)
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        assert(type(obj) == Witness)
        return obj

    def __lt__(self, other):
        return self.lcb < other.lcb

    def __eq__(self, other):
        return self.lcb == other.lcb

    def __str__(self):
        d = {str(k): str(v) for k, v in self.__dict__.items()}
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
