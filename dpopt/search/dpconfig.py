"""
MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich
"""
import multiprocessing


class DPConfig:
    """
    An object holding global configuration.
    """

    def __init__(self,
                 c=0.01,
                 n_train=10_700_000,
                 n_check=10_700_000,
                 n_final=200_000_000,
                 prediction_batch_size=10_000,
                 training_batch_size=100_000,
                 n_processes=1,
                 confidence=0.9):
        """
        Args:
            c: the probability lower bound (DP-Sniper avoids estimating probabilities below c)
            n_train: number of input samples used to train the machine learning attack model
            n_check: number of samples used to select the probability threshold during attack optimization
            n_final: number of samples used to estimate probabilities P[M(a) in S] with high precision
            prediction_batch_size: number of samples in a single batch for prediction
            training_batch_size: number of samples in a single batch for training
            n_processes: number of processes to use for parallelization
            confidence: requested confidence for the computed lower bound on epsilon

        Note:
            Small batch sizes lead to higher runtime, while large batch sizes require more memory
        """
        self.c = c
        self.n_train = n_train
        self.n = n_check
        self.n_final = n_final
        self.prediction_batch_size = prediction_batch_size
        self.training_batch_size = training_batch_size
        self.n_processes = n_processes
        self.confidence = confidence

        if self.n_final >= 10000 and self.n_final % 1000 != 0:
            raise ValueError("Sample sizes >= 10000 must be multiples of 1000 for numerical stability")

    def __str__(self):
        s = "{"
        for name, value in self.__dict__.items():
            s = s + "{}={}, ".format(name, str(value))
        return s + "}"
