from typing import List
from dpopt.optimizer.abstract import Optimizer


class OptimizerGenerator:
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers

    def get_optimizers(self):
        return self.optimizers
