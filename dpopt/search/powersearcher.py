from dpopt.attack.dpopt import DPOpt
from dpopt.input.input_pair_generator import InputPairGenerator
from dpopt.mechanisms.abstract import Mechanism
from dpopt.probability.estimators import PrEstimator
from dpopt.search.dpconfig import DPConfig
from dpopt.search.witness import Witness
from dpopt.utils.my_logging import log, time_measure


def class_name(obj):
    return type(obj).__name__.split(".")[-1]


class PowerSearcher:
    """
    The external algorithm for producing lower bound of differential privacy.
    """

    def __init__(self,
                 mechanism: Mechanism,
                 attack_optimizer: DPOpt,
                 input_generator: InputPairGenerator,
                 config: DPConfig):
        """
        Creates the optimizer.

        Args:
            mechanism: mechanism to test
            attack_optimizer: optimizer finding attacks for given input pairs
            input_generator: generator of input pairs
            config: configuration
        """
        self.mechanism = mechanism
        self.attack_optimizer = attack_optimizer
        self.input_generator = input_generator
        self.config = config
        self.pr_estimator = PrEstimator(mechanism, self.config.n, self.config)

    def run(self) -> Witness:
        """
        Runs the optimizer and returns the result.
        """
        with time_measure("time_power_searcher_all_inputs"):
            wits = self._compute_results_for_all_inputs()

        for wit in wits:
            log.data("result_temp", wit.to_json())
            log.info('result temp : %s', str(wit))

        # find best result
        best_wit = None
        for wit in wits:
            if best_wit is None or wit > best_wit:
                best_wit = wit

        return best_wit

    def _compute_results_for_all_inputs(self):
        results = []
        for (a1, a2) in self.input_generator.get_input_pairs():
            result = self._one_input_pair(a1, a2)
            results.append(result)
        return results

    # @staticmethod
    def _one_input_pair(self, a1, a2):
        with time_measure('time_dpopt_method'):
            attack, lcb, optmeth = self.attack_optimizer.best_attack(a1, a2)

        wit = Witness(a1, a2, attack, optmeth)
        wit.set_lcb(lcb)
        attack.save_model()

        log.debug("Done searching best attack for mechanism %s, a1 %s, a2 %s",
                  type(self.mechanism).__name__,
                  str(a1),
                  str(a2))
        return wit
