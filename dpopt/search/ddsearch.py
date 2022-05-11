"""
MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich
"""
from dpopt.attack.dpsniper import DPSniper
from dpopt.input.input_pair_generator import InputPairGenerator
from dpopt.mechanisms.abstract import Mechanism
from dpopt.search.ddwitness import DDWitness
from dpopt.search.dpconfig import DPConfig
from dpopt.utils.my_logging import log, time_measure


def class_name(obj):
    return type(obj).__name__.split(".")[-1]


class DDSearch:
    """
    The main DD-Search algorithm for testing differential privacy.
    """

    def __init__(self,
                 mechanism: Mechanism,
                 attack_optimizer: DPSniper,
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

    def run(self) -> DDWitness:
        """
        Runs the optimizer and returns the result.
        """
        with time_measure("time_dd_search_all_inputs"):
            wits = self._compute_results_for_all_inputs()

        for dpsniper_wit in wits:
            log.data("result_temp_dpsniper", dpsniper_wit.to_json())
            log.info('result temp dpsniper: %s', str(dpsniper_wit))

        # find best result
        best_dpsniper_wit = None
        for dpsniper_wit in wits:
            if best_dpsniper_wit is None or dpsniper_wit > best_dpsniper_wit:
                best_dpsniper_wit = dpsniper_wit

        return best_dpsniper_wit

    def _compute_results_for_all_inputs(self):
        # log.debug("generating inputs...")
        results = []
        for (a1, a2) in self.input_generator.get_input_pairs():
            result = self._one_input_pair(a1, a2)
            results.append(result)

        return results

    # @staticmethod
    def _one_input_pair(self, a1, a2):
        # set context for child process
        log.debug("Selecting best attack...")
        with time_measure("time_dpsniper_method"):
            dpsniper_att = self.attack_optimizer.best_attack(a1, a2)

        # record dpsniper wit
        dpsniper_wit = DDWitness(a1, a2, dpsniper_att, 'dpsniper')
        dpsniper_att.save_model()

        with time_measure("time_estimate_eps"):
            dpsniper_wit.comput_lcb_low_precision_origin(self.mechanism, self.config, dpsniper_att)

        log.debug("Done searching best attack for mechanism %s, a1 %s, a2 %s",
                  type(self.mechanism).__name__,
                  str(a1),
                  str(a2))
        return dpsniper_wit
