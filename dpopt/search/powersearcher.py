from dpopt.attack.dpopt import DPOpt
from dpopt.input.input_pair_generator import InputPairGenerator
from dpopt.probability.estimators import PrEstimator, EpsEstimator
from dpopt.mechanisms.abstract import Mechanism
from dpopt.utils.my_logging import log, time_measure
from dpopt.utils.my_multiprocessing import the_parallel_executor
from dpopt.search.dpconfig import DPConfig
from dpopt.search.witness import Witness

import os


def class_name(obj):
    return type(obj).__name__.split(".")[-1]


class PowerSearcher:
    """
    The main DD-Search algorithm for testing differential privacy.
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

        # find best result
        best_wit = None
        for wit in wits:
            if best_wit is None or wit > best_wit:
                best_wit = wit

        log.data('best_result', best_wit.to_json())

        return best_wit

    def _compute_results_for_all_inputs(self):
        log.debug("generating inputs...")
        inputs = []
        for (a1, a2) in self.input_generator.get_input_pairs():
            log.debug("%s, %s", a1, a2)
            inputs.append((self, a1, a2))

        log.debug("submitting parallel tasks...")
        result_files = the_parallel_executor.execute(PowerSearcher._one_input_pair, inputs)
        log.debug("parallel tasks done!")

        results = []
        for filename in result_files:
            cur = Witness.from_file(filename)
            os.remove(filename)
            results.append(cur)
        return results


    @staticmethod
    def _one_input_pair(task):
        # set context for child process
        self, a1, a2 = task
        log.append_context(class_name(self.mechanism))

        # log.debug("a1={}".format(a1))
        # log.debug("a2={}".format(a2))

        log.debug("selecting attack...")
        with time_measure("time_dp_opt"):
            attack, lcb = self.attack_optimizer.best_attack(a1, a2)
        log.debug("attack: %s", attack)
        log.debug("current low sample lcb: %f", lcb)

        #TODO simply return, don't save to file?
        wit = Witness(a1, a2, attack)
        wit.set_lcb(lcb)

        log.debug("storing result...")
        filename = wit.to_tmp_file()

        # log.data("temp_result", wit.to_json())

        log.debug("done!")
        log.pop_context()
        return filename

