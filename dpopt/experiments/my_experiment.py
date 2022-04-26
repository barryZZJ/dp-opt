from dpopt.attack.dpopt import DPOpt
from dpopt.experiments.experiment_runner import BaseExperiment
from dpopt.utils.my_logging import time_measure, log
from dpopt.search.powersearcher import PowerSearcher


class MyExperiment(BaseExperiment):
    def __init__(self, name, mechanism, input_pair_sampler, classifier_factory, optimizer_generator, config):
        super().__init__(name)
        self.mechanism = mechanism
        self.input_pair_sampler = input_pair_sampler
        self.classifier_factory = classifier_factory
        self.optimizer_generator = optimizer_generator
        self.config = config

    def run(self):
        log.info("using configuration %s", self.config)
        log.data("config", self.config.__dict__)

        attack_opt = DPOpt(self.mechanism, self.classifier_factory, self.optimizer_generator, self.config)


        with time_measure("time_power_searcher"):
            log.debug("running power searcher...")
            opt = PowerSearcher(self.mechanism, attack_opt, self.input_pair_sampler, self.config)
            best_wit = opt.run()
        log.info("finished power searcher, preliminary lcb=%f", best_wit.lcb)

        # log.warning("Skipped estimating eps with high precision to save time")

        with time_measure("time_final_estimate_lcb"):
            log.debug("computing final eps estimate...")
            best_wit.compute_lcb_high_precision(self.mechanism, self.config)

        log.info("done!")
        log.info("> a1      = {}".format(best_wit.a1))
        log.info("> a2      = {}".format(best_wit.a2))
        log.info("> attack  = {}".format(best_wit.attack))
        log.info("> eps lcb = {}".format(best_wit.lcb))


