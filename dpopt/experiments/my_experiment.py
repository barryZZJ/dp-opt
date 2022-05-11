from dpopt.attack.dpopt import DPOpt
from dpopt.attack.dpsniper import DPSniper
from dpopt.experiments.experiment_runner import BaseExperiment
from dpopt.search.ddsearch import DDSearch
from dpopt.search.powersearcher import PowerSearcher
from dpopt.utils.my_logging import time_measure, log


class MyExperiment(BaseExperiment):
    """
    Experiment for PowerSearcher.
    """
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
        log.data('optimizers', [str(opt) for opt in self.optimizer_generator.get_optimizers()])

        attack_opt = DPOpt(self.mechanism, self.classifier_factory, self.optimizer_generator, self.config)

        with time_measure("time_power_searcher"):
            log.debug("running power searcher...")
            opt = PowerSearcher(self.mechanism, attack_opt, self.input_pair_sampler, self.config)
            best_wit = opt.run()
        log.info("finished power searcher, preliminary lcb=%f", best_wit.low_lcb)

        # log.warning("Skipped estimating eps with high precision to save time")

        with time_measure("time_final_estimate_lcb"):
            log.debug("computing final eps estimate...")
            best_wit.compute_lcb_high_precision(self.mechanism, self.config)

        log.data('result_best', best_wit.to_json())  # comp by lcb
        log.info('result best: %s', str(best_wit))

        log.info("done for %s!", type(self.mechanism).__name__.split('.')[-1])


class BaselineExperiment(BaseExperiment):
    """
    Experiment for DD-Search as the baseline.
    """
    def __init__(self, name, mechanism, input_pair_sampler, classifier_factory, config):
        super().__init__(name)
        self.mechanism = mechanism
        self.input_pair_sampler = input_pair_sampler
        self.classifier_factory = classifier_factory
        self.config = config

    def run(self):
        log.info("using configuration %s", self.config)
        log.data("config", self.config.__dict__)

        attack_opt = DPSniper(self.mechanism, self.classifier_factory, self.config)

        with time_measure("time_dd_search"):
            log.debug("running dd-search...")
            opt = DDSearch(self.mechanism, attack_opt, self.input_pair_sampler, self.config)
            best_wit = opt.run()
        log.info("finished dd search, preliminary lcb=%f", best_wit.low_lcb)

        # log.warning("Skipped estimating eps with high precision to save time")

        with time_measure("time_final_estimate_lcb"):
            log.debug("computing final eps estimate...")
            best_wit.compute_lcb_high_precision_origin(self.mechanism, self.config)

        log.data('result_best', best_wit.to_json())  # comp by lcb
        log.info('result best: %s', str(best_wit))

        log.info("done for %s!", type(self.mechanism).__name__.split('.')[-1])
