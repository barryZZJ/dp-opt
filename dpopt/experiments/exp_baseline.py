"""
MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich
"""
from dpopt.classifiers.classifier_factory import *
from dpopt.classifiers.feature_transformer import FlagsFeatureTransformer
from dpopt.classifiers.torch_optimizer_factory import SGDOptimizerFactory
from dpopt.experiments.experiment_runner import ExperimentRunner
from dpopt.experiments.my_experiment import BaselineExperiment
from dpopt.input.input_domain import InputDomain, InputBaseType
from dpopt.input.pattern_generator import PatternGenerator
from dpopt.mechanisms.laplace import LaplaceMechanism
from dpopt.mechanisms.noisy_hist import *
from dpopt.mechanisms.rappor import *
from dpopt.mechanisms.report_noisy_max import *
from dpopt.mechanisms.sparse_vector_technique import *


def class_name(obj):
    return type(obj).__name__.split(".")[-1]


def run_exp_baseline(series_name: str, output_path: str, config: 'DPConfig', log_level, file_level):
    runner = ExperimentRunner(output_path, series_name, log_level=log_level, file_level=file_level)

    def append_exp(mechanism, input_generator: PatternGenerator, output_size, feature_transform=None):
        factory = LogisticRegressionFactory(
            feature_transform=feature_transform,
            in_dimensions=output_size,
            optimizer_factory=SGDOptimizerFactory(learning_rate=0.3, momentum=0.3, step_size=500),
            regularization_weight=0.001,
            epochs=10,
            label=class_name(mechanism))
        runner.experiments.append(
            BaselineExperiment(class_name(mechanism), mechanism, input_generator, factory, config))

    domain = InputDomain(1, InputBaseType.FLOAT, [-10, 10])
    append_exp(LaplaceMechanism(), PatternGenerator(domain, False), 1)

    domain = InputDomain(5, InputBaseType.INT, [0, 10])
    append_exp(NoisyHist1(), PatternGenerator(domain, False), 5)
    append_exp(NoisyHist2(), PatternGenerator(domain, False), 5)

    domain = InputDomain(5, InputBaseType.FLOAT, [-10, 10])
    append_exp(ReportNoisyMax1(), PatternGenerator(domain, True), 1)
    append_exp(ReportNoisyMax2(), PatternGenerator(domain, True), 1)
    append_exp(ReportNoisyMax3(), PatternGenerator(domain, True), 1)
    append_exp(ReportNoisyMax4(), PatternGenerator(domain, True), 1)

    domain = InputDomain(10, InputBaseType.FLOAT, [-10, 10])
    append_exp(SparseVectorTechnique1(c=1, t=0.5), PatternGenerator(domain, True), 20,
               feature_transform=FlagsFeatureTransformer([-1]))
    append_exp(SparseVectorTechnique2(c=1, t=1), PatternGenerator(domain, True), 20,
               feature_transform=FlagsFeatureTransformer([-1]))
    append_exp(SparseVectorTechnique3(c=1, t=1), PatternGenerator(domain, True), 30,
               feature_transform=FlagsFeatureTransformer([-1000.0, -2000.0]))
    append_exp(SparseVectorTechnique4(c=1, t=1), PatternGenerator(domain, True), 20,
               feature_transform=FlagsFeatureTransformer([-1]))
    append_exp(SparseVectorTechnique5(c=1, t=1), PatternGenerator(domain, True), 10)
    append_exp(SparseVectorTechnique6(c=1, t=1), PatternGenerator(domain, True), 10)

    domain = InputDomain(1, InputBaseType.INT, [-10, 10])
    r = Rappor()
    append_exp(r, PatternGenerator(domain, False), r.filter_size)
    otr = OneTimeRappor()
    append_exp(otr, PatternGenerator(domain, False), otr.filter_size)

    runner.run_all(config.n_processes)
