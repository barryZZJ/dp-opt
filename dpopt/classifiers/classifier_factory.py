from dpopt.classifiers.stable_classifier import StableClassifier
from dpopt.classifiers.logistic_regression import LogisticRegression


class ClassifierFactory:
    """
    A factory constructing classifiers.
    """

    def __init__(self, clazz: type, **args):
        """
        Construct a factory which creates instances of a fixed type using fixed arguments.

        Args:
            clazz: the type of the classifier
            **args: the arguments to be passed when constructing the classifier
        """
        self.clazz = clazz
        self.args = args

    def create(self) -> StableClassifier:
        """
        Returns:
            a new StableClassifier
        """
        return self.clazz(**self.args)


class LogisticRegressionFactory(ClassifierFactory):
    """
    A factory constructing logistic regression classifiers.
    """

    def __init__(self, **args):
        """
        Args:
            **args: arguments to be passed when constructing the classifier (see LogisticRegression.__init__)
        """
        super().__init__(LogisticRegression, **args)

