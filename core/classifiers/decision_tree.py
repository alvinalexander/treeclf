import scipy.stats
import numpy as np
from collections import Counter
from core.types.split_rule import SplitRule

class DecisionTree(object):
    def __init__(self, max_depth = None):
        self._max_depth = max_depth
        self._root = None

    def train(self, X, y):
        self._X_train = X
        self._y_train = y

    @staticmethod
    def entropy(labels):
        n = len(labels)
        prob_dist = [ v/n for v in Counter(labels).values()]
        return scipy.stats.entropy(pk=prob_dist, base=2)


    @staticmethod
    def split(samples, labels, split_rule):
        """
        :param samples: A list of sample points in row order
        :param split_rule: A split rule object
        :type SplitRule
        :return: two lists of samples partitioned by the split rule
        """

        true_samples, true_labels, false_samples, false_labels = [], [], [], []
        for i in range(len(samples)):
            if split_rule.match(samples[i]):
                true_samples.append(samples[i])
                true_labels.append(labels[i])
            else:
                false_samples.append(samples[i])
                false_labels.append(labels[i])

        return np.array(true_samples), np.array(true_labels), np.array(false_samples), np.array(false_labels)


    def information_gain(self, feature_idx, value, data, labels):
        """
        Calculates the information gain achieved by performing a split according feature_idx and value
        :param feature_idx:
        :param value:
        :param data:
        :param labels:
        :return:
        """
        n = len(labels)
        current_entropy = self.entropy(labels)
        true_data, true_labels, false_data, false_labels = self.split(data, labels, SplitRule(feature_idx, value))
        entropy_true, entropy_false = self.entropy(true_labels), self.entropy(false_labels)
        size_true, size_false = len(true_labels), len(false_labels)
        return current_entropy - (size_true * entropy_true + size_false * entropy_false) / n


