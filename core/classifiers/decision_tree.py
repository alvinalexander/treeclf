import scipy.stats
import numpy as np
from collections import Counter
from core.types.split_rule import SplitRule
from core.types.leaf_node import LeafNode
from core.types.internal_node import InternalNode

class DecisionTree(object):
    def __init__(self, max_depth= -1, header=None):
        self.max_depth = max_depth
        self.root = None
        self.header = header

    def train(self, X, y):
        self._X_train = X
        self._y_train = y
        self.root = self.build_tree(X, y)

    def predict(self, X):
        #create predictions for a list of samples
        assert self.root != None

        res = []
        for x in X:
            res.append(self.classify(x))

        return np.array(res)

    def classify(self, x):
        """
        Classify a single sample point
        :param x:
        :return:
        """
        def classification_helper(x, node):
            if isinstance(node, LeafNode):
                return node.prediction()

            if node.split_rule.match(x):
                return classification_helper(x, node.true_child)
            else:
                return classification_helper(x, node.false_child)

        return classification_helper(x, self.root)

    def build_tree(self, samples, labels, depth=1):
        info_gain, split_rule = self.find_best_split(samples, labels)

        if info_gain == 0 or depth == self.max_depth:
            return LeafNode(labels)

        else:
            true_samples, true_labels, false_samples, false_labels = self.split(samples, labels, split_rule)

            true_child = self.build_tree(true_samples, true_labels, depth + 1)

            false_child = self.build_tree(false_samples, false_labels, depth + 1)

            return InternalNode(true_child, false_child, split_rule)

    @staticmethod
    def entropy(labels):
        n = len(labels)
        prob_dist = [v / n for v in Counter(labels).values()]
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


    def find_best_split(self, samples, labels):
        """
        Finds the best split by maximizing information gain.
        :param samples:
        :param labels:
        :return:
        """
        assert len(samples) > 0
        assert len(samples) == len(labels)

        num_samples = len(samples)
        num_features = len(samples[0])
        max_info_gain = 0
        best_split_rule = None

        for i in range(num_features):
            values = set([sample[i] for sample in samples])
            for v in values:
                cur_split_rule = SplitRule(i, v)

                true_samples, true_labels, false_samples, false_labels = self.split(samples, labels, cur_split_rule)

                #Ignore if split rule does not partition the data set.
                if len(true_samples) == 0 or len(false_samples) == 0:
                    continue

                cur_info_gain = self.information_gain(i, v, samples, labels)

                if cur_info_gain >= max_info_gain:
                    max_info_gain = cur_info_gain
                    best_split_rule = cur_split_rule

        return max_info_gain, best_split_rule

    def print_tree(self):
        """
        function borrwod from here:
        :return:
        """
        self.print_tree_helper(self.root)

    @staticmethod
    def print_tree_helper(node, spacing=""):

        # Base case: we've reached a leaf
        if isinstance(node, LeafNode):
            print(spacing + "Predict", node.prediction())
            print(spacing + "Predict", node.class_counts)
            return

        # Print the question at this node
        print(spacing + str(node.split_rule))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        DecisionTree.print_tree_helper(node.true_child, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        DecisionTree.print_tree_helper(node.false_child, spacing + "  ")




