import unittest
import numpy as np
from core.classifiers.decision_tree import DecisionTree
from core.types.split_rule import SplitRule
class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.train_data, self.train_labels, self.test_data, self.test_labels = (np.array([]), np.array([]),
                                                                                np.array([]), np.array([]))
        self.params = {
            'max_depth': 1,
        }
        self.dt = DecisionTree(self.params)


    def test_train(self):
        clf = self.dt.train(self.train_data, self.train_labels)

    def test_entropy(self):
        labels_1 = np.array([0, 0, 0, 0])
        labels_2 = np.array([1,1,0,0])
        labels_3 = np.array([1,2,3,4])
        self.assertEqual(self.dt.entropy(labels_1), 0)
        self.assertEqual(self.dt.entropy(labels_2), 1)
        self.assertEqual(self.dt.entropy(labels_3), 2)

    def test_information_gain(self):
        test_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        test_labels = np.array([0, 1, 0, 1])

    def test_split(self):
        test_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        test_labels = np.array([0, 1, 0, 1])
        split_rule = SplitRule(0, 3)
        true_rows, true_labels,  false_rows, false_labels = self.dt.split(test_data, test_labels, split_rule)
        truth = [s for s in test_data if s[0] == 0]
        for s in truth:
            self.assertTrue(s in false_rows)

        self.assertEqual(len(false_rows), 3)

    def test_information_gain(self):
        test_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        test_labels = np.array([0, 0, 0, 1])
        test_data_1 = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [0, 5, 6],
            [0, 3, 5]
        ])
        test_labels_1 = np.array([0, 0, 0, 1])
        information_gain_1 = self.dt.information_gain(0, 3, test_data, test_labels)
        self.assertGreater(information_gain_1, 0)

        information_gain_2 = self.dt.information_gain(0, 0, test_data_1, test_labels_1)
        print(information_gain_2)
        self.assertEqual(information_gain_2, 0)



if __name__ == '__main__':
    unittest.main()
