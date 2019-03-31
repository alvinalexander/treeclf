import unittest
import numpy as np
from core.classifiers.decision_tree import DecisionTree
from core.types.split_rule import SplitRule
from core.types.leaf_node import LeafNode

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.train_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        self.train_labels = np.array([0, 1, 0, 3])

        self.dt = DecisionTree()
        self.dt.train(self.train_data, self.train_labels)
        print("===== INITIAL TREE =====")
        self.dt.print_tree()

    def test_predict(self):
        """
        Test that the predict methods works as expected.
        :return:
        """
        data = np.array([
            [1, 1, 5],
            [4, 3, 5],
            [5, 5, 6],
            [3, 6, 10]
        ])
        predictions = self.dt.predict(data)
        self.assertEqual(len(predictions), len(data))
        for p in predictions:
            self.assertTrue(p in set(self.train_labels))

        print("Predictins", predictions)


    def test_entropy(self):
        labels_1 = np.array([0, 0, 0, 0])
        labels_2 = np.array([1,1,0,0])
        labels_3 = np.array([1,2,3,4])
        self.assertEqual(self.dt.entropy(labels_1), 0)
        self.assertEqual(self.dt.entropy(labels_2), 1)
        self.assertEqual(self.dt.entropy(labels_3), 2)

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

    def test_find_best_split(self):
        test_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        test_labels = np.array([0, 0, 0, 1])

        max_info_gain, best_split_rule = self.dt.find_best_split(test_data, test_labels)
        #print(max_info_gain)
        #print(best_split_rule)

    def test_build_tree(self):
        def max_depth(tree):
            if isinstance(tree, LeafNode):
                return 1
            else:
                return 1 + max(max_depth(tree.true_child), max_depth(tree.false_child))


        test_data = np.array([
            [0, 1, 5],
            [0, 3, 5],
            [3, 5, 6],
            [0, 3, 5]
        ])
        test_labels = np.array([0, 0, 0, 1])

        root = self.dt.build_tree(test_data, test_labels)
        self.dt.print_tree_helper(root)

        #test max_depth
        self.dt.max_depth = 1
        root = self.dt.build_tree(test_data, test_labels)
        self.dt.print_tree_helper(root)
        self.assertLessEqual(max_depth(root), 1)


        #Todo Test build tree with max_depth.
        #TODO: fit and predict.




if __name__ == '__main__':
    unittest.main()
