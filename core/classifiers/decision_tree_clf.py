from core.classifiers.decision_tree import DecisionTree
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, max_depth=-1):
        self.max_depth = max_depth
        self.dt = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.dt = DecisionTree(max_depth=self.max_depth)
        # Return the classifier
        self.dt.train(self.X_, self.y_)

        return self

    def predict(self, X):
      # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
      # Input validation
        X = check_array(X)
        return self.dt.predict(X)

    def print_tree(self):
        if self.dt != None:
            self.dt.print_tree()

