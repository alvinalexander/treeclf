class SplitRule(object):
    def __init__(self, column_idx, value):
        self.column_idx = column_idx
        self.value = value

    def match(self, sample):
        #NOTE: here we assume that the design matrix only has numeric columns.
        #this means that categorical data must be encoded using one hot encoding
        return sample[self.column_idx] >= self.value


    @staticmethod
    def is_numeric(value):
        return isinstance(value, int) or isinstance(value, float)