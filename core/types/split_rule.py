class SplitRule(object):
    def __init__(self, column_idx, value, column_name=''):
        self.column_idx = column_idx
        self.value = value
        self.column_name = column_name

    def match(self, sample):
        #NOTE: here we assume that the design matrix only has numeric columns.
        #this means that categorical data must be encoded using one hot encoding
        return sample[self.column_idx] >= self.value

    def __repr__(self):
        condition = '>='
        if self.column_name:
            return "%s %s %s ?" % (self.column_name, condition, str(self.value))
        else:
            return "column[%d] %s %s ?" % (self.column_idx, condition, str(self.value))


    @staticmethod
    def is_numeric(value):
        return isinstance(value, int) or isinstance(value, float)