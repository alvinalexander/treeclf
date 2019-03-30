from collections import Counter
class LeafNode(object):

    def __init__(self, labels):
        self.labels = labels
        self.class_counts = Counter(labels)
        #Todo: make sure this is correct.
        self.most_likely_class = max(self.class_counts.keys(), key= lambda x: self.class_counts[x])

    def prediction(self):
        return self.most_likely_class

    #Todo: define prediction function.


