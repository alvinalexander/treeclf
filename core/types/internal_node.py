
class InternalNode(object):

    def __init__(self, true_child, false_child, split_rule):
        self.true_child = true_child
        self.false_child = false_child
        self.split_rule = split_rule


