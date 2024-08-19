import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list,  target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        for count in counts.values():
            prob = count / float(len(rows))
            if prob > 0:
                impurity -= prob * np.log2(prob)

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        # Calculate the entropy of the left and right child nodes
        left_entropy = ID3.entropy(left, left_labels)
        right_entropy = ID3.entropy(right, right_labels)

        # Calculate the weighted impurity of the children nodes
        total_samples = len(left) + len(right)
        weighted_impurity = (len(left) / total_samples) * left_entropy + (len(right) / total_samples) * right_entropy

        # Info gain is the difference between current uncertainty and the weighted impurity
        info_gain_value = current_uncertainty - weighted_impurity

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        true_rows, true_labels, false_rows, false_labels = [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        for idx, row in enumerate(rows):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(labels[idx])
            else:
                false_rows.append(row)
                false_labels.append(labels[idx])

            # Convert lists to numpy arrays for consistency
        true_rows = np.array(true_rows)
        true_labels = np.array(true_labels)
        false_rows = np.array(false_rows)
        false_labels = np.array(false_labels)

        # Calculate the information gain from this partition
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        rows = np.array(rows)
        for i, feature in enumerate(rows.T):
            feature_sample = []
            for sample in rows:
                feature_sample.append(sample[i])
            for f_value in feature_sample:
                question_feat = Question(feature, i, f_value)
                gain_f, true_r_f, true_lab_f, false_r_f, false_lab_f = self.partition(rows, labels, question_feat, current_uncertainty)
                if gain_f >= best_gain:
                    best_gain = gain_f
                    best_true_rows, best_true_labels, best_false_rows, best_false_labels = true_r_f, true_lab_f, false_r_f, false_lab_f
                    best_question = question_feat

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        # Step 1: Find the best split
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(
            rows, labels)

        # Step 2: Base case: If no gain, return a Leaf
        if best_gain == 0:  #self.entropy(rows, labels) == 0 or self.min_for_pruning >= len(rows):
            return Leaf(rows, labels)

        # Step 3: Recursively build the true branch
        true_branch = self.build_tree(best_true_rows, best_true_labels)

        # Step 4: Recursively build the false branch
        false_branch = self.build_tree(best_false_rows, best_false_labels)

        # Step 5: Return a DecisionNode with the best question
        return DecisionNode(best_question, true_branch, false_branch)


    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        #Build the tree that fits the input data and save the root to self.tree_root

        self.tree_root = self.build_tree(x_train, y_train)

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root

        if type(node) is Leaf:
            # if there is only one label then return the label
            if (len(list(node.predictions.keys()))) <= 1:
                return list(node.predictions.keys())[0]
            # If there is multiple labels in the node, then take the majority
            if list(node.predictions.values())[0] < list(node.predictions.values())[1]:
                return list(node.predictions.keys())[1]
            return list(node.predictions.keys())[0]

        if node.question.match(row):
            prediction = self.predict_sample(row, node.true_branch)
        else:
            prediction = self.predict_sample(row, node.false_branch)

        return prediction
    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        #  Implement ID3 class prediction for set of data.

        y_pred = [self.predict_sample(row) for row in rows]

        return np.array(y_pred)
