#!/usr/bin/env python3
"""
Decision Tree and Random Forest implementation
with Node and Leaf classes.
"""

import numpy as np


class Node:
    """A node class that generalizes everything including root and leaves."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Find maximum depth below."""
        if self.is_leaf:
            return self.depth

        left = (
            self.left_child.max_depth_below()
            if self.left_child
            else self.depth
        )
        right = (
            self.right_child.max_depth_below()
            if self.right_child
            else self.depth
        )
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below."""
        if self.is_leaf:
            return 1

        left = (
            self.left_child.count_nodes_below(only_leaves)
            if self.left_child
            else 0
        )
        right = (
            self.right_child.count_nodes_below(only_leaves)
            if self.right_child
            else 0
        )

        if only_leaves:
            return left + right

        return 1 + left + right

    def pred(self, x):
        """Return prediction."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Terminal node."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def pred(self, x):
        return self.value


class DecisionTree:
    """Decision Tree class."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def pred(self, x):
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Train tree."""
        self.explanatory = explanatory
        self.target = target

        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.root.sub_population = np.ones_like(target, dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose:
            print(
                f"""
Training finished
Depth: {self.depth()}
Nodes: {self.count_nodes()}
Leaves: {self.count_nodes(True)}
Accuracy: {self.accuracy(explanatory, target)}
"""
            )

    def fit_node(self, node):
        """Recursive fitting."""
        sub_target = self.target[node.sub_population]

        if (
            node.sub_population.sum() <= self.min_pop
            or node.depth >= self.max_depth
            or np.all(sub_target == sub_target[0])
        ):
            value = np.bincount(sub_target).argmax()
            leaf = Leaf(value, depth=node.depth)
            leaf.sub_population = node.sub_population
            return leaf

        node.feature, node.threshold = self.split_criterion(node)

        mask = (
            self.explanatory[:, node.feature] > node.threshold
        )

        left_population = node.sub_population & mask
        right_population = node.sub_population & (~mask)

        node.left_child = self.get_leaf_child(
            node, left_population
        )
        node.right_child = self.get_leaf_child(
            node, right_population
        )

    def get_leaf_child(self, node, sub_population):
        value = np.bincount(self.target[sub_population]).argmax()
        leaf = Leaf(value, depth=node.depth + 1)
        leaf.sub_population = sub_population
        return leaf

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def random_split_criterion(self, node):
        """Random split."""
        diff = 0

        while diff == 0:
            feature = self.rng.integers(
                0, self.explanatory.shape[1]
            )
            values = self.explanatory[:, feature][
                node.sub_population
            ]
            fmin, fmax = np.min(values), np.max(values)
            diff = fmax - fmin

        x = self.rng.uniform()
        threshold = (1 - x) * fmin + x * fmax

        return feature, threshold
