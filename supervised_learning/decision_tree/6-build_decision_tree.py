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
        """Documendted"""
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
        """Documendted"""
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

    def __str__(self):
        """Documendted"""
        if self.is_root:
            s = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )
        else:
            s = (
                f"node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        if self.left_child:
            left_str = self.left_child.__str__()
            s += "\n" + self.left_child_add_prefix(
                left_str
            ).rstrip("\n")

        if self.right_child:
            right_str = self.right_child.__str__()
            s += "\n" + self.right_child_add_prefix(
                right_str
            ).rstrip("\n")

        return s

    def left_child_add_prefix(self, text):
        """Documendted"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Documendted"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def get_leaves_below(self):
        """Documendted"""
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Documendted"""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            feature = self.feature
            threshold = self.threshold

            if child is self.left_child:
                child.lower[feature] = threshold
            else:
                child.upper[feature] = threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Documendted"""
        def is_large_enough(x):
            return np.all(
                [x[:, j] >= bound for j, bound in self.lower.items()],
                axis=0,
            )

        def is_small_enough(x):
            """Documendted"""
            return np.all(
                [x[:, j] <= bound for j, bound in self.upper.items()],
                axis=0,
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):
        """Documendted"""
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

    def max_depth_below(self):
        """Documendted"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Documendted"""
        return 1

    def __str__(self):
        """Documendted"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Documendted"""
        return [self]

    def update_bounds_below(self):
        """Documendted"""
        pass

    def pred(self, x):
        """Documendted"""
        return self.value


class Decision_Tree:
    """Documendted"""

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
        """Documendted"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Documendted"""
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        """Documendted"""
        return self.root.get_leaves_below()

    def __str__(self):
        """Documendted"""
        return self.root.__str__() + "\n"

    def update_bounds(self):
        """Documendted"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Documendted"""
        return self.root.pred(x)

    def update_predict(self):
        """Documendted"""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: self._predict_from_leaves(A, leaves)

    def _predict_from_leaves(self, A, leaves):
        """Documendted"""
        res = np.zeros(A.shape[0])
        for leaf in leaves:
            mask = leaf.indicator(A)
            res[mask] = leaf.value
        return res

    def fit(self, explanatory, target, verbose=0):
        """Documendted"""
        self.explanatory = explanatory
        self.target = target

        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion

        self.root.sub_population = np.ones_like(
            target, dtype=bool
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(True)}
- Accuracy on training data :
  {self.accuracy(self.explanatory, self.target)}
"""
            )

    def fit_node(self, node):
        """Documendted"""
        sub_target = self.target[node.sub_population]

        if (
            node.sub_population.sum() <= self.min_pop
            or node.depth >= self.max_depth
            or np.all(sub_target == sub_target[0])
        ):
            node.value = np.bincount(sub_target).argmax()
            leaf = Leaf(node.value)
            leaf.depth = node.depth
            leaf.sub_population = node.sub_population
            return leaf

        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            node.sub_population
            & (
                self.explanatory[:, node.feature]
                > node.threshold
            )
        )

        right_population = (
            node.sub_population
            & ~(
                self.explanatory[:, node.feature]
                > node.threshold
            )
        )

        if (
            left_population.sum() <= self.min_pop
            or node.depth + 1 >= self.max_depth
            or np.all(
                self.target[left_population]
                == self.target[left_population][0]
            )
        ):
            node.left_child = self.get_leaf_child(
                node, left_population
            )
        else:
            node.left_child = self.get_node_child(
                node, left_population
            )
            self.fit_node(node.left_child)

        if (
            right_population.sum() <= self.min_pop
            or node.depth + 1 >= self.max_depth
            or np.all(
                self.target[right_population]
                == self.target[right_population][0]
            )
        ):
            node.right_child = self.get_leaf_child(
                node, right_population
            )
        else:
            node.right_child = self.get_node_child(
                node, right_population
            )
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Documendted"""
        value = np.bincount(
            self.target[sub_population]
        ).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Documendted"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        preds = self.predict(test_explanatory)
        return np.sum(preds == test_target) / test_target.size

    def np_extrema(self, arr):
        """Documendted"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Documendted"""
        diff = 0

        while diff == 0:
            feature = self.rng.integers(
                0, self.explanatory.shape[1]
            )

            feature_values = self.explanatory[
                :, feature
            ][node.sub_population]

            feature_min, feature_max = self.np_extrema(
                feature_values
            )

            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (
            (1 - x) * feature_min + x * feature_max
        )

        return feature, threshold
