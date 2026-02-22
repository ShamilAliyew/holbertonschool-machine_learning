#!/usr/bin/env python3
"""
Decision Tree and Random Forest implementation
with Node and Leaf classes.
"""
from pyexpat import features

import numpy as np


class Node:
    """A node class that generalizes everything including root and leaves."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Construct the Node object."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Find the maximum depth."""
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below()\
            if self.left_child else self.depth
        right = self.right_child.max_depth_below()\
            if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below, only leaves if specified."""
        if self.is_leaf:
            return 1

        if only_leaves:
            left = self.left_child.count_nodes_below(True)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(True)\
                if self.right_child else 0
            return left + right
        else:
            left = self.left_child.count_nodes_below(False)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(False)\
                if self.right_child else 0
            return 1 + left + right

    def __str__(self):
        """STR"""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            s = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            left_str = self.left_child.__str__()
            s += "\n" + self.left_child_add_prefix(left_str).rstrip("\n")

        if self.right_child:
            right_str = self.right_child.__str__()
            s += "\n" + self.right_child_add_prefix(right_str).rstrip("\n")

        return s

    def left_child_add_prefix(self, text):
        """Left Child"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Right Child"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Get Leaves"""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update Bounds"""
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
        """Update Indicator"""

        def is_large_enough(x):
            """Large enough"""
            return np.all(
                [x[:, j] >= bound for j, bound in self.lower.items()],
                axis=0
            )

        def is_small_enough(x):
            """Small Enough"""
            return np.all(
                [x[:, j] <= bound for j, bound in self.upper.items()],
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )


class Leaf(Node):
    """Terminal node which is a leaf."""

    def __init__(self, value, depth=None):
        """Construct the leaf object."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Retur the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the count of 1 leaf."""
        return 1

    def __str__(self):
        """STR"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Get Leaves"""
        return [self]

    def update_bounds_below(self):
        """Update Bounds"""
        pass


class Decision_Tree():
    """The whole Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Construct the decision tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the count of leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Get Leaves"""
        return self.root.get_leaves_below()

    def __str__(self):
        """STR"""
        return self.root.__str__() + "\n"

    def update_bounds(self):
        """Update Bounds"""
        self.root.update_bounds_below()
