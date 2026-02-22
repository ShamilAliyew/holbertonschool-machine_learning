#!/usr/bin/env python3
"""
Decision Tree and Random Forest implementation
with Node and Leaf classes.
"""

import numpy as np


class Node:
    """A decision tree node with optional children and split feature."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node with optional children and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth below this node, including leaves."""
        if self.is_leaf:
            return self.depth
        left_depth = self.depth
        right_depth = self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Recursive node counter"""
        if self.is_leaf:
            return 1 if only_leaves else 1

        left_count = self.left_child.count_nodes_below(only_leaves) \
            if self.left_child else 0
        right_count = self.right_child.count_nodes_below(only_leaves) \
            if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count


class Leaf(Node):
    """A leaf node in a decision tree containing a value."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf with a value and optional depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Node Below"""
        return 1 if only_leaves else 1


class Decision_Tree:
    """Decision tree object containing the root node."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree with optional parameters."""
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
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes from the root"""
        if not self.root:
            return 0
        return self.root.count_nodes_below(only_leaves)
