#!/usr/bin/env python3
"""
documented
"""


class Node:
    """Decision tree internal node."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False
    ):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        """Add the left-branch prefixes to a multiline string."""
        lines = text.split("\n")
        out = "+--" + lines[0]
        for line in lines[1:]:
            out += "\n| " + line
        return out

    def right_child_add_prefix(self, text):
        """Add the right-branch prefixes to a multiline string."""
        lines = text.split("\n")
        out = "+--" + lines[0]
        for line in lines[1:]:
            out += "\n" + line
        return out

    def __str__(self):
        """Return a string representation of the subtree."""
        if self.is_root:
            head = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            head = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is None or self.right_child is None:
            return head

        left_txt = self.left_child_add_prefix(str(self.left_child))
        right_txt = self.right_child_add_prefix(str(self.right_child))
        return head + "\n" + left_txt + "\n" + right_txt


class Leaf:
    """Decision tree leaf."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Return a string representation of the leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Decision tree container."""

    def __init__(self, root=None):
        """Initialize Decision_Tree."""
        self.root = root

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()
