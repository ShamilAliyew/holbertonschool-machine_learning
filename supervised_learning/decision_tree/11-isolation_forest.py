#!/usr/bin/env python3
"""Isolation Random Forest Algorithm."""
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree  # noqa: E402
import numpy as np


class Isolation_Random_Forest():
    """Isolation Random Forest Class."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize IRF object."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Make a prediction."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Fit the model."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """Return n_suspects rows in explanatory with smallest mean depth."""
        depths = self.predict(explanatory)

        # Get indices of samples sorted by depth (smallest first)
        suspect_indices = np.argsort(depths)[:n_suspects]

        # Return both the actual rows from explanatory data AND their depths
        return explanatory[suspect_indices], depths[suspect_indices]
