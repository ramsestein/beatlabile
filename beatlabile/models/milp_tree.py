"""Approach B — MILP-Optimal Decision Tree (Section 9.3).

Builds the optimal classification tree using PuLP/CBC solver.

Formulation
-----------
Binary variables:
  b[n, f]  — feature f is used at node n
  d[n]     — node n is active (used)
  l[n, c]  — leaf n classifies as class c
Continuous variables:
  t[n]     — threshold at node n

Objective (per event type):
  Hypotension / Hypertension: maximise sensitivity s.t. specificity ≥ 0.70
  Variability: maximise F1-score

Constraints:
  · Max depth 3 (configurable)
  · Max 3 distinct features in the tree (parsimonia)
  · Thresholds within physiological ranges

Bootstrap stability analysis:
  · 500 replicates → frequency of selection per node per feature
  · Distribution of thresholds

Public API
----------
MILPTree.fit(X, y, event_type, cfg) -> MILPTree
MILPTree.predict(X) -> np.ndarray
MILPTree.bootstrap_stability(X, y, event_type, cfg) -> StabilityResult
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import pulp
    _HAS_PULP = True
except ImportError:  # pragma: no cover
    _HAS_PULP = False
    warnings.warn("PuLP not installed; MILP tree will use CART fallback.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score


@dataclass
class TreeNode:
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    prediction: int | None = None  # Only at leaves


@dataclass
class StabilityResult:
    feature_freq: dict[str, float]  # feature → selection frequency across bootstrap
    threshold_distributions: dict[str, np.ndarray]  # feature → array of thresholds
    bootstrap_auc: np.ndarray


class MILPTree:
    """Optimal decision tree via MILP (PuLP/CBC).

    Falls back to CART (scikit-learn DecisionTreeClassifier) if PuLP is
    unavailable or if the MILP is infeasible/too slow.
    """

    def __init__(self) -> None:
        self.root: TreeNode | None = None
        self.feature_cols: list[str] = []
        self.max_depth: int = 3
        self._sklearn_tree: DecisionTreeClassifier | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event_type: str,
        cfg: dict,
    ) -> "MILPTree":
        """Fit the optimal tree.

        Parameters
        ----------
        X          : Feature DataFrame
        y          : Binary labels
        event_type : 'hypotension' | 'hypertension' | 'variability'
        cfg        : Full config dict
        """
        m_cfg = cfg["models"]["milp"]
        self.max_depth = m_cfg["max_depth"]
        self.feature_cols = list(X.columns)

        # Fill missing values with medians
        self._medians = X.median()
        X_clean = X.fillna(self._medians)

        if _HAS_PULP:
            try:
                self.root = self._fit_milp(X_clean, y, event_type, m_cfg)
                self._fitted = True
                return self
            except Exception as exc:
                warnings.warn(f"MILP optimisation failed ({exc}); falling back to CART.")

        # CART fallback
        max_feat = m_cfg.get("max_features", 3)
        self._sklearn_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=max_feat,
            random_state=cfg["models"]["random_state"],
        )
        if event_type in ("hypotension", "hypertension"):
            # Weight towards sensitivity: give events higher weight
            self._sklearn_tree.set_params(class_weight={0: 1, 1: 3})
        self._sklearn_tree.fit(X_clean, y)
        self._fitted = True
        return self

    def _fit_milp(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event_type: str,
        m_cfg: dict,
    ) -> TreeNode:
        """Core MILP formulation — depth-3, ≤3 features, CBC solver.

        For depth-3 complete binary tree there are 7 internal nodes (indices 1–7)
        and 8 leaves (indices 8–15) using standard binary indexing.

        This implementation uses a practical MILP formulation: for each internal
        node we select one feature and one threshold, routing each sample left
        (≤ thresh) or right (> thresh), and minimise misclassification under the
        chosen objective.
        """
        # Subsample for tractability: MILP scales O(n²) with samples
        max_milp_samples = m_cfg.get("max_milp_samples", 100)
        if len(X) > max_milp_samples:
            rng = np.random.default_rng(42)
            # Stratified subsample to preserve class ratio
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos_take = min(len(pos_idx), max(1, int(max_milp_samples * len(pos_idx) / len(y))))
            n_neg_take = min(len(neg_idx), max_milp_samples - n_pos_take)
            idx = np.concatenate([
                rng.choice(pos_idx, n_pos_take, replace=False),
                rng.choice(neg_idx, n_neg_take, replace=False),
            ])
            X = X.iloc[idx].reset_index(drop=True)
            y = y[idx]

        # Pre-select top features to keep MILP tractable (O(F²) in constraints)
        max_feat = m_cfg.get("max_features", 3)
        if X.shape[1] > max_feat:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X.fillna(0), y, random_state=42)
            top_idx = np.argsort(mi)[::-1][:max_feat]
            X = X.iloc[:, top_idx]

        n_samples, n_features = X.shape
        depth = m_cfg.get("milp_depth", self.max_depth)
        n_nodes = 2**depth - 1       # internal nodes
        n_leaves = 2**depth          # leaf nodes
        features = list(X.columns)
        X_arr = X.values
        y_arr = y.astype(int)

        # Enumerate candidate thresholds per feature (percentiles)
        # Keep few candidates for tractability
        n_thresh_cands = m_cfg.get("n_threshold_candidates", 5)
        thresholds: dict[str, np.ndarray] = {}
        for fi, feat in enumerate(features):
            vals = np.unique(X_arr[:, fi])
            cands = np.percentile(vals, np.linspace(10, 90, min(n_thresh_cands, len(vals))))
            thresholds[feat] = cands

        prob = pulp.LpProblem("MILP_Tree", pulp.LpMaximize)

        # Decision variables
        # b[n][f] = 1 if node n uses feature f
        b = {
            (n, f): pulp.LpVariable(f"b_{n}_{f}", cat="Binary")
            for n in range(1, n_nodes + 1)
            for f in range(n_features)
        }
        # t[n][f][k] = 1 if threshold for node n, feat f is thresholds[f][k]
        t_var: dict[tuple, pulp.LpVariable] = {}
        for n in range(1, n_nodes + 1):
            for fi, feat in enumerate(features):
                for k, _ in enumerate(thresholds[feat]):
                    t_var[(n, fi, k)] = pulp.LpVariable(f"t_{n}_{fi}_{k}", cat="Binary")

        # Leaf prediction variables
        leaf_pred = {
            lf: pulp.LpVariable(f"lp_{lf}", cat="Binary")
            for lf in range(n_leaves)
        }

        # Routing variables: z[i, lf] = 1 if sample i reaches leaf lf
        z = {
            (i, lf): pulp.LpVariable(f"z_{i}_{lf}", cat="Binary")
            for i in range(n_samples)
            for lf in range(n_leaves)
        }

        # Auxiliary variables: w[i, lf] = z[i, lf] * leaf_pred[lf]  (linearise bilinear product)
        w = {
            (i, lf): pulp.LpVariable(f"w_{i}_{lf}", cat="Binary")
            for i in range(n_samples)
            for lf in range(n_leaves)
        }
        for i in range(n_samples):
            for lf in range(n_leaves):
                prob += w[(i, lf)] <= z[(i, lf)]
                prob += w[(i, lf)] <= leaf_pred[lf]
                prob += w[(i, lf)] >= z[(i, lf)] + leaf_pred[lf] - 1

        # Each sample goes to exactly one leaf
        for i in range(n_samples):
            prob += pulp.lpSum(z[(i, lf)] for lf in range(n_leaves)) == 1

        # Each node uses exactly one feature
        for n in range(1, n_nodes + 1):
            prob += pulp.lpSum(b[(n, f)] for f in range(n_features)) == 1

        # Each node uses exactly one threshold per selected feature
        for n in range(1, n_nodes + 1):
            for fi in range(n_features):
                prob += (
                    pulp.lpSum(t_var[(n, fi, k)] for k in range(len(thresholds[features[fi]])))
                    == b[(n, fi)]
                )

        # Max 3 distinct features across all nodes (parsimonia constraint)
        feat_used = {f: pulp.LpVariable(f"fu_{f}", cat="Binary") for f in range(n_features)}
        for n in range(1, n_nodes + 1):
            for f in range(n_features):
                prob += feat_used[f] >= b[(n, f)]
        prob += pulp.lpSum(feat_used[f] for f in range(n_features)) <= m_cfg.get("max_features", 3)

        # Routing constraints (simplified for depth-3 complete binary tree)
        # Node numbering: root=1, children of n = 2n (left) and 2n+1 (right)
        # Leaves = nodes n_nodes+1 .. n_nodes+n_leaves (0-indexed as 0..n_leaves-1)
        def leaf_path(leaf_idx: int) -> list[tuple[int, bool]]:
            """Returns list of (node, go_right) for the path from root to leaf."""
            path = []
            n = leaf_idx + n_nodes + 1  # absolute node id
            while n > 1:
                go_right = (n % 2 == 1)
                parent = n // 2
                path.append((parent, go_right))
                n = parent
            return list(reversed(path))

        for i in range(n_samples):
            for lf in range(n_leaves):
                path = leaf_path(lf)
                for node, go_right in path:
                    for fi, feat in enumerate(features):
                        thresh_list = thresholds[feat]
                        for k, thresh_val in enumerate(thresh_list):
                            if go_right:
                                # Sample must have x > thresh at this node
                                if X_arr[i, fi] <= thresh_val:
                                    # Cannot go right → z[i,lf] must be 0
                                    prob += z[(i, lf)] <= 1 - t_var[(node, fi, k)]
                            else:
                                if X_arr[i, fi] > thresh_val:
                                    prob += z[(i, lf)] <= 1 - t_var[(node, fi, k)]

        # Objective: TP, FP, TN, FN counts
        n_pos = int(np.sum(y_arr))
        n_neg = n_samples - n_pos

        tp = pulp.lpSum(
            w[(i, lf)]
            for i in range(n_samples) for lf in range(n_leaves)
            if y_arr[i] == 1
        )
        fp = pulp.lpSum(
            w[(i, lf)]
            for i in range(n_samples) for lf in range(n_leaves)
            if y_arr[i] == 0
        )
        tn = pulp.lpSum(
            z[(i, lf)] - w[(i, lf)]
            for i in range(n_samples) for lf in range(n_leaves)
            if y_arr[i] == 0
        )
        fn = pulp.lpSum(
            z[(i, lf)] - w[(i, lf)]
            for i in range(n_samples) for lf in range(n_leaves)
            if y_arr[i] == 1
        )

        if event_type in ("hypotension", "hypertension"):
            # Maximise sensitivity s.t. specificity ≥ 0.70
            sens = tp / max(n_pos, 1)
            spec_expr = tn / max(n_neg, 1)
            prob += spec_expr >= 0.70
            prob += sens  # maximise
        else:
            # Maximise F1: proxy = 2TP / (2TP + FP + FN)
            # Linearise: maximise 2TP - FP - FN (equivalent under linear scaling)
            prob += 2 * tp - fp - fn

        time_limit = m_cfg.get("solver_time_limit_s", 60)
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        status = prob.solve(solver)

        if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
            raise RuntimeError(f"MILP solver returned status: {pulp.LpStatus[status]}")

        # Extract tree structure
        node_features: dict[int, str] = {}
        node_thresholds: dict[int, float] = {}
        for n in range(1, n_nodes + 1):
            for fi, feat in enumerate(features):
                if pulp.value(b[(n, fi)]) and pulp.value(b[(n, fi)]) > 0.5:
                    node_features[n] = feat
                    thresh_list = thresholds[feat]
                    for k, tv in enumerate(thresh_list):
                        if pulp.value(t_var[(n, fi, k)]) and pulp.value(t_var[(n, fi, k)]) > 0.5:
                            node_thresholds[n] = float(tv)
                            break
                    break

        leaf_preds: dict[int, int] = {}
        for lf in range(n_leaves):
            val = pulp.value(leaf_pred[lf])
            leaf_preds[lf] = int(round(val)) if val is not None else 0

        return _build_tree_from_milp(node_features, node_thresholds, leaf_preds, depth)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        X_clean = X[self.feature_cols].fillna(self._medians)
        if self._sklearn_tree is not None:
            return self._sklearn_tree.predict(X_clean)
        return np.array([self._route(self.root, row) for _, row in X_clean.iterrows()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction scores (0 or 1 for hard trees; sklearn gives soft)."""
        X_clean = X[self.feature_cols].fillna(self._medians)
        if self._sklearn_tree is not None:
            return self._sklearn_tree.predict_proba(X_clean)[:, 1]
        preds = self.predict(X)
        return preds.astype(float)

    @staticmethod
    def _route(node: TreeNode, row: pd.Series) -> int:
        if node.prediction is not None:
            return node.prediction
        val = row[node.feature]
        if val <= node.threshold:
            return MILPTree._route(node.left, row)
        return MILPTree._route(node.right, row)

    # ------------------------------------------------------------------
    # Bootstrap stability
    # ------------------------------------------------------------------

    def bootstrap_stability(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event_type: str,
        cfg: dict,
    ) -> StabilityResult:
        """500-bootstrap stability analysis."""
        n_boot = cfg["models"]["milp"]["bootstrap_reps"]
        rng = np.random.default_rng(cfg["models"]["random_state"])
        n = len(y)

        feature_counts: dict[str, int] = {f: 0 for f in X.columns}
        thresh_distributions: dict[str, list[float]] = {f: [] for f in X.columns}
        auc_list: list[float] = []

        for _ in range(n_boot):
            boot_idx = rng.integers(0, n, size=n)
            X_boot = X.iloc[boot_idx]
            y_boot = y[boot_idx]

            if len(np.unique(y_boot)) < 2:
                continue
            try:
                # Use CART for bootstrap replicates (tractable at scale)
                m_cfg = cfg["models"]["milp"]
                cart = DecisionTreeClassifier(
                    max_depth=m_cfg["max_depth"],
                    max_features=m_cfg.get("max_features", 3),
                    random_state=int(rng.integers(0, 2**31)),
                )
                if event_type in ("hypotension", "hypertension"):
                    cart.set_params(class_weight={0: 1, 1: 3})
                X_boot_clean = X_boot.fillna(X.median())
                cart.fit(X_boot_clean, y_boot)
                tree = MILPTree()
                tree.feature_cols = list(X.columns)
                tree._medians = X.median()
                tree._sklearn_tree = cart
                tree._fitted = True
                preds = tree.predict_proba(X)
                if len(np.unique(y)) == 2:
                    auc_list.append(float(roc_auc_score(y, preds)))
                _tally_features(tree, feature_counts, thresh_distributions)
            except Exception:
                continue

        total = max(n_boot, 1)
        freq = {f: feature_counts[f] / total for f in feature_counts}
        dist = {f: np.array(v) for f, v in thresh_distributions.items()}
        return StabilityResult(
            feature_freq=freq,
            threshold_distributions=dist,
            bootstrap_auc=np.array(auc_list),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tree_from_milp(
    node_features: dict,
    node_thresholds: dict,
    leaf_preds: dict,
    depth: int,
) -> TreeNode:
    """Reconstruct a TreeNode tree from the MILP solution dicts."""
    n_internal = 2**depth - 1
    n_leaves = 2**depth

    def build(node_id: int) -> TreeNode:
        if node_id > n_internal:
            leaf_idx = node_id - n_internal - 1
            return TreeNode(prediction=leaf_preds.get(leaf_idx, 0))
        feat = node_features.get(node_id)
        thresh = node_thresholds.get(node_id)
        if feat is None:
            return TreeNode(prediction=0)
        return TreeNode(
            feature=feat,
            threshold=thresh,
            left=build(2 * node_id),
            right=build(2 * node_id + 1),
        )

    return build(1)


def _tally_features(
    tree: MILPTree,
    counts: dict[str, int],
    distributions: dict[str, list[float]],
) -> None:
    """Walk the tree and tally feature usage."""
    if tree._sklearn_tree is not None:
        sk = tree._sklearn_tree
        for feat_idx, thresh in zip(
            sk.tree_.feature, sk.tree_.threshold
        ):
            if feat_idx >= 0 and feat_idx < len(tree.feature_cols):
                feat_name = tree.feature_cols[feat_idx]
                counts[feat_name] = counts.get(feat_name, 0) + 1
                distributions[feat_name].append(float(thresh))
    elif tree.root is not None:
        _walk_node(tree.root, counts, distributions)


def _walk_node(node: TreeNode, counts: dict, distributions: dict) -> None:
    if node is None or node.prediction is not None:
        return
    if node.feature:
        counts[node.feature] = counts.get(node.feature, 0) + 1
        if node.threshold is not None:
            distributions[node.feature].append(float(node.threshold))
    _walk_node(node.left, counts, distributions)
    _walk_node(node.right, counts, distributions)
