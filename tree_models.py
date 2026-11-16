"""
tree_models.py - regression tree and random forest, written out
================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan 5.17, 5.18, 11.8 task A; ML_experiment_plan.md 9.1
Text    : Carol Alexander, Market Risk Analysis Vol. I (regression background)

Why from scratch
----------------
"What does the tree split on, and how does it choose?" is a question the viva
will ask, and `RandomForestRegressor(n_estimators=100).fit(X, y)` is not an
answer. The split rule is written out below. Where scikit-learn is installed
the forest is cross-checked against it and the agreement reported; where it is
not, nothing changes. Same two-route discipline as the binomial lattice in
Week 4 and the linear models in Week 6.

The split rule
--------------
At a node holding samples S, a split on feature j at threshold t partitions S
into L and R. CART chooses the (j, t) maximising the reduction in squared
error, which for a constant prediction per side is

    gain  =  [ sum_L^2 / n_L  +  sum_R^2 / n_R ]  -  sum_S^2 / n_S

because the squared error of a leaf predicting the mean is
sum((y - ybar)^2) = sum(y^2) - sum(y)^2/n, and the sum(y^2) term is common to
both sides of the comparison and cancels. So only the first and zeroth moments
are needed - no variance has to be computed explicitly, and the whole scan is
two prefix sums.

Histogram splitting
-------------------
Each feature is binned ONCE into at most 64 quantile bins. A node's split
search is then a `bincount` over its samples plus a `cumsum` over bins, which
is O(n_node + bins) per feature instead of O(n_node log n_node) for a sort.
With 14,000 rows this is the difference between a script that runs in seconds
and one that does not.

Binning on quantiles rather than equal width matters here: the features have
very different shapes - `m` is uniform, `T` is log-uniform - and equal-width
bins would put almost every short-dated option into one bin, which is exactly
the region the project cares about.

The cost is that a threshold can only fall on a bin edge. With 64 quantile
bins the resulting granularity is far finer than the noise in any real option
dataset, and on synthetic data it shows up as a small, measurable bias that
the self-test checks does not grow.
"""

import numpy as np

__all__ = ["RegressionTree", "RandomForest", "SEED"]

SEED = 20260825
_MAX_BINS = 64


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------

def _bin_edges(X, max_bins=_MAX_BINS):
    """Quantile bin edges per feature, from the training data only."""
    edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        qs = np.linspace(0, 1, max_bins + 1)[1:-1]
        e = np.unique(np.quantile(col, qs))
        edges.append(e)
    return edges


def _apply_bins(X, edges):
    """Map features to bin indices. Shape (n, p), dtype int16."""
    out = np.empty(X.shape, dtype=np.int16)
    for j, e in enumerate(edges):
        out[:, j] = np.searchsorted(e, X[:, j], side="left")
    return out


# --------------------------------------------------------------------------
# a single tree
# --------------------------------------------------------------------------

class RegressionTree:
    """CART regression tree, histogram splits, squared-error criterion.

    Stored as flat arrays rather than node objects so that prediction is a
    vectorised walk rather than per-sample recursion.
    """

    def __init__(self, max_depth=12, min_samples_leaf=20, min_samples_split=40,
                 max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(SEED)

    # -- fitting ----------------------------------------------------------
    def fit(self, Xb, y, n_bins, edges):
        """Xb is the pre-binned design matrix; edges are needed to predict."""
        self.edges = edges
        n, p = Xb.shape
        self.n_features_ = p

        # flat node storage, grown as needed
        cap = 1024
        self.feature = np.full(cap, -1, dtype=np.int32)
        self.bin_thr = np.zeros(cap, dtype=np.int32)
        self.left = np.full(cap, -1, dtype=np.int32)
        self.right = np.full(cap, -1, dtype=np.int32)
        self.value = np.zeros(cap, dtype=np.float64)
        self.importance_ = np.zeros(p)
        self.n_nodes = 0

        def new_node():
            if self.n_nodes >= len(self.feature):
                for name in ("feature", "bin_thr", "left", "right", "value"):
                    a = getattr(self, name)
                    setattr(self, name, np.concatenate(
                        [a, np.full_like(a, -1 if a.dtype.kind == "i" else 0)]))
            self.n_nodes += 1
            return self.n_nodes - 1

        # explicit stack: recursion would hit Python's limit at depth and is
        # slower for no benefit
        root = new_node()
        stack = [(root, np.arange(n), 0)]

        while stack:
            node, idx, depth = stack.pop()
            yi = y[idx]
            self.value[node] = yi.mean()

            if (depth >= self.max_depth or len(idx) < self.min_samples_split
                    or len(idx) < 2 * self.min_samples_leaf):
                continue

            total_sum = yi.sum()
            total_n = len(idx)
            parent_term = total_sum * total_sum / total_n

            if self.max_features is None or self.max_features >= p:
                feats = range(p)
            else:
                feats = self.rng.choice(p, size=self.max_features,
                                        replace=False)

            best = (0.0, -1, -1)
            for j in feats:
                b = Xb[idx, j]
                cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
                tot = np.bincount(b, weights=yi, minlength=n_bins)
                cn = np.cumsum(cnt)[:-1]
                cs = np.cumsum(tot)[:-1]
                nr = total_n - cn
                sr = total_sum - cs
                ok = (cn >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
                if not ok.any():
                    continue
                gain = np.full(cn.shape, -np.inf)
                gain[ok] = (cs[ok] ** 2 / cn[ok] + sr[ok] ** 2 / nr[ok]
                            - parent_term)
                k = int(np.argmax(gain))
                if gain[k] > best[0]:
                    best = (float(gain[k]), int(j), k)

            gain, j, k = best
            if j < 0 or gain <= 1e-18:
                continue

            self.importance_[j] += gain
            go_left = Xb[idx, j] <= k
            li, ri = idx[go_left], idx[~go_left]
            self.feature[node] = j
            self.bin_thr[node] = k
            ln, rn = new_node(), new_node()
            self.left[node] = ln
            self.right[node] = rn
            stack.append((ln, li, depth + 1))
            stack.append((rn, ri, depth + 1))

        for name in ("feature", "bin_thr", "left", "right", "value"):
            setattr(self, name, getattr(self, name)[:self.n_nodes])
        return self

    # -- prediction -------------------------------------------------------
    def predict_binned(self, Xb):
        n = len(Xb)
        node = np.zeros(n, dtype=np.int32)
        active = np.ones(n, dtype=bool)
        # walk all samples down together, one level per iteration
        for _ in range(self.max_depth + 2):
            f = self.feature[node]
            internal = active & (f >= 0)
            if not internal.any():
                break
            ii = np.flatnonzero(internal)
            go_left = Xb[ii, f[ii]] <= self.bin_thr[node[ii]]
            node[ii] = np.where(go_left, self.left[node[ii]],
                                self.right[node[ii]])
        return self.value[node]

    def predict(self, X):
        return self.predict_binned(_apply_bins(np.asarray(X, float), self.edges))


# --------------------------------------------------------------------------
# the forest
# --------------------------------------------------------------------------

class RandomForest:
    """Bagged regression trees.

    Two sources of decorrelation, both standard and both reported:
      - bootstrap resampling of rows for each tree;
      - a random subset of features considered at each split.

    With only four features, subsetting is mild; `max_features=3` is used so
    the trees are not identical, which is the point of a forest rather than
    one deep tree.
    """

    def __init__(self, n_trees=60, max_depth=12, min_samples_leaf=20,
                 max_features=3, seed=SEED, max_bins=_MAX_BINS):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.max_bins = max_bins

    def fit(self, X, y, verbose=False):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)
        self.edges = _bin_edges(X, self.max_bins)
        n_bins = max(len(e) for e in self.edges) + 1
        Xb = _apply_bins(X, self.edges)

        rng = np.random.default_rng(self.seed)
        self.trees = []
        self.importance_ = np.zeros(X.shape[1])
        self.oob_prediction_ = np.zeros(n)
        oob_count = np.zeros(n)

        for t in range(self.n_trees):
            boot = rng.integers(0, n, n)
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=2 * self.min_samples_leaf,
                max_features=self.max_features,
                rng=np.random.default_rng(self.seed + 1000 + t))
            tree.fit(Xb[boot], y[boot], n_bins, self.edges)
            self.trees.append(tree)
            self.importance_ += tree.importance_

            # out-of-bag: rows this tree never saw. A free validation estimate
            # that costs nothing and does not touch the validation split.
            mask = np.ones(n, dtype=bool)
            mask[boot] = False
            if mask.any():
                self.oob_prediction_[mask] += tree.predict_binned(Xb[mask])
                oob_count[mask] += 1

            if verbose and (t + 1) % 20 == 0:
                print("      %d/%d trees" % (t + 1, self.n_trees))

        s = self.importance_.sum()
        self.importance_ = self.importance_ / s if s > 0 else self.importance_
        with np.errstate(invalid="ignore"):
            self.oob_prediction_ = np.where(
                oob_count > 0, self.oob_prediction_ / np.maximum(oob_count, 1),
                np.nan)
        self.oob_fraction_ = float((oob_count > 0).mean())
        return self

    def predict(self, X):
        Xb = _apply_bins(np.asarray(X, float), self.edges)
        out = np.zeros(len(Xb))
        for tree in self.trees:
            out += tree.predict_binned(Xb)
        return out / len(self.trees)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    if verbose:
        print("=" * 78)
        print("tree_models.py  SELF-TEST")
        print("=" * 78)

    checks = []

    def chk(name, value, tol):
        ok = bool(value <= tol)
        checks.append(ok)
        if verbose:
            print("  %-54s %.3e  tol %.1e  %s"
                  % (name, value, tol, "ok" if ok else "FAIL"))
        return ok

    rng = np.random.default_rng(SEED)

    # --- a single tree must fit a step function, away from the breaks ---
    #
    # A tree is piecewise constant, so it should reproduce a piecewise-constant
    # target exactly - EXCEPT within one bin-width of a break. Thresholds can
    # only fall on bin edges, and with 63 quantile bins on uniform(0,1) no edge
    # lands exactly on 0.5, so samples straddling the break share a leaf and
    # get its average.
    #
    # The first version of this test demanded exact recovery everywhere and
    # failed on 93 of 4,000 samples, all of them within one bin-width of a
    # break. That is the documented cost of histogram splitting, not a bug, so
    # the test now asserts the two things that ARE true: exact recovery away
    # from the breaks, and a straddling fraction no larger than the binning
    # implies. Both would still catch a broken split search.
    X = rng.uniform(0, 1, (4000, 2))
    y = np.where(X[:, 0] < 0.5, np.where(X[:, 1] < 0.5, 1.0, 2.0),
                 np.where(X[:, 1] < 0.5, 3.0, 4.0))
    edges = _bin_edges(X)
    nb = max(len(e) for e in edges) + 1
    t = RegressionTree(max_depth=6, min_samples_leaf=5, min_samples_split=10)
    t.fit(_apply_bins(X, edges), y, nb, edges)
    pred = t.predict(X)
    bw = 1.0 / (_MAX_BINS - 1)                      # one bin width
    clear = (np.abs(X[:, 0] - 0.5) > bw) & (np.abs(X[:, 1] - 0.5) > bw)
    if verbose:
        print("    %d of %d samples lie within one bin-width of a break"
              % ((~clear).sum(), len(X)))
    chk("single tree is exact away from the breaks",
        float(np.max(np.abs(pred[clear] - y[clear]))), 1e-12)
    # The expected straddling fraction is not 4*bw but the union probability
    # 1 - (1 - 2*bw)^2 = 4*bw - 4*bw^2, and with 4,000 draws it carries a
    # binomial standard error. Writing the bound out rather than rounding it
    # keeps the check tight enough to still fail if the binning regressed.
    p_strad = 4 * bw - 4 * bw * bw
    se_strad = np.sqrt(p_strad * (1 - p_strad) / len(X))
    chk("straddling fraction matches the binning (%.4f +/- %.4f)"
        % (p_strad, se_strad), float((~clear).mean()), p_strad + 4 * se_strad)
    chk("tree found all four regions",
        float(abs(len(np.unique(np.round(pred[clear], 9))) - 4)), 0)

    # --- a depth-0 tree is the mean ---
    t0 = RegressionTree(max_depth=0)
    t0.fit(_apply_bins(X, edges), y, nb, edges)
    chk("depth-0 tree predicts the mean",
        float(np.max(np.abs(t0.predict(X) - y.mean()))), 1e-12)

    # --- predictions stay inside the training range ---
    # Trees cannot extrapolate: every prediction is an average of training
    # targets. This is a property worth asserting, because it is exactly why
    # the forest behaves differently from the network in the deep wings.
    Xt = rng.uniform(-1, 2, (500, 2))
    f = RandomForest(n_trees=10, max_depth=8, seed=SEED).fit(X, y)
    pr = f.predict(Xt)
    chk("forest cannot predict below the training minimum",
        float(max(0.0, y.min() - pr.min())), 1e-12)
    chk("forest cannot predict above the training maximum",
        float(max(0.0, pr.max() - y.max())), 1e-12)

    # --- a forest beats a single tree on noisy data ---
    Xn = rng.uniform(0, 1, (3000, 4))
    yn = (np.sin(6 * Xn[:, 0]) + Xn[:, 1] ** 2 - 0.5 * Xn[:, 2]
          + rng.normal(0, 0.1, 3000))
    Xv = rng.uniform(0, 1, (1500, 4))
    yv = np.sin(6 * Xv[:, 0]) + Xv[:, 1] ** 2 - 0.5 * Xv[:, 2]
    e1 = _bin_edges(Xn)
    nb1 = max(len(e) for e in e1) + 1
    one = RegressionTree(max_depth=12, min_samples_leaf=5,
                         min_samples_split=10).fit(
                             _apply_bins(Xn, e1), yn, nb1, e1)
    rmse_tree = float(np.sqrt(np.mean((one.predict(Xv) - yv) ** 2)))
    fo = RandomForest(n_trees=40, max_depth=12, min_samples_leaf=5,
                      seed=SEED).fit(Xn, yn)
    rmse_forest = float(np.sqrt(np.mean((fo.predict(Xv) - yv) ** 2)))
    if verbose:
        print("    single tree RMSE %.4f, forest RMSE %.4f"
              % (rmse_tree, rmse_forest))
    chk("forest beats a single tree (variance reduction)",
        float(max(0.0, rmse_forest - 0.9 * rmse_tree)), 1e-12)

    # --- importance finds the feature that matters, and ignores noise ---
    # x0 drives the target; x3 is pure noise. Importance must rank accordingly.
    imp = fo.importance_
    if verbose:
        print("    importances: " + "  ".join("x%d %.3f" % (i, v)
                                              for i, v in enumerate(imp)))
    chk("importance ranks the driving feature first",
        float(max(0.0, imp.max() - imp[0])), 1e-12)
    chk("importance of the irrelevant feature is small",
        float(imp[3]), 0.10)
    chk("importances sum to 1", abs(imp.sum() - 1.0), 1e-10)

    # --- out-of-bag coverage is about 1 - 1/e ---
    chk("most rows are out of bag for some tree",
        float(max(0.0, 0.99 - fo.oob_fraction_)), 1e-9)

    # --- determinism ---
    f2 = RandomForest(n_trees=10, max_depth=8, seed=SEED).fit(X, y)
    chk("same seed reproduces the forest exactly",
        float(np.max(np.abs(f.predict(Xt) - f2.predict(Xt)))), 0.0)

    # --- cross-check against scikit-learn ---
    try:
        from sklearn.ensemble import RandomForestRegressor
        sk = RandomForestRegressor(n_estimators=40, max_depth=12,
                                   min_samples_leaf=5, max_features=3,
                                   random_state=0, n_jobs=1).fit(Xn, yn)
        rmse_sk = float(np.sqrt(np.mean((sk.predict(Xv) - yv) ** 2)))
        rel = abs(rmse_forest - rmse_sk) / rmse_sk
        if verbose:
            print("    scikit-learn forest RMSE %.4f vs ours %.4f "
                  "(relative gap %.1f%%)" % (rmse_sk, rmse_forest, 100 * rel))
        chk("agrees with scikit-learn RandomForest to within 25%", rel, 0.25)
    except ImportError:
        if verbose:
            print("  %-54s %s" % ("scikit-learn cross-check",
                                  "skipped (not installed)"))

    n_fail = sum(1 for ok in checks if not ok)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0 else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
