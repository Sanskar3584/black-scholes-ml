"""
linear_models.py - OLS and Ridge regression baselines
======================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan sections 5.12 to 5.16, 11.7; ML_experiment_plan.md
Text    : Carol Alexander, Market Risk Analysis Vol. I, sections I.4.1 to I.4.4

Why closed form rather than a library solver
---------------------------------------------
Both estimators have exact solutions:

    OLS      beta = argmin ||y - X beta||^2            =>  (X'X) beta = X'y
    Ridge    beta = argmin ||y - X beta||^2 + a||beta||^2
                                                       =>  (X'X + aI) beta = X'y

so there is no convergence tolerance to argue about and the fit is reproducible
to machine precision on any machine. Where scikit-learn is installed, both fits
are cross-checked against it and the agreement is reported; where it is not,
nothing changes. This is the same two-route discipline used for the binomial
lattice in Week 4, and it has the same purpose: a single implementation that
agrees with nothing is not evidence.

The normal equations are solved by QR least squares rather than by forming and
inverting X'X, because squaring the design matrix squares its condition number.
With four features that is not currently dangerous, but the habit matters once
the extended feature set introduces Gamma and Vega, which are strongly related.

The intercept is not penalised
-------------------------------
Ridge shrinks coefficients towards zero. Shrinking the intercept would shrink
predictions towards zero rather than towards the mean, which is not what is
wanted and makes the result depend on where the origin happens to sit. The
standard fix, used here, is to centre the target, fit the slopes on centred
data, and recover the intercept afterwards.

Scaling and leakage
-------------------
The standardiser is fitted on TRAINING ROWS ONLY and then applied unchanged to
validation and test (master plan 5.12, 5.13). Fitting it on everything would
leak test information into training through the mean and standard deviation.

A useful consequence: OLS is invariant to any invertible linear rescaling of
the features, so OLS predictions must be IDENTICAL with and without the
standardiser, to floating-point precision. The self-test checks this, which
catches a scaler applied in the wrong direction - a mistake that would
otherwise show up only as mildly worse numbers.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Standardiser", "FitResult", "fit_ols", "fit_ridge",
    "select_ridge_alpha", "metrics", "ALPHA_GRID",
]

# Ridge penalty grid, fixed in ML_experiment_plan.md section 1.
ALPHA_GRID = np.logspace(-6, 3, 25)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

class Standardiser:
    """z = (x - mu) / s, with mu and s estimated on the training split only.

    A constant column would give s = 0; such columns are left untouched rather
    than producing infinities, and are reported so the caller knows.
    """

    def __init__(self):
        self.mu = None
        self.s = None
        self.constant = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        s = X.std(axis=0, ddof=0)
        self.constant = s <= 0
        s = np.where(self.constant, 1.0, s)        # leave constant columns alone
        self.s = s
        return self

    def transform(self, X):
        if self.mu is None:
            raise RuntimeError("Standardiser.fit must be called first")
        return (np.asarray(X, dtype=float) - self.mu) / self.s

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    """A fitted linear model: intercept, slopes, and how it was obtained."""
    name: str
    intercept: float
    coef: np.ndarray
    alpha: float = 0.0
    feature_names: tuple = ()
    extras: dict = field(default_factory=dict)

    def predict(self, Z):
        return self.intercept + np.asarray(Z, dtype=float) @ self.coef

    def as_row(self):
        d = {"model": self.name, "alpha": self.alpha,
             "intercept": self.intercept}
        for nm, c in zip(self.feature_names, self.coef):
            d["beta_" + nm] = float(c)
        return d


def fit_ols(Z, y, feature_names=()):
    """Ordinary least squares, solved by QR least squares on [1, Z]."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.hstack([np.ones((len(Z), 1)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return FitResult("OLS", float(beta[0]), beta[1:],
                     alpha=0.0, feature_names=tuple(feature_names))


def fit_ridge(Z, y, alpha, feature_names=()):
    """Ridge regression with the intercept excluded from the penalty.

    Centre y and the columns of Z, solve the penalised system for the slopes,
    then recover the intercept from the means. Solving the augmented system

        [ Zc      ]          [ yc ]
        [ sqrt(a)I]  beta =  [ 0  ]

    by least squares is algebraically identical to (Zc'Zc + aI)beta = Zc'yc but
    avoids forming Zc'Zc, so the condition number is not squared.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    zbar = Z.mean(axis=0)
    ybar = y.mean()
    Zc = Z - zbar
    yc = y - ybar

    p = Z.shape[1]
    A = np.vstack([Zc, np.sqrt(alpha) * np.eye(p)])
    b = np.concatenate([yc, np.zeros(p)])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)

    intercept = float(ybar - zbar @ beta)
    return FitResult("Ridge", intercept, beta,
                     alpha=float(alpha), feature_names=tuple(feature_names))


def select_ridge_alpha(Z_tr, y_tr, Z_val, y_val, grid=ALPHA_GRID,
                       feature_names=(), verbose=False):
    """Choose alpha on the VALIDATION split only.

    Returns (best_alpha, best_fit, table) where table is a list of
    (alpha, val_rmse) over the whole grid, so the selection can be plotted and
    defended rather than merely asserted.
    """
    table = []
    best = (np.inf, None, None)
    for a in grid:
        f = fit_ridge(Z_tr, y_tr, a, feature_names=feature_names)
        rmse = float(np.sqrt(np.mean((y_val - f.predict(Z_val)) ** 2)))
        table.append((float(a), rmse))
        if rmse < best[0]:
            best = (rmse, float(a), f)
    if verbose:
        print("    ridge alpha chosen on validation: %.3e "
              "(val RMSE %.6e)" % (best[1], best[0]))
    return best[1], best[2], table


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(y_true, y_pred, scale=1.0):
    """MAE, RMSE and R^2.

    scale multiplies MAE and RMSE so they can be reported in currency units as
    well as in the scaled target units; R^2 is scale-free.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_pred - y_true
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(e))) * scale,
        "RMSE": float(np.sqrt(np.mean(e ** 2))) * scale,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "max_abs_error": float(np.max(np.abs(e))) * scale,
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    if verbose:
        print("=" * 78)
        print("linear_models.py  SELF-TEST")
        print("=" * 78)

    checks = []

    def chk(name, value, tol):
        ok = bool(value <= tol)
        checks.append(ok)
        if verbose:
            print("  %-54s %.3e  tol %.1e  %s"
                  % (name, value, tol, "ok" if ok else "FAIL"))
        return ok

    rng = np.random.default_rng(20260825)
    n, p = 800, 4
    X = rng.normal(size=(n, p)) * np.array([1.0, 10.0, 0.01, 100.0])
    beta_true = np.array([2.0, -1.5, 0.5, 3.0])
    b0_true = 7.0

    # --- exact recovery on noiseless data ---
    y = b0_true + X @ beta_true
    f = fit_ols(X, y)
    chk("OLS recovers exact coefficients (noiseless)",
        float(np.max(np.abs(f.coef - beta_true))), 1e-10)
    chk("OLS recovers exact intercept (noiseless)",
        abs(f.intercept - b0_true), 1e-9)
    chk("OLS residual is zero (noiseless)",
        float(np.max(np.abs(f.predict(X) - y))), 1e-9)

    # --- the invariance check that catches a misapplied scaler ---
    y = b0_true + X @ beta_true + rng.normal(scale=0.5, size=n)
    sc = Standardiser().fit(X)
    f_raw = fit_ols(X, y)
    f_std = fit_ols(sc.transform(X), y)
    chk("OLS predictions invariant to standardisation",
        float(np.max(np.abs(f_raw.predict(X) - f_std.predict(sc.transform(X))))),
        1e-9)

    # --- ridge behaves as the theory says ---
    f0 = fit_ridge(sc.transform(X), y, 0.0)
    chk("Ridge at alpha = 0 equals OLS",
        float(np.max(np.abs(f0.coef - f_std.coef))), 1e-8)
    norms = [float(np.linalg.norm(fit_ridge(sc.transform(X), y, a).coef))
             for a in (0.0, 1.0, 1e2, 1e4, 1e6)]
    mono = max(norms[i + 1] - norms[i] for i in range(len(norms) - 1))
    chk("Ridge coefficient norm is non-increasing in alpha", max(0.0, mono), 1e-9)
    f_big = fit_ridge(sc.transform(X), y, 1e12)
    chk("Ridge shrinks slopes to ~0 as alpha -> inf",
        float(np.max(np.abs(f_big.coef))), 1e-6)
    chk("Ridge intercept -> mean(y) as alpha -> inf",
        abs(f_big.intercept - y.mean()), 1e-8)

    # --- ridge really is penalised least squares ---
    # compare against the textbook normal-equation form on centred data
    Z = sc.transform(X)
    a = 3.7
    Zc, yc = Z - Z.mean(axis=0), y - y.mean()
    beta_ref = np.linalg.solve(Zc.T @ Zc + a * np.eye(p), Zc.T @ yc)
    chk("Ridge matches (Z'Z + aI)^-1 Z'y on centred data",
        float(np.max(np.abs(fit_ridge(Z, y, a).coef - beta_ref))), 1e-10)

    # --- metrics ---
    m = metrics(y, y)
    chk("metrics: perfect fit gives MAE 0", m["MAE"], 1e-15)
    chk("metrics: perfect fit gives R2 1", abs(m["R2"] - 1.0), 1e-15)
    m2 = metrics(y, np.full(n, y.mean()))
    chk("metrics: mean prediction gives R2 0", abs(m2["R2"]), 1e-12)
    m3 = metrics(y, y + 2.0)
    chk("metrics: constant offset gives MAE = offset", abs(m3["MAE"] - 2.0), 1e-12)
    chk("metrics: scale argument multiplies RMSE",
        abs(metrics(y, y + 2.0, scale=100.0)["RMSE"] - 100 * m3["RMSE"]), 1e-10)

    # --- alpha selection uses validation, not training ---
    Ztr, ytr = Z[:600], y[:600]
    Zva, yva = Z[600:], y[600:]
    a_best, f_best, tab = select_ridge_alpha(Ztr, ytr, Zva, yva)
    val_rmses = [t[1] for t in tab]
    chk("selected alpha attains the minimum validation RMSE",
        abs(min(val_rmses) - min(t[1] for t in tab if t[0] == a_best)), 1e-15)
    chk("alpha grid spans the stated range",
        abs(np.log10(ALPHA_GRID[0]) + 6) + abs(np.log10(ALPHA_GRID[-1]) - 3), 1e-9)

    # --- cross-check against scikit-learn, if present ---
    try:
        from sklearn.linear_model import LinearRegression, Ridge
        sk_ols = LinearRegression().fit(Z, y)
        chk("OLS agrees with scikit-learn LinearRegression",
            float(np.max(np.abs(f_std.coef - sk_ols.coef_))), 1e-8)
        sk_r = Ridge(alpha=a, fit_intercept=True, solver="cholesky").fit(Z, y)
        chk("Ridge agrees with scikit-learn Ridge",
            float(np.max(np.abs(fit_ridge(Z, y, a).coef - sk_r.coef_))), 1e-8)
    except ImportError:
        if verbose:
            print("  %-54s %s"
                  % ("scikit-learn cross-check", "skipped (not installed)"))

    n_fail = sum(1 for ok in checks if not ok)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0 else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    ok = _self_test()
    raise SystemExit(0 if ok else 1)
