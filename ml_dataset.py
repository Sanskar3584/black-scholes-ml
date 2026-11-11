"""
ml_dataset.py - synthetic option dataset for the machine-learning stage
=======================================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan sections 5.2, 5.10 to 5.13, 11.6; ML_experiment_plan.md
Text    : Carol Alexander, Market Risk Analysis Vol. I, section I.3

What this builds
----------------
A table in which every row is one option, and every row carries the complete
chain required by the Week 6 project milestone:

    (S, K, T, r, sigma, q, type)        the sampled inputs
            |
            v
    (C_BS, Delta, Gamma, Vega)          the Black-Scholes quantities
            |
            v
    x = [m, T, r, sigma]                the features
            |
            v
    y = C_BS / K                        the target

Nothing in the chain is held in memory between stages: every intermediate is a
column in the file. The audit in verify_traceability() below reads the file back
and recomputes the target from the stored inputs alone, which is what makes the
claim "fully traceable" checkable rather than asserted.

Design choices
--------------
All of these are argued in ML_experiment_plan.md; the short version:

  S fixed at 100      Black-Scholes is homogeneous of degree one in (S, K), so
                      the price over strike depends on S and K only through
                      m = ln(S/K). Sampling S as well would be redundant, and
                      would let a model fit level effects that cannot exist.

  K log-uniform       so that m is UNIFORM on [-0.6, 0.6]. Sampling K uniformly
                      would bunch m near zero.

  T log-uniform       short maturities are where the pricing surface is most
                      curved, and therefore where the linear baseline is
                      supposed to fail. Uniform T on [0.02, 2] would put only
                      about 1% of rows below T = 0.04 and hide the result.

  target C_BS / K     same homogeneity argument. With K varying, an unscaled
                      price carries a strike-driven scale that the scale-free
                      features cannot explain.

  CALLS ONLY          see below. This one is not obvious and was decided by
                      measurement, not by assumption.

Why the modelling set is calls only
------------------------------------
The first version of this dataset sampled calls and puts 50/50. That is wrong,
and the way it is wrong is worth recording.

A call and a put sharing the same (S, K, T, r, sigma, q) have the SAME feature
vector x = [m, T, r, sigma] but different prices. The target is therefore
two-valued at every point in feature space, and no function of x - linear,
polynomial, tree or network - can separate them. Roughly half the variance of
the target is unresolvable by construction. Measured: a linear fit on the mixed
set reaches R^2 = 0.06, and the degree-two expansion only reaches 0.20, which
would look like a damning verdict on the features when it is really a defect in
the sampling design.

Including an is_call indicator would fix the ambiguity, but it would also be
redundant, because put-call parity

    P = C - S e^{-qT} + K e^{-rT}

makes the put an EXACT affine function of the call at the same parameters. A
model trained on calls therefore prices puts exactly, with no additional
fitting. Adding put rows adds no information and doubles the data.

So the dataset is calls, and puts are recovered by parity. build_dataset() can
still produce the mixed set via include_puts=True; the driver uses it once, as
a labelled diagnostic, to show the effect rather than assert it.

Note that the parity audit does not need put rows: put_call_parity_error()
computes both sides from the parameters, so the structural check survives the
restriction intact.

The split (70/15/15) is drawn ONCE here, under the project seed, and written
into the file as a column. Downstream scripts read it; they never re-draw it.
That is what guarantees the Week 7 models see exactly the Week 6 test set.
"""

import os

import numpy as np
import pandas as pd

from black_scholes import bs_price, moneyness, put_call_parity_error
from greeks import delta, gamma, vega

__all__ = [
    "SEED", "FEATURES_BASIC", "FEATURES_EXTENDED", "TARGET",
    "sample_parameters", "build_dataset", "split_indices",
    "load_dataset", "get_xy", "verify_traceability",
]

SEED = 20260825

# Feature sets, fixed in ML_experiment_plan.md section 1.
FEATURES_BASIC = ["m", "T", "r", "sigma"]
FEATURES_EXTENDED = ["m", "T", "r", "sigma", "delta", "gamma", "vega"]
TARGET = "y"

# Sampling ranges, ML_experiment_plan.md section 6.
S_FIXED = 100.0
M_RANGE = (-0.6, 0.6)        # log-moneyness, uniform
T_RANGE = (1.0 / 52.0, 2.0)  # one week to two years, log-uniform
R_RANGE = (0.03, 0.08)
SIGMA_RANGE = (0.10, 0.60)
Q_FIXED = 0.015              # NIFTY 50 index dividend yield, carried from Week 5

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

DATA_PATH = os.path.join("data", "synthetic_option_dataset.csv")


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def sample_parameters(n, seed=SEED, include_puts=False):
    """Draw n sets of option parameters.

    Returns a DataFrame with S, K, T, r, sigma, q, option_type.

    m is drawn uniform and K is then back-solved from K = S exp(-m), which is
    the same thing as drawing K log-uniform but makes the intent explicit: it
    is the moneyness distribution that is being controlled, not the strike.

    include_puts=True produces the 50/50 mixed set. It exists so the driver can
    MEASURE the ambiguity described in the module docstring rather than assert
    it; it is not the modelling set.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    m = rng.uniform(M_RANGE[0], M_RANGE[1], n)
    K = S_FIXED * np.exp(-m)                      # so ln(S/K) == m exactly

    # log-uniform T: uniform in log space, exponentiated back
    logT = rng.uniform(np.log(T_RANGE[0]), np.log(T_RANGE[1]), n)
    T = np.exp(logT)

    r = rng.uniform(R_RANGE[0], R_RANGE[1], n)
    sigma = rng.uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], n)

    if include_puts:
        option_type = np.where(rng.random(n) < 0.5, "call", "put")
    else:
        option_type = np.full(n, "call", dtype=object)

    return pd.DataFrame({
        "S": np.full(n, S_FIXED),
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": np.full(n, Q_FIXED),
        "option_type": option_type,
    })


def split_indices(n, fractions=SPLIT_FRACTIONS, seed=SEED):
    """Return an array of 'train' / 'val' / 'test' labels of length n.

    One random permutation under the project seed, cut at the cumulative
    fractions. Disjoint by construction, which the self-test verifies rather
    than assumes.
    """
    f_train, f_val, f_test = fractions
    total = f_train + f_val + f_test
    if abs(total - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to 1, got %r" % (fractions,))

    rng = np.random.default_rng(seed + 1)          # +1: independent of sampling
    perm = rng.permutation(n)

    n_train = int(round(f_train * n))
    n_val = int(round(f_val * n))
    # test takes the remainder, so the three always sum to n exactly
    labels = np.empty(n, dtype=object)
    labels[perm[:n_train]] = "train"
    labels[perm[n_train:n_train + n_val]] = "val"
    labels[perm[n_train + n_val:]] = "test"
    return labels


# --------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------

def build_dataset(n=20000, seed=SEED, verbose=False, include_puts=False):
    """Build the full synthetic dataset as a DataFrame.

    Columns, in the order of the traceability chain:

        S K T r sigma q option_type        inputs
        C_BS delta gamma vega              Black-Scholes quantities
        m                                  derived feature
        y                                  target, C_BS / K
        split                              train / val / test

    include_puts=True returns the mixed call/put set, which is NOT the
    modelling set; see the module docstring.
    """
    df = sample_parameters(n, seed=seed, include_puts=include_puts)

    S = df["S"].to_numpy()
    K = df["K"].to_numpy()
    T = df["T"].to_numpy()
    r = df["r"].to_numpy()
    sg = df["sigma"].to_numpy()
    q = df["q"].to_numpy()

    # Greeks and prices are computed per option type. delta differs between
    # calls and puts; gamma and vega do not, but they are computed through the
    # same dispatch anyway so that a future change to greeks.py cannot silently
    # break the assumption.
    C = np.empty(n)
    dl = np.empty(n)
    gm = np.empty(n)
    vg = np.empty(n)
    for opt in ("call", "put"):
        k = (df["option_type"] == opt).to_numpy()
        if not k.any():
            continue
        C[k] = bs_price(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        dl[k] = delta(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        gm[k] = gamma(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        vg[k] = vega(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)

    df["C_BS"] = C
    df["delta"] = dl
    df["gamma"] = gm
    df["vega"] = vg
    df["m"] = moneyness(S, K)
    df["y"] = C / K
    df["split"] = split_indices(n, seed=seed)

    if verbose:
        print("  built %d rows: %d call, %d put"
              % (n, (df.option_type == "call").sum(),
                 (df.option_type == "put").sum()))
        if not include_puts:
            print("    calls only: puts follow exactly by put-call parity, so "
                  "put rows\n    would add no information and would make the "
                  "target two-valued\n    at every feature vector. See "
                  "ml_dataset.py docstring.")
        for s in ("train", "val", "test"):
            print("    %-6s %6d rows (%.1f%%)"
                  % (s, (df.split == s).sum(), 100 * (df.split == s).mean()))
    return df


def save_dataset(df, path=DATA_PATH, verbose=True):
    """Write the dataset, creating the directory if needed.

    Float columns are written at full repr precision, not rounded. Rounding
    here would break the traceability audit, which is the whole point of the
    file: a reader must be able to recompute C_BS from the stored inputs and
    land on the stored value to machine precision.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g")
    if verbose:
        print("  wrote %s  (%d rows, %d columns, %.1f KB)"
              % (path, len(df), df.shape[1], os.path.getsize(path) / 1024))
    return path


def load_dataset(path=DATA_PATH):
    """Read the dataset back from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found. Run build_dataset() and save_dataset(), or "
            "run postmidsem_week6.py." % path)
    return pd.read_csv(path)


def get_xy(df, features=FEATURES_BASIC, split=None, target=TARGET):
    """Return (X, y) as float arrays for one split, or for all rows.

    split is 'train', 'val', 'test', or None for everything.
    """
    sub = df if split is None else df[df["split"] == split]
    if len(sub) == 0:
        raise ValueError("split %r selected no rows" % (split,))
    X = sub[list(features)].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    return X, y


# --------------------------------------------------------------------------
# the traceability audit - the Week 6 project milestone
# --------------------------------------------------------------------------

def verify_traceability(df, verbose=True):
    """Recompute every stage of the chain from the stored inputs alone.

    This is the check the Week 6 milestone asks for. It deliberately reads only
    the raw input columns and recomputes everything downstream, so that a bug in
    build_dataset() which wrote an inconsistent row would be caught here rather
    than propagating into the models.

    Returns a dict of maximum absolute discrepancies.
    """
    S = df["S"].to_numpy()
    K = df["K"].to_numpy()
    T = df["T"].to_numpy()
    r = df["r"].to_numpy()
    sg = df["sigma"].to_numpy()
    q = df["q"].to_numpy()
    n = len(df)

    C = np.empty(n)
    dl = np.empty(n)
    gm = np.empty(n)
    vg = np.empty(n)
    for opt in ("call", "put"):
        k = (df["option_type"] == opt).to_numpy()
        if not k.any():
            continue
        C[k] = bs_price(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        dl[k] = delta(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        gm[k] = gamma(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)
        vg[k] = vega(S[k], K[k], T[k], r[k], sg[k], q[k], option=opt)

    out = {
        "C_BS": float(np.max(np.abs(C - df["C_BS"].to_numpy()))),
        "delta": float(np.max(np.abs(dl - df["delta"].to_numpy()))),
        "gamma": float(np.max(np.abs(gm - df["gamma"].to_numpy()))),
        "vega": float(np.max(np.abs(vg - df["vega"].to_numpy()))),
        "m": float(np.max(np.abs(np.log(S / K) - df["m"].to_numpy()))),
        "y": float(np.max(np.abs(C / K - df["y"].to_numpy()))),
    }

    # An independent structural check that does not go through bs_price at all:
    # put-call parity must hold for the parameter sets in the file.
    out["parity"] = float(np.max(np.abs(
        put_call_parity_error(S, K, T, r, sg, q))))

    if verbose:
        print("  stage                         max abs discrepancy")
        for k in ("m", "C_BS", "delta", "gamma", "vega", "y"):
            print("    %-26s  %.3e" % (k, out[k]))
        print("    %-26s  %.3e" % ("put-call parity (independent)", out["parity"]))
    return out


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    """Checks that must pass before the dataset is used for anything."""
    if verbose:
        print("=" * 78)
        print("ml_dataset.py  SELF-TEST")
        print("=" * 78)

    checks = []

    def chk(name, value, tol, fmt="%.3e"):
        ok = bool(value <= tol)
        checks.append((name, value, tol, ok))
        if verbose:
            print(("  %-52s " + fmt + "  %-9s %s")
                  % (name, value, "tol " + ("%.1e" % tol), "ok" if ok else "FAIL"))
        return ok

    n = 5000
    df = build_dataset(n=n, seed=SEED)

    # --- the traceability chain ---
    tr = verify_traceability(df, verbose=False)
    chk("traceability: C_BS recomputed from stored inputs", tr["C_BS"], 1e-12)
    chk("traceability: Greeks recomputed from stored inputs",
        max(tr["delta"], tr["gamma"], tr["vega"]), 1e-12)
    chk("traceability: m = ln(S/K) exactly", tr["m"], 1e-15)
    chk("traceability: y = C_BS / K exactly", tr["y"], 1e-18)
    chk("put-call parity on the sampled parameters", tr["parity"], 1e-10)

    # --- the split ---
    s = df["split"]
    sizes = {k: int((s == k).sum()) for k in ("train", "val", "test")}
    chk("split sizes sum to n", abs(sum(sizes.values()) - n), 0)
    chk("train fraction is 70%", abs(sizes["train"] / n - 0.70), 1e-3)
    chk("val fraction is 15%", abs(sizes["val"] / n - 0.15), 1e-3)
    chk("test fraction is 15%", abs(sizes["test"] / n - 0.15), 1e-3)
    # disjointness: every row has exactly one label, and the labels are the
    # three expected ones. With a label column this is the meaningful form of
    # D_train ^ D_val = empty (master plan 5.10).
    chk("split labels are exactly {train, val, test}",
        len(set(s.unique()) ^ {"train", "val", "test"}), 0)
    chk("no row is unlabelled", int(s.isna().sum()), 0)

    # --- the sampling distribution does what the plan says ---
    m = df["m"].to_numpy()
    chk("m within the stated range", max(abs(m.min() - M_RANGE[0]),
                                         abs(m.max() - M_RANGE[1])), 0.01)
    # uniformity of m: compare decile counts against n/10
    counts, _ = np.histogram(m, bins=10, range=M_RANGE)
    chk("m is uniform (max decile deviation)",
        float(np.max(np.abs(counts - n / 10)) / (n / 10)), 0.12)
    # log-uniformity of T: log T should be uniform
    lt = np.log(df["T"].to_numpy())
    cl, _ = np.histogram(lt, bins=10,
                         range=(np.log(T_RANGE[0]), np.log(T_RANGE[1])))
    chk("log T is uniform (max decile deviation)",
        float(np.max(np.abs(cl - n / 10)) / (n / 10)), 0.12)
    frac_short = float((df["T"] < 0.04).mean())
    if verbose:
        print("    short-maturity coverage: %.1f%% of rows have T < 0.04 "
              "(uniform T would give ~1%%)" % (100 * frac_short))
    chk("short maturities are well represented (T < 0.04)",
        max(0.0, 0.03 - frac_short), 1e-9)

    # --- financial sanity of the generated prices ---
    C = df["C_BS"].to_numpy()
    K = df["K"].to_numpy()
    S = df["S"].to_numpy()
    T = df["T"].to_numpy()
    r = df["r"].to_numpy()
    q = df["q"].to_numpy()
    is_call = (df["option_type"] == "call").to_numpy()
    chk("all prices non-negative", float(max(0.0, -C.min())), 1e-15)
    # no-arbitrage upper bounds
    ub = np.where(is_call, S * np.exp(-q * T), K * np.exp(-r * T))
    chk("prices within the no-arbitrage upper bound",
        float(np.max(np.maximum(0.0, C - ub))), 1e-10)
    # intrinsic lower bound on the forward
    lb = np.maximum(0.0, np.where(is_call,
                                  S * np.exp(-q * T) - K * np.exp(-r * T),
                                  K * np.exp(-r * T) - S * np.exp(-q * T)))
    chk("prices above the discounted-intrinsic lower bound",
        float(np.max(np.maximum(0.0, lb - C))), 1e-10)

    # --- Greeks sanity ---
    d_ = df["delta"].to_numpy()
    chk("call delta in (0, 1)",
        float(max(0.0, np.max(d_[is_call] - 1.0), np.max(-d_[is_call]))), 1e-12)
    chk("gamma strictly positive", float(max(0.0, -df["gamma"].min())), 1e-18)
    chk("vega strictly positive", float(max(0.0, -df["vega"].min())), 1e-18)

    # --- the modelling set is calls only, and the reason is checkable ---
    chk("modelling set contains no put rows",
        float((df["option_type"] != "call").sum()), 0)
    # The argument for excluding puts: admitting them makes the target
    # two-valued at one and the same feature vector. Demonstrate it by pricing
    # BOTH types at every sampled parameter set and comparing. The gap is what
    # no function of x = [m, T, r, sigma] could ever resolve.
    mixed = build_dataset(n=2000, seed=SEED, include_puts=True)
    Sm, Km = mixed["S"].to_numpy(), mixed["K"].to_numpy()
    Tm, rm = mixed["T"].to_numpy(), mixed["r"].to_numpy()
    sm, qm = mixed["sigma"].to_numpy(), mixed["q"].to_numpy()
    y_call = bs_price(Sm, Km, Tm, rm, sm, qm, option="call") / Km
    y_put = bs_price(Sm, Km, Tm, rm, sm, qm, option="put") / Km
    gap = np.abs(y_call - y_put)
    spread = float(np.std(np.concatenate([y_call, y_put])))
    if verbose:
        print("    admitting puts: median |y_call - y_put| at the SAME feature"
              "\n    vector is %.4f, against a target spread of %.4f - so "
              "roughly\n    half the variance would be unresolvable by any "
              "function of x" % (float(np.median(gap)), spread))
    chk("admitting puts would make the target two-valued at one x",
        float(1.0 if np.median(gap) < 0.2 * spread else 0.0), 0.0)
    pm = put_call_parity_error(Sm, Km, Tm, rm, sm, qm)
    chk("parity audit still works without put rows",
        float(np.max(np.abs(pm))), 1e-10)

    # --- reproducibility ---
    df2 = build_dataset(n=n, seed=SEED)
    same = float(np.max(np.abs(df["y"].to_numpy() - df2["y"].to_numpy())))
    chk("same seed reproduces the dataset exactly", same, 0.0)
    df3 = build_dataset(n=n, seed=SEED + 7)
    diff = float(np.max(np.abs(df["y"].to_numpy() - df3["y"].to_numpy())))
    if verbose:
        print("    a different seed gives a different sample (max |dy| = %.3e)" % diff)
    chk("a different seed is actually different", float(diff > 1e-6) - 1.0, 0.0)

    # --- the extended feature set is present and finite ---
    for col in FEATURES_EXTENDED + [TARGET]:
        chk("column %r is finite" % col,
            float(np.sum(~np.isfinite(df[col].to_numpy()))), 0)

    n_fail = sum(1 for *_, ok in checks if not ok)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0 else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    ok = _self_test()
    raise SystemExit(0 if ok else 1)
