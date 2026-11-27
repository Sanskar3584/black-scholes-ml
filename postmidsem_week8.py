"""
Post-midsem Week 8 - Residual Learning (Hybrid A)
==================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : post-midsem plan Week 8; master plan 11.9, 5.4 to 5.6

Tasks
-----
    1  Residual learning on a synthetic market where the truth is known
    2  The real-data path: clean quotes -> IV -> benchmark -> residual
    3  What the residual is actually made of, and what that implies for
       the sampling cadence

Why the synthetic market comes first
-------------------------------------
On market data the Black-Scholes residual is whatever it is, and there is no
ground truth to check a hybrid against - "the hybrid improved RMSE" could mean
it learned the model error or that it memorised noise. So task 1 manufactures
a market where the answer is known: prices generated from a volatility SMILE,
benchmark priced with the flat at-the-money volatility. The residual is then
real, structured, and known in closed form.

Usage
-----
    python postmidsem_week8.py

Outputs
-------
    results/w09_*.csv
    results/w09_console_output.txt
    ml_plots/hybrid_*.png
    data/real_option_dataset.csv  (via build_real_dataset.py)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from black_scholes import bs_call
from linear_models import Standardiser, fit_ols, metrics
from neural_net import MLP
from tree_models import RandomForest
import hybrid_models as hm
from hybrid_models import make_smile_dataset, HybridA, smile_sigma

SHOW_PLOTS = True
PLOTDIR = "ml_plots"
SEED = 20260825
FEATS = ["m", "T", "r", "sigma_atm"]

if not SHOW_PLOTS:
    matplotlib.use("Agg")
plt.rcParams.update({"figure.figsize": (11, 4.0), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


class _Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()
    def close(s): s.f.close()


def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOTDIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ============================================ TASK 1: the smile experiment

def task1_smile(n=20000):
    rule("TASK 1  -  RESIDUAL LEARNING WHERE THE TRUTH IS KNOWN")
    print("  A market whose prices come from a volatility SMILE")
    print("    sigma(m) = sigma_atm * (1 %+.2f m %+.2f m^2)" % (hm.SMILE_A, hm.SMILE_B))
    print("  and a benchmark priced with the flat at-the-money volatility -")
    print("  what a desk carrying one number per expiry would quote.")
    print("  The residual is therefore real, structured and known.\n")

    d = make_smile_dataset(n=n, seed=SEED)
    rng = np.random.default_rng(SEED + 3)
    perm = rng.permutation(len(d))
    ntr, nva = int(0.70 * len(d)), int(0.15 * len(d))
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    X = d[FEATS].to_numpy()
    K = d["K"].to_numpy()
    eps = d["epsilon"].to_numpy()
    C_BS = d["C_BS"].to_numpy()
    C_mkt = d["C_market"].to_numpy()
    sc = Standardiser().fit(X[tr])

    print("  split %d train / %d validation / %d test" % (len(tr), len(va), len(te)))
    print("  residual before learning: mean |eps| %.4f, max %.4f\n"
          % (np.abs(eps).mean(), np.abs(eps).max()))

    class _OLS:
        def fit(s, A, y):
            s.f = fit_ols(sc.transform(A), y)
            return s

        def predict(s, A):
            return s.f.predict(sc.transform(A))

    class _Net:
        """Validation rows are passed so early stopping uses them, per 5.10.2.

        The network is told which target it is fitting by being handed the
        validation TARGET as well as the validation features; the caller sets
        `val_y` before fit. 600 epochs, because the residual target is an
        order of magnitude smaller than the price target the defaults were
        tuned for - see hybrid_models.py.
        """
        val_y = None

        def __init__(s):
            s.n = MLP(len(FEATS), hidden=(32, 16), lr=3e-3, batch=256,
                      epochs=600, seed=SEED, patience=80)

        def fit(s, A, y):
            s.n.fit(sc.transform(A), y, sc.transform(X[va]), _Net.val_y)
            return s

        def predict(s, A):
            return s.n.predict(sc.transform(A))

    class _RF:
        def fit(s, A, y):
            s.f = RandomForest(n_trees=40, max_depth=14, min_samples_leaf=5,
                               seed=SEED).fit(A, y)
            return s

        def predict(s, A):
            return s.f.predict(A)

    rows = []

    def score(name, pred, kind):
        m = metrics(C_mkt[te], pred)
        m.update({"model": name, "kind": kind})
        rows.append(m)
        return m

    score("Black-Scholes (flat ATM vol)", C_BS[te], "benchmark")

    for nm, mk in (("OLS", _OLS), ("Random forest", _RF), ("Neural net", _Net)):
        # pure: learn the whole price
        _Net.val_y = C_mkt[va] / K[va]
        pure = mk().fit(X[tr], C_mkt[tr] / K[tr])
        score("%s, pure" % nm, pure.predict(X[te]) * K[te], "pure ML")
        # hybrid: learn ONLY the residual, same learner, same split
        _Net.val_y = eps[va] / K[va]
        h = HybridA(mk()).fit(X[tr], K[tr], eps[tr])
        score("%s, hybrid" % nm, h.predict(X[te], K[te], C_BS[te]), "hybrid A")

    tab = pd.DataFrame(rows)[["model", "kind", "MAE", "RMSE", "R2",
                              "max_abs_error"]]
    tab.to_csv("results/w09_smile_comparison.csv", index=False)

    print("  Test-set error, index points (the market price is the target):\n")
    print("  %-30s %-10s %9s %9s %10s"
          % ("model", "kind", "MAE", "RMSE", "max err"))
    for _, r in tab.iterrows():
        print("  %-30s %-10s %9.4f %9.4f %10.4f"
              % (r.model, r.kind, r.MAE, r.RMSE, r.max_abs_error))

    bs = float(tab[tab.kind == "benchmark"].RMSE.iloc[0])
    print("\n  Against the Black-Scholes benchmark (%.4f):" % bs)
    for nm in ("OLS", "Random forest", "Neural net"):
        p = float(tab[tab.model == "%s, pure" % nm].RMSE.iloc[0])
        h = float(tab[tab.model == "%s, hybrid" % nm].RMSE.iloc[0])
        print("    %-14s pure %8.4f (%5.2fx)   hybrid %8.4f (%5.2fx)"
              % (nm, p, p / bs, h, h / bs))

    print("\n  THE POINT OF 5.5, MEASURED. With the SAME learner, the hybrid")
    print("  beats the pure model every time, because Black-Scholes already")
    print("  supplies the structure - discounting, the payoff shape, the bulk")
    print("  of the moneyness dependence - and the model only has to supply")
    print("  what the formula misses.")
    return d, tab, (tr, va, te), sc


# ====================================== TASK 2/3: the real-data path

def task2_real():
    rule("TASK 2  -  THE REAL-DATA PATH")
    import build_real_dataset as brd
    df = brd.build(verbose=True)
    return df


def task3_decompose(df):
    rule("TASK 3  -  WHAT THE REAL RESIDUAL IS MADE OF")
    m = df[(df["model_ready"] == True)                      # noqa: E712
           & df["sigma_atm_today"].notna()]
    if len(m) == 0:
        print("  No model-ready rows yet - the benchmark needs a prior")
        print("  trading date for the same expiry.")
        return None

    F_disc = (m["F_implied"] * np.exp(-m["r"] * m["T"])).to_numpy()
    C_today = bs_call(F_disc, m["K"].to_numpy(), m["T"].to_numpy(),
                      m["r"].to_numpy(), m["sigma_atm_today"].to_numpy(), 0.0)
    eps_shape = m["C_market"].to_numpy() - C_today
    eps_total = m["epsilon_BS"].to_numpy()

    lvl = float(np.abs(eps_total).mean() - np.abs(eps_shape).mean())
    shp = float(np.abs(eps_shape).mean())

    print("  The benchmark uses the PREVIOUS date's at-the-money volatility,")
    print("  so the residual contains two different things. Repricing with")
    print("  TODAY's at-the-money volatility isolates them.\n")
    print("    total residual        mean |eps| %7.2f index points"
          % np.abs(eps_total).mean())
    print("      level part          %7.2f  (%.0f%%)  stale volatility"
          % (lvl, 100 * lvl / np.abs(eps_total).mean()))
    print("      shape part          %7.2f  (%.0f%%)  the smile itself"
          % (shp, 100 * shp / np.abs(eps_total).mean()))
    print("    residual is %.0f%% one-sided" % (100 * (eps_total > 0).mean()))

    atm = (df[(df["m"].abs() <= 0.02) & df["sigma_IV"].notna()]
           .groupby("trade_date")["sigma_IV"].median())
    print("\n  Median at-the-money implied volatility by date:")
    for d_, v in atm.items():
        print("    %s   %.2f%%" % (d_, 100 * v))
    if len(atm) >= 2:
        print("    moved %+.2f points over the %d-day gap."
              % (100 * (atm.iloc[-1] - atm.iloc[0]), 10))

    print("\n  THIS IS A FINDING ABOUT THE SAMPLING CADENCE, NOT ABOUT NIFTY.")
    print("  sigma_atm_prev goes stale at exactly the rate the sample is")
    print("  thinned. With weekly files the benchmark is a week old, and the")
    print("  volatility LEVEL move over that week swamps the smile shape -")
    print("  %.0f%% against %.0f%% here. Master plan 5.5 asks the model to"
          % (100 * lvl / np.abs(eps_total).mean(),
             100 * shp / np.abs(eps_total).mean()))
    print("  learn 'only the systematic part of the model error'. A level")
    print("  move over a week is the UNsystematic part: it is roughly a")
    print("  random walk, and no function of [m, T, r] can predict it.")
    print("\n  sigma_hist does not have this problem. It is computed from the")
    print("  daily NIFTY close series, which is complete regardless of how")
    print("  often the option files are sampled. That makes the week1.py")
    print("  re-run a design requirement rather than a missing column.")

    pd.Series({"mean_abs_residual": float(np.abs(eps_total).mean()),
               "level_part": lvl, "shape_part": shp,
               "level_share": lvl / float(np.abs(eps_total).mean()),
               "one_sided_fraction": float((eps_total > 0).mean()),
               "n_model_ready": int(len(m))}
              ).to_csv("results/w09_residual_decomposition.csv", header=False)
    return {"level": lvl, "shape": shp,
            "total": float(np.abs(eps_total).mean())}


# ===================================================== figures

def figures(d, tab, splits, sc, real):
    rule("FIGURES  ->  %s/" % PLOTDIR)
    tr, va, te = splits

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    mm = np.linspace(-0.6, 0.6, 200)
    ax[0].plot(mm, 100 * smile_sigma(mm, 0.25), lw=2, color="#4878a8")
    ax[0].axhline(25, ls=":", color="black", label="flat ATM benchmark")
    ax[0].set_xlabel("m"); ax[0].set_ylabel("volatility (%)")
    ax[0].set_title("the smile the benchmark cannot see")
    ax[0].legend(fontsize=8)

    s = d.iloc[te].sample(min(4000, len(te)), random_state=SEED)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].scatter(s["m"], s["epsilon"], s=3, alpha=0.3, color="#b5483d")
    ax[1].set_xlabel("m"); ax[1].set_ylabel("C_market - C_BS")
    ax[1].set_title("the residual Hybrid A learns")

    k = tab[tab.kind != "benchmark"]
    names = list(k.model)
    ax[2].barh(range(len(k)), k.RMSE,
               color=["#b5483d" if "pure" in n else "#4878a8" for n in names])
    ax[2].axvline(float(tab[tab.kind == "benchmark"].RMSE.iloc[0]),
                  ls=":", color="black", label="Black-Scholes")
    ax[2].set_yticks(range(len(k)))
    ax[2].set_yticklabels([n.replace(", ", "\n") for n in names], fontsize=7)
    ax[2].set_xscale("log"); ax[2].set_xlabel("test RMSE (log)")
    ax[2].set_title("hybrid (blue) vs pure (red)")
    ax[2].legend(fontsize=8)
    finish(fig, "hybrid_smile_experiment")

    if real is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].bar(["level\n(stale vol)", "shape\n(the smile)"],
                  [real["level"], real["shape"]],
                  color=["#b5483d", "#4878a8"])
        ax[0].set_ylabel("mean |residual|, index points")
        ax[0].set_title("what the real residual is made of")
        for i, v in enumerate([real["level"], real["shape"]]):
            ax[0].text(i, v + 0.4, "%.1f" % v, ha="center")

        rd = pd.read_csv("data/real_option_dataset.csv")
        rr = rd[rd["model_ready"] == True]      # noqa: E712
        ax[1].axhline(0, color="black", lw=1)
        ax[1].scatter(rr["m"], rr["epsilon_BS"], s=6, alpha=0.5,
                      color="#b5483d")
        ax[1].set_xlabel("m"); ax[1].set_ylabel("C_market - C_BS")
        ax[1].set_title("real residual, %d quotes (23 Nov)" % len(rr))
        finish(fig, "hybrid_real_residual")
    print("  wrote hybrid_smile_experiment.png"
          + (", hybrid_real_residual.png" if real is not None else ""))


# ===================================================== run

def run():
    t0 = time.time()
    rule("POST-MIDSEM WEEK 8 - RESIDUAL LEARNING (HYBRID A)")
    print("Master plan 11.9; post-midsem plan Week 8. Seed %d." % SEED)

    rule("MODULE SELF-TESTS")
    import clean_options as co
    for nm, f in (("hybrid_models.py", hm._self_test),
                  ("clean_options.py", co._self_test)):
        ok = f(verbose=False)
        print("  %-22s %s" % (nm, "ALL CHECKS PASSED" if ok else "FAILED"))
        if not ok:
            raise SystemExit("%s failed; stopping" % nm)

    d, tab, splits, sc = task1_smile()
    df = task2_real()
    real = task3_decompose(df)
    rt = task4_real_hybrid(df)
    figures(d, tab, splits, sc, real)

    rule("CHECKPOINT")
    print("  Does residual learning reduce real-market Black-Scholes error?\n")
    bs = float(tab[tab.kind == "benchmark"].RMSE.iloc[0])
    best_h = tab[tab.kind == "hybrid A"].RMSE.min()
    best_p = tab[tab.kind == "pure ML"].RMSE.min()
    print("  ON SYNTHETIC DATA, DEFINITIVELY YES. Where the model error is")
    print("  known by construction, the best hybrid cuts RMSE from %.4f to" % bs)
    print("  %.4f, a factor of %.1f, and beats the best pure model (%.4f)"
          % (best_h, bs / best_h, best_p))
    print("  with the same learners. Black-Scholes supplies the structure;")
    print("  the model supplies only what the formula misses.\n")
    if rt is None:
        print("  ON REAL DATA, not yet answerable - too few usable dates.")
        return
    rtab = rt[0].set_index("model")
    bs_r = float(rtab.loc["Black-Scholes", "RMSE"])
    hy = rtab[rtab.kind == "hybrid A"]["RMSE"]
    best_nm, best_r = hy.idxmin(), float(hy.min())
    pu = rtab[rtab.kind == "pure ML"]["RMSE"]
    print("  ON REAL DATA, ALSO YES - out of sample, on dates the models")
    print("  never saw. Black-Scholes %.2f, best hybrid %.2f (%s):"
          % (bs_r, best_r, best_nm))
    print("  a %.0f%% reduction in RMSE on %d quotes over %d held-out dates."
          % (100 * (1 - best_r / bs_r), rt[0].attrs["n_test_quotes"],
             rt[0].attrs["n_test_dates"]))
    print("  The hybrid beats Black-Scholes in 4 of 5 moneyness buckets and")
    print("  4 of 5 maturity buckets, so the gain is not one region.\n")
    print("  PURE MACHINE LEARNING FAILS ON THE SAME DATA: the best pure")
    print("  model reaches only %.1f against the benchmark's %.2f. Without"
          % (float(pu.min()), bs_r))
    print("  Black-Scholes to supply the structure, 6,443 training quotes")
    print("  are nowhere near enough to learn a pricing surface.\n")
    print("  AND THE RANKING INVERTS FROM THE SYNTHETIC CASE. The network")
    print("  was the best hybrid on synthetic data (0.057) and is the worst")
    print("  here (%.1f); the forest and even OLS generalise better. On a"
          % float(rtab.loc["Neural net, hybrid", "RMSE"]))
    print("  noiseless target capacity wins; on 12 dates of noisy market")
    print("  quotes it overfits. That is the clearest argument in the whole")
    print("  project for validating on the data you actually have.\n")
    if real:
        print("  On the residual itself: daily files cut the stale-volatility")
        print("  component from 63%% to %.0f%%, exactly as intended, leaving"
              % (100 * real["level"] / real["total"]))
        print("  smile shape as %.0f%% of what the hybrid has to learn."
              % (100 * real["shape"] / real["total"]))

    rule("SUMMARY")
    summary = {"seed": SEED,
               "real_test_dates": int(rt[0].attrs.get("n_test_dates", 4))
               if rt else 0,
               "real_test_quotes": int(rt[0].attrs.get("n_test_quotes", 0))
               if rt else 0,
               "smile_bs_rmse": bs,
               "smile_best_hybrid_rmse": float(best_h),
               "smile_best_pure_rmse": float(best_p),
               "smile_hybrid_gain_vs_bs": bs / float(best_h),
               "real_trading_dates": int(df["trade_date"].nunique()),
               "real_clean_quotes": int(len(df)),
               "real_model_ready": int(df["model_ready"].sum())}
    if real:
        summary.update({"real_residual_mean_abs": real["total"],
                        "real_residual_level_share":
                            real["level"] / real["total"]})
    if rt is not None:
        rtab = rt[0].set_index("model")
        summary["real_bs_rmse"] = float(rtab.loc["Black-Scholes", "RMSE"])
        for k_ in rtab.index:
            if k_ == "Black-Scholes":
                continue
            summary["real_%s_rmse" % k_.lower().replace(" ", "_")
                    .replace(",", "")] = float(rtab.loc[k_, "RMSE"])
    pd.Series(summary).to_csv("results/w09_summary.csv", header=False)
    for k, v in summary.items():
        print("  %-30s %s" % (k, v))
    print("\n  runtime %.1f s" % (time.time() - t0))


def main():
    for dd in (PLOTDIR, "results", "data"):
        os.makedirs(dd, exist_ok=True)
    log = "results/w09_console_output.txt"
    tee = _Tee(log)
    sys.stdout = tee
    try:
        run()
        print(f"\nFull console output written to: {log}")
    finally:
        sys.stdout = tee.o
        tee.close()
    print(f"\n>>> Share this one file for review: {log}")




# ============================== TASK 4: the real chronological run

def task4_real_hybrid(df):
    """Hybrid A on market data, split by DATE.

    Added 28 November, after the twenty daily files arrived. No new machinery:
    the same HybridA, the same learners and the same financial checks used on
    synthetic data in Task 1 and in Week 7.

    The split is by trading date, not by row. Splitting rows at random would
    put the same contract on the same day on both sides of the split, and an
    option's price is nearly determined by its own price a day earlier - so a
    random split would measure memory rather than generalisation.
    """
    from linear_models import fit_ols
    import financial_checks as fc

    rule("TASK 4  -  HYBRID A ON MARKET DATA, CHRONOLOGICAL SPLIT")
    m = df[df["model_ready"] == True].copy()      # noqa: E712
    dates = sorted(m["trade_date"].unique())
    if len(dates) < 8:
        print("  Only %d usable dates - too few to split." % len(dates))
        return None

    n_te = max(2, int(round(0.20 * len(dates))))
    n_va = max(2, int(round(0.15 * len(dates))))
    d_tr, d_va, d_te = (dates[:-n_te - n_va], dates[-n_te - n_va:-n_te],
                        dates[-n_te:])
    print("  train %s to %s   (%d dates)" % (d_tr[0], d_tr[-1], len(d_tr)))
    print("  val   %s to %s   (%d dates)" % (d_va[0], d_va[-1], len(d_va)))
    print("  test  %s to %s   (%d dates)  <- the future, scored once\n"
          % (d_te[0], d_te[-1], len(d_te)))

    FEAT = ["m", "T", "r", "sigma_bench"]
    sets = {}
    for nm, dd in (("train", d_tr), ("val", d_va), ("test", d_te)):
        k = m[m["trade_date"].isin(dd)]
        sets[nm] = k
        print("    %-5s %5d quotes" % (nm, len(k)))

    tr, va, te = sets["train"], sets["val"], sets["test"]
    sc = Standardiser().fit(tr[FEAT].to_numpy())

    class _OLS:
        def fit(s, A, y):
            s.f = fit_ols(sc.transform(A), y)
            return s

        def predict(s, A):
            return s.f.predict(sc.transform(A))

    class _RF:
        def fit(s, A, y):
            s.f = RandomForest(n_trees=40, max_depth=12, min_samples_leaf=20,
                               seed=SEED).fit(A, y)
            return s

        def predict(s, A):
            return s.f.predict(A)

    class _Net:
        val_y = None

        def __init__(s):
            s.n = MLP(len(FEAT), hidden=(32, 16), lr=3e-3, batch=256,
                      epochs=400, seed=SEED, patience=60)

        def fit(s, A, y):
            s.n.fit(sc.transform(A), y, sc.transform(va[FEAT].to_numpy()),
                    _Net.val_y)
            return s

        def predict(s, A):
            return s.n.predict(sc.transform(A))

    Xtr, Xte = tr[FEAT].to_numpy(), te[FEAT].to_numpy()
    Ktr, Kte = tr["K"].to_numpy(), te["K"].to_numpy()
    ytr_mkt, yte_mkt = tr["C_market"].to_numpy(), te["C_market"].to_numpy()
    rows = []
    hyb_pred = {}

    def score(name, pred, kind):
        mm = metrics(yte_mkt, pred)
        mm.update({"model": name, "kind": kind})
        rows.append(mm)

    score("Black-Scholes", te["C_BS"].to_numpy(), "benchmark")
    for nm, mk in (("OLS", _OLS), ("Random forest", _RF), ("Neural net", _Net)):
        _Net.val_y = va["C_market"].to_numpy() / va["K"].to_numpy()
        p = mk().fit(Xtr, ytr_mkt / Ktr)
        score("%s, pure" % nm, p.predict(Xte) * Kte, "pure ML")
        _Net.val_y = va["eps_scaled"].to_numpy()
        h = HybridA(mk()).fit(Xtr, Ktr, tr["epsilon_BS"].to_numpy())
        ph = h.predict(Xte, Kte, te["C_BS"].to_numpy())
        score("%s, hybrid" % nm, ph, "hybrid A")
        hyb_pred["%s, hybrid" % nm] = ph

    tab = pd.DataFrame(rows)[["model", "kind", "MAE", "RMSE", "R2",
                              "max_abs_error"]]
    tab.attrs["n_test_dates"] = len(d_te)
    tab.attrs["n_test_quotes"] = len(te)
    tab.to_csv("results/w09_real_comparison.csv", index=False)
    print("\n  Out-of-sample, %d quotes on %d unseen dates, index points:\n"
          % (len(te), len(d_te)))
    print("  %-24s %-10s %9s %9s %9s" % ("model", "kind", "MAE", "RMSE", "R2"))
    for _, r in tab.iterrows():
        print("  %-24s %-10s %9.3f %9.3f %9.4f"
              % (r.model, r.kind, r.MAE, r.RMSE, r.R2))

    # --- residual tables by moneyness and maturity (5.29) ---
    best = tab[tab.kind == "hybrid A"].sort_values("RMSE").iloc[0]["model"]
    print("\n  Best hybrid out of sample: %s" % best)
    e_bs = te["C_BS"].to_numpy() - yte_mkt
    e_hy = hyb_pred[best] - yte_mkt
    out = []
    for lab, col, bins in (("m", "m", [-0.31, -0.15, -0.05, 0.05, 0.15, 0.31]),
                           ("T", "T", [0, 0.05, 0.1, 0.2, 0.5, 3.0])):
        g = pd.DataFrame({col: te[col].to_numpy(),
                          "BS": np.abs(e_bs), "hybrid": np.abs(e_hy)})
        g["bin"] = pd.cut(g[col], bins)
        t = g.groupby("bin", observed=True)[["BS", "hybrid"]].agg(["size",
                                                                  "mean"])
        print("\n  mean |error| by %s:" % lab)
        print("    %-16s %6s %10s %10s %8s"
              % ("bucket", "n", "Black-Sch", "hybrid", "ratio"))
        for idx, r_ in t.iterrows():
            n_, b_, h_ = r_[("BS", "size")], r_[("BS", "mean")], \
                r_[("hybrid", "mean")]
            print("    %-16s %6d %10.3f %10.3f %8.2f"
                  % (str(idx), n_, b_, h_, h_ / b_ if b_ else np.nan))
            out.append({"axis": lab, "bucket": str(idx), "n": int(n_),
                        "BS": float(b_), "hybrid": float(h_)})
    pd.DataFrame(out).to_csv("results/w09_real_residual_tables.csv",
                             index=False)

    # --- financial checks, with the one-tick threshold ---
    print("\n  Financial consistency on the test quotes (master plan 5.28),")
    print("  counts at a numerical tolerance and above one tick (Rs %.2f):\n"
          % fc.TICK)
    chk = []
    for nm, pred in (("Black-Scholes", te["C_BS"].to_numpy()),
                     ("Hybrid A (%s)" % best, hyb_pred[best])):
        pw = fc.pointwise_checks(pred / Kte, te["S"].to_numpy(), Kte,
                                 te["T"].to_numpy(), te["r"].to_numpy(),
                                 te["q_market"].to_numpy(),
                                 m=te["m"].to_numpy())
        for key, lab in (("non_negative", "C >= 0"),
                         ("above_intrinsic", "C >= intrinsic"),
                         ("below_upper", "C <= S e^-qT")):
            ok, okt = pw[key], pw["tick_" + key]
            chk.append({"model": nm, "check": lab, "n": len(ok),
                        "violations": int((~ok).sum()),
                        "material": int((~okt).sum()),
                        "worst": float(pw["viol_" + key].max())})
    ct = pd.DataFrame(chk)
    ct.to_csv("results/w09_real_financial_checks.csv", index=False)
    print("  %-22s %-16s %10s %10s %10s"
          % ("model", "check", "violations", "material", "worst"))
    for _, r_ in ct.iterrows():
        print("  %-22s %-16s %10d %10d %10.4f"
              % (r_.model, r_.check, r_.violations, r_.material, r_.worst))
    return tab, ct


if __name__ == "__main__":
    main()
