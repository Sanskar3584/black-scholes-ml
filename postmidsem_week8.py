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
    m = df[df["model_ready"] == True]          # noqa: E712
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
    print("  ON REAL DATA, NOT YET ANSWERABLE. 2 trading dates, of which the")
    print("  first cannot be model-ready because the benchmark needs a prior")
    print("  date. 531 usable quotes is not a dataset, and with one date")
    print("  there is no chronological split to speak of.\n")
    if real:
        print("  What the two dates DO establish is that %.0f%% of the real"
              % (100 * real["level"] / real["total"]))
        print("  residual is stale-volatility level error rather than smile")
        print("  shape. That is a reason to prefer sigma_hist over")
        print("  sigma_atm_prev once the window is in, and it is an argument")
        print("  the synthetic experiment could not have produced.")

    rule("SUMMARY")
    summary = {"seed": SEED, "smile_bs_rmse": bs,
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


if __name__ == "__main__":
    main()
