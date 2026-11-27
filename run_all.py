"""
run_all.py - rerun every result in the report, in order
========================================================

    python run_all.py              # everything
    python run_all.py --tests      # module self-tests only (fast)

Deliverable 18, the reproducibility package. Seed 20260825 throughout.
Real-data steps are skipped with a clear message if the bhavcopy files are
absent, so the script completes on a fresh clone.
"""
import os, subprocess, sys, time

TESTS = ["black_scholes.py", "greeks.py", "implied_volatility.py",
         "binomial.py", "monte_carlo.py", "ml_dataset.py", "linear_models.py",
         "tree_models.py", "neural_net.py", "financial_checks.py",
         "hybrid_models.py", "test_nse_data.py"]

STAGES = [("week1.py",            "Week 1  returns and distributions"),
          ("week2.py",            "Week 2  Brownian motion and GBM"),
          ("week3.py",            "Week 3  Ito and Black-Scholes"),
          ("postmidsem_week3.py", "Week 4  Greeks and implied volatility"),
          ("postmidsem_week4.py", "Week 5  binomial lattice"),
          ("postmidsem_week5.py", "Week 6  Monte Carlo and the gate"),
          ("postmidsem_week6.py", "Week 7  synthetic dataset and regression"),
          ("postmidsem_week7.py", "Week 8  tree, network, Stage Gate 2"),
          ("postmidsem_week8.py", "Week 9  residual hybrid")]

REAL = [("clean_options.py",      "clean the bhavcopies"),
        ("build_real_dataset.py", "assemble the real modelling set")]


def run(script, label, env=None):
    print("\n" + "=" * 70)
    print("  %-26s %s" % (script, label))
    print("=" * 70, flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, script], env=env)
    print("  -> exit %d in %.1f s" % (r.returncode, time.time() - t))
    return r.returncode == 0


def main():
    env = dict(os.environ, MPLBACKEND="Agg")   # never block on a figure window
    ok = True
    for s in TESTS:
        if os.path.exists(s):
            ok &= run(s, "self-test", env)
    if "--tests" in sys.argv:
        raise SystemExit(0 if ok else 1)

    have_real = os.path.isdir("data/raw/fo_bhavcopy") and any(
        f.endswith(".zip") for f in os.listdir("data/raw/fo_bhavcopy"))
    for s, lab in STAGES:
        if s == "postmidsem_week8.py" and not have_real:
            print("\n  SKIPPING the real-data path: no bhavcopies in "
                  "data/raw/fo_bhavcopy.\n  See data_sources.md section 5.")
        if os.path.exists(s):
            ok &= run(s, lab, env)
    if have_real:
        for s, lab in REAL:
            ok &= run(s, lab, env)

    print("\n" + "=" * 70)
    print("  ALL STAGES COMPLETED" if ok else "  *** SOME STAGES FAILED ***")
    print("=" * 70)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
