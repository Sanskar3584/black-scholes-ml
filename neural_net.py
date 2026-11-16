"""
neural_net.py - a small feed-forward network, written out
==========================================================

Project : AI/ML-Augmented Black-Scholes Framework
Ref     : master plan 5.19, 5.20, 11.8 task B; ML_experiment_plan.md 9.1

Architecture, fixed by master plan 5.20
----------------------------------------
    p -> 32 -> 16 -> 1,  ReLU on both hidden layers, linear output.

Section 5.20 is explicit that the objective is "controlled nonlinear
approximation, not maximum possible network complexity", so there is no
architecture search and no depth beyond two hidden layers.

Why the passes are written out
-------------------------------
Section 5.31 requires an answer, for every model, to what is being minimised,
with what initialisation and what hyperparameters. "The library does it" is
not an answer. So:

    z1 = X W1 + b1      h1 = relu(z1)
    z2 = h1 W2 + b2     h2 = relu(z2)
    yhat = h2 W3 + b3

    L = mean( (yhat - y)^2 )

and backward, with  d = dL/d(pre-activation)  at each layer,

    d3 = 2 (yhat - y) / n
    dW3 = h2' d3        db3 = sum d3
    d2 = (d3 W3') * 1[z2 > 0]
    dW2 = h1' d2        db2 = sum d2
    d1 = (d2 W2') * 1[z1 > 0]
    dW1 = X' d1         db1 = sum d1

The self-test checks every one of those gradients against a central finite
difference. That check is the whole reason this is safe to write by hand: a
sign error in backprop produces a network that still trains, just worse, and
is otherwise invisible.

Initialisation
--------------
He initialisation, `W ~ N(0, 2/fan_in)`, which is the right variance for ReLU:
ReLU zeroes half the pre-activations, so the factor of 2 keeps the forward
variance stable across layers. Biases start at zero.

Optimiser
---------
Adam, with the standard defaults (0.9, 0.999, 1e-8) and a fixed learning rate.
Full-batch would be simpler but converges slowly on 14,000 rows; mini-batches
of 256 are used, shuffled each epoch under the project seed.

Early stopping
--------------
Validation loss is evaluated every epoch and the best weights are kept. This
uses the validation split for exactly what master plan 5.10.2 says it is for -
model selection - and keeps the test split untouched.

The deep-wing prediction
------------------------
`ML_experiment_plan.md` section 9.1 records, before any fitting, the
expectation that an unconstrained linear output will produce NEGATIVE prices
in the deep wings: the true target there is ~0, and squared-error loss has no
reason to prefer a small positive error to a small negative one. `softplus`
output is available via `output="softplus"` so the constrained variant can be
compared, but the default is unconstrained, because the point is to MEASURE
the failure rather than to design it away before anyone sees it.
"""

import numpy as np

__all__ = ["MLP", "SEED"]

SEED = 20260825


def _softplus(z):
    # log(1 + e^z), computed so large z does not overflow
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))


class MLP:
    """p -> 32 -> 16 -> 1 feed-forward network, ReLU, MSE, Adam."""

    def __init__(self, n_features, hidden=(32, 16), lr=3e-3, batch=256,
                 epochs=300, seed=SEED, output="linear", l2=0.0,
                 patience=40):
        self.sizes = [n_features] + list(hidden) + [1]
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.seed = seed
        self.output = output
        self.l2 = l2
        self.patience = patience
        self._init_params()

    # -- parameters -------------------------------------------------------
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            # He initialisation: variance 2/fan_in, correct for ReLU
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in),
                                     (fan_in, self.sizes[i + 1])))
            self.b.append(np.zeros(self.sizes[i + 1]))

    def get_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, W, b):
        self.W = [w.copy() for w in W]
        self.b = [x.copy() for x in b]

    # -- forward ----------------------------------------------------------
    def _forward(self, X):
        """Return (yhat, cache). Cache holds pre-activations for backward."""
        a = X
        zs, acts = [], [X]
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(a)
        out = a[:, 0]
        if self.output == "softplus":
            out = _softplus(out)
        return out, (zs, acts)

    def predict(self, X):
        return self._forward(np.asarray(X, float))[0]

    # -- loss and gradients ----------------------------------------------
    def loss(self, X, y):
        p = self.predict(X)
        mse = float(np.mean((p - y) ** 2))
        if self.l2:
            mse += self.l2 * sum(float(np.sum(w * w)) for w in self.W)
        return mse

    def _backward(self, X, y):
        n = len(X)
        yhat, (zs, acts) = self._forward(X)

        d = (2.0 * (yhat - y) / n)[:, None]
        if self.output == "softplus":
            # d(softplus)/dz = sigmoid(z)
            z_last = zs[-1]
            d = d * (1.0 / (1.0 + np.exp(-np.clip(z_last, -30, 30))))

        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if self.l2:
                gW[i] = gW[i] + 2.0 * self.l2 * self.W[i]
            if i > 0:
                d = (d @ self.W[i].T) * (zs[i - 1] > 0)
        return gW, gb

    # -- training ---------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)
        rng = np.random.default_rng(self.seed + 1)

        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0

        self.history_ = {"epoch": [], "train": [], "val": []}
        best = (np.inf, None, None, -1)

        for ep in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                gW, gb = self._backward(X[idx], y[idx])
                step += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                    mWh = mW[i] / (1 - b1 ** step)
                    vWh = vW[i] / (1 - b2 ** step)
                    mbh = mb[i] / (1 - b1 ** step)
                    vbh = vb[i] / (1 - b2 ** step)
                    self.W[i] -= self.lr * mWh / (np.sqrt(vWh) + eps)
                    self.b[i] -= self.lr * mbh / (np.sqrt(vbh) + eps)

            tr = self.loss(X, y)
            va = self.loss(X_val, y_val) if X_val is not None else np.nan
            self.history_["epoch"].append(ep + 1)
            self.history_["train"].append(tr)
            self.history_["val"].append(va)

            # keep the best validation weights - model selection, 5.10.2
            score = va if X_val is not None else tr
            if score < best[0] - 1e-12:
                W, B = self.get_params()
                best = (score, W, B, ep)
            elif ep - best[3] >= self.patience:
                if verbose:
                    print("      early stop at epoch %d (best %d)"
                          % (ep + 1, best[3] + 1))
                break

            if verbose and (ep + 1) % 50 == 0:
                print("      epoch %3d  train %.3e  val %.3e" % (ep + 1, tr, va))

        if best[1] is not None:
            self.set_params(best[1], best[2])
        self.best_epoch_ = best[3] + 1
        self.best_score_ = best[0]
        return self


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test(verbose=True):
    if verbose:
        print("=" * 78)
        print("neural_net.py  SELF-TEST")
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

    # ---- THE gradient check ----
    # Every analytic gradient against a central finite difference. A sign
    # error in backprop still trains, just worse, and is invisible otherwise.
    for out in ("linear", "softplus"):
        net = MLP(3, hidden=(5, 4), seed=SEED, output=out)
        X = rng.normal(size=(40, 3))
        y = rng.normal(size=40)
        gW, gb = net._backward(X, y)
        h = 1e-6
        worst = 0.0
        for i in range(len(net.W)):
            for arr, grad in ((net.W[i], gW[i]), (net.b[i], gb[i])):
                it = np.ndindex(arr.shape)
                for _ in range(min(25, arr.size)):
                    k = next(it)
                    orig = arr[k]
                    arr[k] = orig + h
                    lp = net.loss(X, y)
                    arr[k] = orig - h
                    lm = net.loss(X, y)
                    arr[k] = orig
                    num = (lp - lm) / (2 * h)
                    worst = max(worst, abs(num - grad[k])
                                / max(1.0, abs(num)))
        chk("analytic gradients match finite differences (%s)" % out,
            worst, 2e-6)

    # ---- the network can represent something a linear model cannot ----
    Xq = rng.uniform(-2, 2, (3000, 2))
    yq = Xq[:, 0] ** 2 + np.sin(2 * Xq[:, 1])
    Xv = rng.uniform(-2, 2, (1000, 2))
    yv = Xv[:, 0] ** 2 + np.sin(2 * Xv[:, 1])
    net = MLP(2, lr=5e-3, epochs=200, seed=SEED).fit(Xq, yq, Xv, yv)
    rm = float(np.sqrt(np.mean((net.predict(Xv) - yv) ** 2)))
    lin = np.linalg.lstsq(np.hstack([np.ones((len(Xq), 1)), Xq]), yq,
                          rcond=None)[0]
    rl = float(np.sqrt(np.mean(
        (np.hstack([np.ones((len(Xv), 1)), Xv]) @ lin - yv) ** 2)))
    if verbose:
        print("    network RMSE %.4f vs linear %.4f on a quadratic target"
              % (rm, rl))
    chk("network beats a linear fit on a nonlinear target",
        float(max(0.0, rm - 0.25 * rl)), 1e-12)

    # ---- training actually reduces the loss, monotonically on average ----
    h0 = net.history_["train"][0]
    hN = min(net.history_["train"])
    chk("training loss falls by at least 10x", float(max(0.0, hN - h0 / 10)),
        1e-12)
    chk("validation history recorded for every epoch",
        float(abs(len(net.history_["val"]) - len(net.history_["epoch"]))), 0)

    # ---- early stopping keeps the BEST weights, not the last ----
    best_val = min(net.history_["val"])
    chk("restored weights match the best validation loss",
        abs(net.loss(Xv, yv) - best_val) / best_val, 1e-9)

    # ---- determinism ----
    n2 = MLP(2, lr=5e-3, epochs=200, seed=SEED).fit(Xq, yq, Xv, yv)
    chk("same seed reproduces the network exactly",
        float(np.max(np.abs(net.predict(Xv) - n2.predict(Xv)))), 0.0)
    n3 = MLP(2, lr=5e-3, epochs=30, seed=SEED + 5).fit(Xq, yq, Xv, yv)
    chk("a different seed gives a different network",
        float(1.0 if np.max(np.abs(net.predict(Xv) - n3.predict(Xv)))
              < 1e-9 else 0.0), 0.0)

    # ---- He initialisation has the variance it claims ----
    big = MLP(100, hidden=(64, 32), seed=SEED)
    v = float(np.var(big.W[0]))
    chk("He initialisation variance is 2/fan_in",
        abs(v - 2.0 / 100) / (2.0 / 100), 0.15)
    chk("biases start at zero",
        float(max(np.abs(b).max() for b in big.b)), 0.0)

    # ---- softplus output cannot go negative ----
    sp = MLP(2, hidden=(8, 4), seed=SEED, output="softplus")
    chk("softplus output is strictly non-negative",
        float(max(0.0, -sp.predict(rng.normal(size=(500, 2))).min())), 0.0)

    # ---- architecture is the one the plan specifies ----
    d = MLP(4)
    chk("default architecture is p -> 32 -> 16 -> 1",
        float(d.sizes != [4, 32, 16, 1]), 0.0)

    n_fail = sum(1 for ok in checks if not ok)
    if verbose:
        print("-" * 78)
        print("  %d checks, %d failed" % (len(checks), n_fail))
        print("  ALL CHECKS PASSED" if n_fail == 0 else "  *** FAILURES ABOVE ***")
        print("=" * 78)
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
