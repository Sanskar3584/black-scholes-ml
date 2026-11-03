// Mid-semester 4-slide progress report.
// Build: node build_slides.js   (run from a dir containing figures/)
const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = process.env.FIGDIR || "figures";
const OUT = process.env.OUTFILE || "Sanskar_MidSem_Presentation.pptx";

const NAVY = "1F3864";
const BLUE = "2E5FA3";
const GREY = "595959";
const RULE = "BFCBDD";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";              // 13.333 x 7.5 in
pres.author = "Sanskar Joshi";
pres.title  = "Black-Scholes and Market Risk Analysis using AI/ML";

const W = 13.333;
const M = 0.55;                            // side margin
const CW = W - 2 * M;                      // content width

// ---------------------------------------------------------------- helpers

function header(slide, n, title, sub) {
  slide.addText(title, {
    x: M, y: 0.28, w: CW - 0.9, h: 0.46, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 25, bold: true, color: NAVY, valign: "middle",
  });
  slide.addText(`${n} / 4`, {
    x: W - M - 0.85, y: 0.28, w: 0.85, h: 0.46, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, color: GREY, align: "right", valign: "middle",
  });
  slide.addShape(pres.ShapeType.line, {
    x: M, y: 0.80, w: CW, h: 0, line: { color: NAVY, width: 1.6 },
  });
  if (sub) {
    slide.addText(sub, {
      x: M, y: 0.86, w: CW, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, italic: true, color: GREY, valign: "middle",
    });
  }
}

// A horizontal chain: A -> B -> C ...
function chain(slide, items, y, fontSize = 11.5) {
  const gap = 0.16;
  const arrowW = 0.30;
  const n = items.length;
  const totalArrow = arrowW * (n - 1) + gap * 2 * (n - 1);
  const boxW = (CW - totalArrow) / n;
  let x = M;
  items.forEach((txt, i) => {
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w: boxW, h: 0.52, rectRadius: 0.06,
      fill: { color: "EAEFF7" }, line: { color: RULE, width: 0.75 },
    });
    slide.addText(txt, {
      x, y, w: boxW, h: 0.52, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize, color: NAVY, bold: true,
      align: "center", valign: "middle",
    });
    x += boxW;
    if (i < n - 1) {
      slide.addText("→", {
        x: x + gap, y, w: arrowW, h: 0.52, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 15, color: BLUE, align: "center", valign: "middle",
      });
      x += arrowW + gap * 2;
    }
  });
}

function bullets(slide, items, opt) {
  slide.addText(
    items.map((t, i) => ({
      text: t.t !== undefined ? t.t : t,
      options: {
        bullet: true, breakLine: i < items.length - 1,
        bold: t.b || false, color: t.c || "000000", paraSpaceAfter: 5,
      },
    })),
    Object.assign({
      isTextBox: true, margin: 0, fontFace: BODY, fontSize: 12.5,
      color: "000000", valign: "top",
    }, opt)
  );
}

// key-value strip of result chips
function chips(slide, data, y, w = CW, x0 = M, h = 0.62) {
  const gap = 0.14;
  const bw = (w - gap * (data.length - 1)) / data.length;
  data.forEach((d, i) => {
    const x = x0 + i * (bw + gap);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w: bw, h, rectRadius: 0.05,
      fill: { color: "F4F7FB" }, line: { color: RULE, width: 0.75 },
    });
    slide.addText(d.v, {
      x, y: y + 0.04, w: bw, h: 0.30, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, bold: true, color: NAVY,
      align: "center", valign: "middle",
    });
    slide.addText(d.k, {
      x, y: y + 0.32, w: bw, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9.5, color: GREY, align: "center", valign: "middle",
    });
  });
}

const FOOTER_Y = 7.00;
const CAP_H = 0.30;

// Size the figure by the vertical space available, not by a guessed width, so it
// can never run past the footer. All panels are 1859 x 640 (aspect 2.90).
function figure(slide, file, y, caption, maxBottom = FOOTER_Y - 0.06) {
  const avail = maxBottom - y - (caption ? CAP_H : 0);
  let h = avail;
  let w = h * 2.90;
  if (w > CW) { w = CW; h = w / 2.90; }    // never wider than the text column
  if (h <= 0.2) throw new Error(`figure ${file}: no vertical room (avail ${avail.toFixed(2)}")`);
  const x = (W - w) / 2;
  slide.addImage({ path: path.join(FIG, file), x, y, w, h });
  if (caption) {
    slide.addText(caption, {
      x: M, y: y + h + 0.02, w: CW, h: CAP_H, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 9.5, italic: true, color: GREY, align: "center",
      valign: "top",
    });
  }
  return y + h;
}

function footer(slide, text) {
  slide.addShape(pres.ShapeType.line, {
    x: M, y: 7.00, w: CW, h: 0, line: { color: RULE, width: 1 },
  });
  slide.addText(text, {
    x: M, y: 7.04, w: CW, h: 0.34, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, color: NAVY, valign: "middle",
  });
}

// ============================================================ SLIDE 1

let s = pres.addSlide();
header(s, 1,
  "Objective and Week 1: Market Data, Returns and GBM",
  "Build from mathematical-finance foundations toward Black-Scholes, numerical validation, machine learning and market-risk analysis.");

chain(s, ["NIFTY 50 data\n2015-2024",
          "Simple and\nlog returns",
          "Statistical\nanalysis",
          "Brownian motion\nand GBM",
          "Model vs data\ncomparison"], 1.20, 10.5);

bullets(s, [
  { t: "2,458 daily log returns. Annualised volatility σ ≈ 0.167, used to calibrate GBM.", b: false },
  "Returns are NOT normal: strong negative skewness, large excess kurtosis, heavy tails.",
  "Extreme events far exceed the model: 10 days beyond 5σ where GBM produced none.",
  "Volatility clustering present: |return| autocorrelation 0.31 at lag 1, still 0.09 at lag 30.",
], { x: M, y: 1.82, w: CW, h: 1.00, fontSize: 11.5 });

chips(s, [
  { v: "−1.41",   k: "skewness" },
  { v: "20.41",        k: "excess kurtosis" },
  { v: "0.000",        k: "Jarque-Bera p" },
  { v: "15 vs 0",      k: "days beyond 4σ  (NIFTY vs GBM)" },
  { v: "10 vs 0",      k: "days beyond 5σ  (NIFTY vs GBM)" },
  { v: "0.167",        k: "annualised σ" },
], 2.86, CW, M, 0.58);

figure(s, "w02_gbm_vs_nifty_histogram.png", 3.54,
  "Empirical NIFTY 50 log-return distribution against GBM calibrated to the same mean and volatility. Linear scale (left), log density (right).");

footer(s, "GBM reproduces mean and volatility by construction, but not the tails, the skewness or the clustering.");
s.addNotes("Week 1 established what the data actually looks like. The key point: GBM was calibrated to the SAME mean and volatility, so matching those is not evidence. Everything it fails on - skewness, kurtosis, tail frequency, clustering - was never targeted. The 15-vs-0 and 10-vs-0 rows are the headline: in matched samples the market delivered fifteen 4-sigma days and ten 5-sigma days; the model produced none.");

// ============================================================ SLIDE 2

s = pres.addSlide();
header(s, 2,
  "Week 2: From Itô's Lemma to the Black-Scholes PDE",
  "The derivation, and the numerical experiments that verify each step.");

chain(s, ["GBM", "Quadratic\nvariation", "Itô's\nlemma", "Derivative\ndynamics",
          "Delta\nhedge", "Elimination\nof dW", "No-\narbitrage", "Black-Scholes\nPDE"], 1.18, 10);

s.addShape(pres.ShapeType.roundRect, {
  x: M, y: 1.80, w: CW, h: 0.54, rectRadius: 0.06,
  fill: { color: "F4F7FB" }, line: { color: NAVY, width: 1 },
});
s.addText("∂V/∂t  +  ½ σ²S² ∂²V/∂S²  +  rS ∂V/∂S  −  rV  =  0", {
  x: M, y: 1.80, w: CW, h: 0.54, isTextBox: true, margin: 0,
  fontFace: "Cambria Math", fontSize: 18, bold: true, color: NAVY,
  align: "center", valign: "middle",
});

bullets(s, [
  { t: "Why μ disappears: it enters dV through V_S dS and enters Δ dS through the same channel. Setting Δ = V_S makes them identical, so the drift and the dW term cancel together — removing the risk and removing the drift are one operation.", b: false },
  "r appears in its place because the hedged portfolio is locally riskless, and a riskless portfolio must earn the risk-free rate under no-arbitrage.",
  "Itô verified numerically: for f(W) = W² the Itô expression reproduces W_T² to 1.2e−14, while ordinary calculus is wrong by exactly T and does not improve with refinement.",
], { x: M, y: 2.44, w: CW, h: 1.05, fontSize: 11.5 });

chips(s, [
  { v: "0.018",     k: "hedged mean P&L spread across μ" },
  { v: "26.56",     k: "unhedged mean P&L spread across μ" },
  { v: "97-98%",    k: "dispersion removed by hedging" },
  { v: "1.2e−14", k: "Itô identity max error" },
], 3.56, CW, M, 0.58);

figure(s, "w03_delta_hedge_mu_invariance.png", 4.24,
  "Short ATM call, delta-hedged 252 times, identical shocks across μ = 0.00, 0.15, 0.30. Unhedged P&L shifts with μ (left); hedged P&L does not (right).");

footer(s, "μ varies, the hedged distribution does not. That invariance is what 'the drift disappears' means operationally.");
s.addNotes("The professor asked specifically that I explain why the second-order term survives, how delta hedging eliminates dW, and why mu disappears. Second-order survives because (dW)^2 = dt - quadratic variation, measured in Week 1. Delta hedging kills dW because the derivative and the underlying are driven by the SAME Brownian motion. And mu goes because it rides on the same dS channel that the hedge neutralises. The right-hand figure is the proof: three different physical drifts, same hedged distribution.");

// ============================================================ SLIDE 3

s = pres.addSlide();
header(s, 3,
  "Implementation and Independent Verification",
  "European call and put implemented in black_scholes.py, then checked by three independent routes.");

bullets(s, [
  { t: "Closed-form European call and put implemented, with the T → 0 and σ → 0 limits handled explicitly rather than left singular.", b: false },
  "PDE substitution: differentiating the closed-form price numerically and substituting into the PDE gives a maximum residual of 3.34e−5 against prices of order 16 — finite-difference truncation error, nothing more.",
  "Put-call parity verified over a full grid of S, K, T, r and σ. Parity is model-free, so this tests the code, not the model.",
  "Self-test over 20,000 random parameter sets: non-negativity, arbitrage upper and lower bounds, and both limiting cases. All passed.",
  { t: "Floating-point limits were identified and documented, not hidden: at σ = 0.001 the density φ(d₁) underflows to exactly zero (d₁ ≈ −173), so vega is positive mathematically but not representable. The test was restated rather than loosened.", b: false },
], { x: M, y: 1.24, w: CW, h: 2.00, fontSize: 11.5 });

chips(s, [
  { v: "3.34e−5",  k: "max PDE residual" },
  { v: "4.26e−14", k: "max put-call parity error" },
  { v: "13,500",       k: "parity grid points" },
  { v: "20,000",       k: "random sets in module self-test" },
], 3.34, CW, M, 0.58);

figure(s, "w03_sensitivity_S.png", 4.02,
  "Parameter sensitivity in S: call price against intrinsic value and the lower bound S − Ke^(−rT) (left); ∂C/∂S rising monotonically from 0 to 1 (right). Monotonicity in K, σ, T and r was checked in the same way.");

footer(s, "The classical Black-Scholes benchmark has been derived, implemented and independently verified.");
s.addNotes("Three independent checks, each testing something different. The PDE residual tests whether the formula really solves the equation I derived - it is the only check that validates the derivation and the code simultaneously. Put-call parity is model-free, so it tests only the implementation. The self-test covers bounds and limits. On the floating-point point: the vega check failed at first, and the honest cause was underflow, not a bug - that distinction is worth making if asked.");

// ============================================================ SLIDE 4

s = pres.addSlide();
header(s, 4,
  "Present Status, Key Learning and Next Stage",
  "Classical foundation complete; the project now moves from pricing to sensitivities.");

s.addText("Completed", {
  x: M, y: 1.22, w: 5.9, h: 0.3, isTextBox: true, margin: 0,
  fontFace: BODY, fontSize: 15, bold: true, color: NAVY,
});
bullets(s, [
  "Market-data analysis: returns, distributions, normality testing, tail behaviour",
  "Brownian motion and geometric Brownian motion, with model-versus-data comparison",
  "Itô's lemma, derived and verified numerically",
  "Delta hedging and elimination of the stochastic term",
  "Black-Scholes PDE, derived from no-arbitrage",
  "European call and put implementation with independent verification",
], { x: M, y: 1.56, w: 5.9, h: 2.10, fontSize: 11.5 });

s.addShape(pres.ShapeType.roundRect, {
  x: 6.85, y: 1.22, w: CW - 6.30, h: 2.44, rectRadius: 0.06,
  fill: { color: "F4F7FB" }, line: { color: NAVY, width: 1 },
});
s.addText("Key modelling insight", {
  x: 7.05, y: 1.34, w: CW - 6.70, h: 0.30, isTextBox: true, margin: 0,
  fontFace: BODY, fontSize: 15, bold: true, color: NAVY,
});
s.addText(
  "Mathematical correctness under the Black-Scholes assumptions is a different thing " +
  "from empirical adequacy for real market returns.\n\n" +
  "The derivation is exact given its assumptions. Those assumptions — continuous " +
  "diffusion, constant volatility, Gaussian innovations, independent increments — " +
  "were each measured against NIFTY 50 data and each is contradicted.\n\n" +
  "Recording that gap precisely is what makes the model a benchmark rather than a " +
  "belief, and it defines what the later AI/ML stage is asked to close.",
  { x: 7.05, y: 1.70, w: CW - 6.70, h: 1.84, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: "000000", valign: "top" });

s.addText("Next sequence", {
  x: M, y: 3.84, w: CW, h: 0.3, isTextBox: true, margin: 0,
  fontFace: BODY, fontSize: 15, bold: true, color: NAVY,
});
chain(s, ["Greeks", "Finite-difference\nverification", "Implied\nvolatility",
          "Binomial\ntree", "Monte Carlo", "ML / hybrid\npricing",
          "Market-risk\nanalysis"], 4.20, 10);

s.addShape(pres.ShapeType.roundRect, {
  x: M, y: 5.00, w: CW, h: 0.70, rectRadius: 0.06,
  fill: { color: "EAEFF7" }, line: { color: BLUE, width: 1 },
});
s.addText("Immediately after the mid-semester examinations: Week 3 — Greeks and Implied Volatility", {
  x: M, y: 5.00, w: CW, h: 0.70, isTextBox: true, margin: 0,
  fontFace: BODY, fontSize: 14, bold: true, color: NAVY,
  align: "center", valign: "middle",
});

bullets(s, [
  "Code, figures, reports, cached data and supervisor plans are organised in the project repository with a fixed random seed (20260825), so every reported number reproduces exactly.",
], { x: M, y: 5.88, w: CW, h: 0.50, fontSize: 11.5 });

footer(s, "Milestone: classical benchmark established. Next phase moves from pricing verification to sensitivities and implied volatility.");
s.addNotes("Close on the distinction between mathematical correctness and empirical adequacy - that is the thread running through the whole project and the reason the ML stage is framed as 'where does ML help' rather than 'is ML better'. Mention reproducibility briefly: fixed seed, cached data, everything in the repo.");

// ============================================================ write

pres.writeFile({ fileName: OUT }).then(f => console.log("written:", f));
