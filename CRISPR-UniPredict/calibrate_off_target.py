"""
Post-hoc probability calibration for the dedicated off-target model.

Motivation
----------
The off-target classifier is trained with **focal loss** (alpha/gamma) to cope
with the ~93:1 (1.06% positive) class imbalance. Focal loss deliberately
re-shapes the loss landscape toward hard examples, which is excellent for
*ranking* (AUROC 0.8753) but is known to leave the raw sigmoid outputs
**miscalibrated** -- the number the model emits is not a trustworthy probability
of an off-target event. For a safety-critical screen that is a real problem:
a practitioner cannot threshold on "flag if P(off-target) > 0.3" if 0.3 does not
mean 30%.

Contribution
------------
We recover calibrated off-target risk estimates *post hoc*, without retraining
and without touching the ranking performance:

  * Calibrators are FIT on the held-out validation split and EVALUATED on the
    test split (no leakage).
  * Three standard methods: temperature scaling, Platt scaling, isotonic
    regression. All are monotonic, so AUROC / AUPRC are provably unchanged --
    we only fix the probability scale.
  * We report calibration error under extreme imbalance correctly (quantile
    binning in addition to uniform), plus proper scoring rules (Brier, NLL).
  * We provide an operating-point table: thresholds (in calibrated-probability
    space) for target false-positive rates, with the recall/precision they buy.

Run
---
    python calibrate_off_target.py                 # full split (cached after 1st run)
    python calibrate_off_target.py --max-rows 8000 # quick smoke test
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from models.off_target_dedicated import OffTargetDedicatedModel
from train_off_target_dedicated import OffTargetDataset, load_split

EPS = 1e-7


# ---------------------------------------------------------------------------
# Inference (captures logits, not just sigmoid; cached to .npz)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_logits(model, df, seq_len, device, batch_size=2048):
    ds = OffTargetDataset(df, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, ys = [], []
    model.eval()
    for i, (pair_oh, pair_lb, y) in enumerate(dl):
        l = model(pair_oh.to(device), pair_lb.to(device)).view(-1)
        logits.append(l.float().cpu().numpy())
        ys.append(y.numpy())
        if (i + 1) % 25 == 0:
            print(f"    batch {i + 1}/{len(dl)}", flush=True)
    return np.concatenate(logits), np.concatenate(ys).astype(int)


def cached_logits(model, split, csv, seq_len, device, cache_dir, max_rows):
    tag = f"{split}_{'full' if max_rows is None else str(max_rows)}"
    cache = cache_dir / f"preds_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  [{split}] loaded cached preds: {cache.name}  (n={len(d['y']):,})")
        return d["logits"], d["y"]
    print(f"  [{split}] running inference on {csv} ...")
    df = load_split(str(csv))
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)
    logits, y = get_logits(model, df, seq_len, device)
    np.savez_compressed(cache, logits=logits, y=y)
    print(f"  [{split}] n={len(y):,}  pos={int(y.sum()):,} ({y.mean():.3%})  cached -> {cache.name}")
    return logits, y


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def nll(p, y):
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def calibration_error(p, y, n_bins=15, strategy="uniform"):
    """Expected & maximum calibration error. Returns (ece, mce, bin_table).

    strategy='uniform'  : fixed-width [0,1] bins.
    strategy='quantile' : equal-count-by-rank bins -- robust when predictions are
                          heavily concentrated near 0 (1% prevalence), where naive
                          quantile *edges* collapse to a couple of bins.
    """
    p = _clip(p)
    n = len(p)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        groups = []
        for k in range(n_bins):
            lo, hi = edges[k], edges[k + 1]
            m = (p >= lo) & (p < hi) if k < n_bins - 1 else (p >= lo) & (p <= hi)
            groups.append((np.where(m)[0], lo, hi))
    else:  # equal-count by rank
        order = np.argsort(p, kind="mergesort")
        groups = [(idx, float(p[idx].min()), float(p[idx].max()))
                  for idx in np.array_split(order, n_bins) if len(idx) > 0]

    ece, mce, table = 0.0, 0.0, []
    for idx, lo, hi in groups:
        if len(idx) == 0:
            continue
        conf, acc, w = p[idx].mean(), y[idx].mean(), len(idx) / n
        gap = abs(conf - acc)
        ece += w * gap
        mce = max(mce, gap)
        table.append({"lo": float(lo), "hi": float(hi), "n": int(len(idx)),
                      "mean_pred": float(conf), "obs_freq": float(acc)})
    return float(ece), float(mce), table


def all_metrics(p, y):
    eceu, mceu, _ = calibration_error(p, y, strategy="uniform")
    eceq, mceq, _ = calibration_error(p, y, strategy="quantile")
    return {
        "ECE_uniform": eceu, "MCE_uniform": mceu,
        "ECE_quantile": eceq, "MCE_quantile": mceq,
        "Brier": brier(p, y), "NLL": nll(p, y),
        "AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
    }


# ---------------------------------------------------------------------------
# Calibrators (fit on val, all monotonic -> AUROC preserved)
# ---------------------------------------------------------------------------

def fit_temperature(logits, y):
    def obj(T):
        return nll(1 / (1 + np.exp(-logits / T)), y)
    res = minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def fit_platt(logits, y):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    lr.fit(logits.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def operating_points(p_val, y_val, p_test, y_test, target_fprs):
    """Threshold chosen on VAL for each target FPR, evaluated on TEST."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    rows = []
    for f in target_fprs:
        i = int(np.argmin(np.abs(fpr - f)))
        t = float(thr[i])
        flag = p_test >= t
        tp = int((flag & (y_test == 1)).sum())
        fp = int((flag & (y_test == 0)).sum())
        fn = int((~flag & (y_test == 1)).sum())
        npos = int((y_test == 1).sum())
        nneg = int((y_test == 0).sum())
        rows.append({
            "target_fpr": f,
            "threshold_calibrated_prob": t,
            "test_fpr": fp / max(nneg, 1),
            "test_recall_tpr": tp / max(npos, 1),
            "test_precision": tp / max(tp + fp, 1),
            "flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def _logit_bin_points(p, y, n=12, min_count=20):
    """Reliability points binned in LOGIT space (spreads the p~=0 region that
    dominates under 1% prevalence). Returns (mean_pred, obs_freq) per bin."""
    p = _clip(p)
    z = np.log(p / (1 - p))
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    edges = np.linspace(lo, hi, n + 1)
    xs, ys = [], []
    for k in range(n):
        m = (z >= edges[k]) & (z < edges[k + 1]) if k < n - 1 else (z >= edges[k]) & (z <= edges[k + 1])
        if m.sum() < min_count:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def reliability_fig(p_raw, p_cal, y, out_png, best_name, m_raw, m_cal):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    # Reliability overlay (log-x so the p~=0 mass spreads out)
    xr, yr = _logit_bin_points(p_raw, y)
    xc, yc = _logit_bin_points(p_cal, y)
    diag = np.logspace(-6, 0, 100)
    ax[0].plot(diag, diag, "--", color="gray", lw=1, label="perfect calibration")
    ax[0].plot(xr, yr, "o-", color="#d62728", ms=5,
               label=f"raw (ECE={m_raw['ECE_uniform']:.4f}, Brier={m_raw['Brier']:.4f})")
    ax[0].plot(xc, yc, "s-", color="#2ca02c", ms=5,
               label=f"{best_name} (ECE={m_cal['ECE_uniform']:.4f}, Brier={m_cal['Brier']:.4f})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(1e-6, 1.1); ax[0].set_ylim(1e-4, 1.1)
    ax[0].set_title("Reliability (logit-binned, log-log)", fontsize=11)
    ax[0].set_xlabel("mean predicted P(off-target)")
    ax[0].set_ylabel("observed frequency")
    ax[0].legend(loc="upper left", fontsize=8); ax[0].grid(alpha=0.25, which="both")

    ax[1].hist(p_raw, bins=60, alpha=0.6, label="raw", color="#d62728")
    ax[1].hist(p_cal, bins=60, alpha=0.6, label="calibrated", color="#2ca02c")
    ax[1].set_yscale("log")
    ax[1].set_title("Predicted-probability distribution", fontsize=11)
    ax[1].set_xlabel("P(off-target)"); ax[1].set_ylabel("count (log)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    fig.suptitle(f"Off-target probability calibration  (fit on val, eval on test; "
                 f"raw→{best_name})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"  saved figure -> {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(BASE / "models/checkpoints/off_target_dedicated/best_model.pt"))
    ap.add_argument("--val", default=str(BASE / "data/processed/combined/val.csv"))
    ap.add_argument("--test", default=str(BASE / "data/processed/combined/test.csv"))
    ap.add_argument("--out-dir", default=str(BASE / "reports/off_target_calibration"))
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    seq_len = ck.get("cfg", {}).get("seq_len", 23)
    model = OffTargetDedicatedModel(seq_len=seq_len).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"Loaded {args.ckpt} (epoch {ck.get('epoch')})")

    mr = args.max_rows
    val_logits, y_val = cached_logits(model, "val", args.val, seq_len, device, out_dir, mr)
    test_logits, y_test = cached_logits(model, "test", args.test, seq_len, device, out_dir, mr)

    sig = lambda l: 1 / (1 + np.exp(-l))
    p_val_raw, p_test_raw = sig(val_logits), sig(test_logits)

    # --- fit on val ---
    T = fit_temperature(val_logits, y_val)
    a, b = fit_platt(val_logits, y_val)
    iso = fit_isotonic(p_val_raw, y_val)

    methods = {
        "raw": p_test_raw,
        "temperature": sig(test_logits / T),
        "platt": sig(a * test_logits + b),
        "isotonic": _clip(iso.transform(p_test_raw)),
    }
    metrics = {name: all_metrics(p, y_test) for name, p in methods.items()}

    # best = lowest quantile-ECE among the calibrated methods
    best = min([m for m in methods if m != "raw"],
               key=lambda m: metrics[m]["ECE_quantile"])

    val_cal = {
        "temperature": sig(val_logits / T),
        "platt": sig(a * val_logits + b),
        "isotonic": _clip(iso.transform(p_val_raw)),
    }
    ops = operating_points(val_cal[best], y_val, methods[best], y_test,
                           target_fprs=[0.001, 0.005, 0.01, 0.05, 0.10])

    reliability_fig(methods["raw"], methods[best], y_test,
                    out_dir / "reliability_diagram.png", best,
                    metrics["raw"], metrics[best])

    results = {
        "setup": {
            "fit_split": "val", "eval_split": "test",
            "n_val": int(len(y_val)), "n_test": int(len(y_test)),
            "test_pos_rate": float(y_test.mean()), "max_rows": mr,
        },
        "fitted_params": {"temperature_T": T, "platt_a": a, "platt_b": b},
        "metrics": metrics,
        "best_method": best,
        "improvement": {
            "ECE_quantile": {"raw": metrics["raw"]["ECE_quantile"],
                             best: metrics[best]["ECE_quantile"],
                             "rel_reduction": 1 - metrics[best]["ECE_quantile"] / max(metrics["raw"]["ECE_quantile"], EPS)},
            "Brier": {"raw": metrics["raw"]["Brier"], best: metrics[best]["Brier"]},
            "NLL": {"raw": metrics["raw"]["NLL"], best: metrics[best]["NLL"]},
            "AUROC_preserved": {"raw": metrics["raw"]["AUROC"], best: metrics[best]["AUROC"]},
        },
        "operating_points": ops,
    }
    with open(out_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- console summary ---
    print("\n" + "=" * 74)
    print("OFF-TARGET CALIBRATION  (fit=val, eval=test)")
    print("=" * 74)
    hdr = f"{'method':<12}{'ECE_q':>9}{'ECE_u':>9}{'Brier':>10}{'NLL':>9}{'AUROC':>9}{'AUPRC':>9}"
    print(hdr); print("-" * len(hdr))
    for name in ["raw", "temperature", "platt", "isotonic"]:
        m = metrics[name]
        star = "  <-- best" if name == best else ""
        print(f"{name:<12}{m['ECE_quantile']:>9.4f}{m['ECE_uniform']:>9.4f}"
              f"{m['Brier']:>10.5f}{m['NLL']:>9.4f}{m['AUROC']:>9.4f}{m['AUPRC']:>9.4f}{star}")
    print(f"\nTemperature T = {T:.3f}  (T>1 => raw outputs over-confident)")
    print(f"\nOperating points (threshold picked on val, evaluated on test, method={best}):")
    print(f"{'tgtFPR':>8}{'thr(prob)':>12}{'testFPR':>10}{'recall':>9}{'precision':>11}{'flagged':>9}")
    for r in ops:
        print(f"{r['target_fpr']:>8.3f}{r['threshold_calibrated_prob']:>12.4f}"
              f"{r['test_fpr']:>10.4f}{r['test_recall_tpr']:>9.4f}"
              f"{r['test_precision']:>11.4f}{r['flagged']:>9d}")
    print(f"\nSaved: {out_dir/'calibration_results.json'}")


if __name__ == "__main__":
    main()
