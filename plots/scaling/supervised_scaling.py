"""Rank IC against training compute, one curve per ViT scale.

THE FIGURE: three panels, one per target, x = training FLOPs (log), y = rank
IC on the held-out month averaged over the 31 reported months, one curve per
rung of VIT_SCALES (tiny / small / base). It is the supervised specialist's
scaling curve and nothing else is on it.

WHAT A POINT IS. Every checkpoint of a run is a point, and every point is an
ANNEALED model: the run holds a stable rate and at each FLOPs target leaves
it for a cooldown, saves, and returns (checkpoint.anneal_steps, set by
scripts/sweeps/supervised_scaling.sh from half-decade FLOPs targets). A
target costs the same FLOPs in every month of a scale -- same batch, same
view length, same architecture -- so the solid curve is the mean over months
at each target EVERY month reached, with the s.e. across months as the
band. The older ladder (raw stable-phase rungs, root annealed at each
month's own length) is still drawn if the metrics carry it: its root goes
hollow at the mean of the per-month (x, y), joined by a dotted stub.

THE SHADE IS THE SCALE, THE COLOR IS THE TARGET. The target owns its color
everywhere in the paper (SERIES_STYLES); within a panel the three scales are
three lightnesses of it, tiny lightest, and carry their own marker so they
survive greyscale. The shared legend therefore keys on shade and marker in
neutral grey, and the panel supplies the hue.

THE STAR IS THE REPORTED MODEL. The paper's supervised specialist is this
same ViT-Small trained the recipe's way -- twelve passes over the trailing
six months -- which is not a FLOPs target but a per-month amount of compute,
measured by scripts/eval/default_model_flops.py and averaged in log. The star
sits ON the Small curve at that compute: it marks WHERE THE REPORTED MODEL
SPENDS, not what it scored, so the reader can see how much of the curve the
paper is standing on and how much is headroom.

THE READOUT IS THE HEAD: the model's own forecast, scored in-job on the eval
month's panel (user, 2026-09-19; the sweeps run POST_TRAIN_PROBE=0, so no
probe is scored at all). ``--readout probe`` still draws the ridge readout
for a tree that happens to carry it, to supervised_scaling_probe.png.

    uv run scripts/eval/collect_supervised_scaling.py
    uv run plots/scaling/supervised_scaling.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_PLOTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLOTS))
from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, add_bottom_legend,
    apply_style, save_figure, set_two_decimal_yticks)

METRICS_JSON = _PLOTS / "metrics/supervised_scaling.json"
DEFAULT_MODEL_JSON = _PLOTS / "metrics/supervised_default_model.json"

PANELS = (
    ("return_900", "Return", "supervised_return"),
    ("volatility_change_900", "Volatility Change", "supervised_vol"),
    ("spread_change_900", "Spread Change", "supervised_spread"),
)
# Drawn in this order, so the reported model's curve sits on top of tiny's.
SCALES = (
    # key,    label,        marker, lightness shift (+ toward white, - toward black)
    ("tiny",  "ViT-Tiny",   "v",    +0.45),
    ("small", "ViT-Small",  "o",     0.0),
    ("base",  "ViT-Base",   "^",    -0.45),
)
LEGEND_GREY = "#555555"


def shade(color: str, shift: float) -> tuple:
    """``color`` moved toward white (shift > 0) or black (shift < 0)."""
    r, g, b = mcolors.to_rgb(color)
    if shift >= 0:
        return tuple(c + (1.0 - c) * shift for c in (r, g, b))
    return tuple(c * (1.0 + shift) for c in (r, g, b))


def default_model(path: Path) -> dict | None:
    """The reported model's compute, or None if it has not been measured."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if d.get("flops") else None


def on_curve(xs, mu, x: float) -> float | None:
    """The curve's height at ``x``, interpolated in log compute.

    None when ``x`` is off the ends of the curve: the star says where the
    reported model spends, and a star hanging past the last rung would say
    something the sweep has not measured yet.
    """
    if len(xs) < 2 or not (xs[0] <= x <= xs[-1]):
        return None
    return float(np.interp(np.log(x), np.log(xs), mu))


def load(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def series(rows: list[dict], task: str, scale: str, readout: str,
           min_step: int = 0):
    """(xs, mean, se, months, endpoint) for one curve, or None.

    ``xs`` are the FLOPs of the ladder rungs EVERY month of this arm reached
    (a rung is a step, and a step is one x for the whole scale); ``endpoint``
    is the mean (x, y) of the annealed run roots, one per month.
    """
    key = f"ic_{readout}"
    pts = [r for r in rows if r["task"] == task and r["scale"] == scale
           and r.get(key) is not None]
    if not pts:
        return None
    months = sorted({r["eval_month"] for r in pts})
    by_step: dict[int, dict[str, tuple[float, float]]] = {}
    ends: dict[str, tuple[float, float]] = {}
    for r in pts:
        if r.get("endpoint", r["annealed"]):
            ends[r["eval_month"]] = (float(r["flops"]), float(r[key]))
        elif int(r["step"]) >= min_step:
            by_step.setdefault(int(r["step"]), {})[r["eval_month"]] = (
                float(r["flops"]), float(r[key]))
    steps = sorted(s for s, d in by_step.items() if len(d) == len(months))
    xs, mu, se = [], [], []
    for s in steps:
        fl = np.array([by_step[s][m][0] for m in months])
        # One x per rung: the same step is the same compute in every month.
        # A spread here means two months trained different batches or view
        # lengths under one project name, which is a different sweep.
        assert np.allclose(fl, fl[0], rtol=1e-6), (task, scale, s, fl)
        ys = np.array([by_step[s][m][1] for m in months])
        xs.append(float(fl[0]))
        mu.append(float(ys.mean()))
        se.append(float(ys.std(ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0)
    endpoint = None
    if ends and len(ends) == len(months):
        endpoint = (float(np.mean([v[0] for v in ends.values()])),
                    float(np.mean([v[1] for v in ends.values()])))
    if not xs and endpoint is None:
        return None
    return np.array(xs), np.array(mu), np.array(se), months, endpoint


def params_label(rows: list[dict], scale: str) -> str:
    p = {r["params_backbone"] for r in rows if r["scale"] == scale}
    if not p:
        return ""
    n = max(p) / 1e6
    return f" ({n:.0f}M)" if n >= 10 else f" ({n:.1f}M)"


def draw(rows: list[dict], readout: str, out: Path, min_step: int = 0,
         min_months: int = 1, default: dict | None = None) -> bool:
    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.36))
    drew: set[str] = set()
    drew_end = drew_star = False
    coverage: dict[tuple, int] = {}
    for ax, (task, title, style_key) in zip(axes, PANELS):
        color = SERIES_STYLES[style_key]["color"]
        for key, label, marker, shift in SCALES:
            got = series(rows, task, key, readout, min_step)
            if got is None:
                continue
            xs, mu, se, months, end = got
            coverage[(task, key)] = len(months)
            if len(months) < min_months:
                continue
            c = shade(color, shift)
            drew.add(key)
            if len(xs):
                ax.plot(xs, mu, marker=marker, ls="-", color=c, ms=2.6, lw=1.1,
                        zorder=3, label=label)
                ax.fill_between(xs, mu - se, mu + se, color=c, alpha=0.16, lw=0)
            if (default is not None and default["scale"] == key and len(xs)):
                y = on_curve(xs, mu, float(default["flops"]))
                if y is not None:
                    drew_star = True
                    ax.plot([default["flops"]], [y], marker="*", ms=9,
                            color=c, mec="white", mew=0.6, ls="none", zorder=5)
            if end is not None:
                drew_end = True
                if len(xs):
                    ax.plot([xs[-1], end[0]], [mu[-1], end[1]], ":", color=c,
                            lw=0.9, alpha=0.7, zorder=3)
                ax.plot([end[0]], [end[1]], marker=marker, ms=5, mfc="none",
                        mec=c, mew=1.1, ls="none", zorder=4)
        ax.set_title(title, pad=4)
        ax.set_xscale("log")
        ax.grid(False)
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
        set_two_decimal_yticks(ax, nbins=5)
    if not drew:
        plt.close(fig)
        return False
    axes[0].set_ylabel("Rank IC")
    axes[1].set_xlabel("Training FLOPs")

    handles, labels = [], []
    for key, label, marker, shift in SCALES:
        if key not in drew:
            continue
        handles.append(Line2D([], [], ls="-", lw=1.1, marker=marker, ms=2.6,
                              color=shade(LEGEND_GREY, shift)))
        labels.append(label + params_label(rows, key))
    if drew_star:
        handles.append(Line2D([], [], marker="*", ms=9, ls="none",
                              color=LEGEND_GREY, mec="white", mew=0.6))
        labels.append(default.get("label", "Reported Model"))
    if drew_end:
        handles.append(Line2D([], [], marker="o", ms=5, mfc="none",
                              mec=LEGEND_GREY, mew=1.1, ls="none"))
        labels.append("Annealed Endpoint")
    fig.tight_layout()
    add_bottom_legend(fig, handles, labels, ncol=len(labels),
                      columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, out)
    plt.close(fig)
    for (task, key), n in sorted(coverage.items()):
        print(f"  {readout:5s} {key:6s} {task:24s} {n:2d} month(s)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, default=METRICS_JSON)
    p.add_argument("--readout", choices=("head", "probe"), default="head")
    p.add_argument("--min-step", type=int, default=0,
                   help="drop ladder rungs below this step (e.g. the warmup)")
    p.add_argument("--min-months", type=int, default=1,
                   help="draw a scale only once this many months have landed")
    p.add_argument("--default-model", type=Path, default=DEFAULT_MODEL_JSON,
                   help="the reported model's measured compute; the star")
    p.add_argument("--no-default-model", action="store_true",
                   help="leave the star off")
    p.add_argument("--out", type=Path, default=None,
                   help="output stem; the probe readout gets a _probe suffix")
    a = p.parse_args()
    rows = load(a.metrics)
    out = a.out or Path(__file__).with_suffix(".png")
    if a.readout == "probe" and a.out is None:
        out = out.with_name(out.stem + "_probe" + out.suffix)
    star = None if a.no_default_model else default_model(a.default_model)
    if star is not None:
        print(f"  reported model at {star['flops']:.3e} FLOPs "
              f"({star['n_months']} month(s))")
    if not draw(rows, a.readout, out, a.min_step, a.min_months, star):
        raise SystemExit(f"no {a.readout} readout to draw in {a.metrics}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
