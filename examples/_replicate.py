"""Shared steps for the replication examples: data, train, score, compare.

Each example trains one released method from scratch on the released data for
one evaluation month, scores it with the paper's scorer, scores the RELEASED
encoder for the same month with the same scorer on the same machine, and checks
the two agree. Scoring both here, rather than comparing against a number in
the paper, means a difference in the scoring environment cannot pass for a
difference in the model.

Everything lands under --work (default runs/replicate/): the downloaded data,
the target tables, checkpoints, embeddings and a results json per example.
Every step skips work that is already on disk, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = "fin-ai-lab"
DENSE = ("Market-1T-1Hz-2019H2-2020-dense", "1Hz_mosaic_mnth")
SPARSE = ("Market-1T-1Hz-2019H2-2020-sparse", "1Hz_mosaic_mnth_sparse")
DAYSTORE = ("Market-1T-1Hz-2019H2-2020-daystore", "1Hz_daystore")
HOLIDAYS = ROOT / "data" / "market_holidays.csv"
TASKS = ("return_900", "volatility_change_900", "spread_change_900")


def parse_args(doc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-month", default="2020-01",
                   help="one of the retrainable released months: 2020-01, "
                        "2020-08, 2020-09, 2020-12")
    p.add_argument("--work", default=str(ROOT / "runs" / "replicate"))
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="largest relative gap to the released encoder that passes")
    p.add_argument("--smoke", action="store_true",
                   help="train 50 steps and score on 4 shards: checks the "
                        "pipeline end to end in minutes; the numbers mean nothing")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel embedding processes; 0 = CPUs / 4, capped at 16")
    p.add_argument("--skip-train", action="store_true",
                   help="only score the released encoder (checks the scorer)")
    return p.parse_args()


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def months_back(ym: str, n: int) -> list[str]:
    """The n months before ym, oldest first."""
    y, m = map(int, ym.split("-"))
    out = []
    for _ in range(n):
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
        out.append(f"{y:04d}-{m:02d}")
    return out[::-1]


class Month:
    """The six-month training span before an evaluation month, and its dates."""

    def __init__(self, eval_month: str):
        self.eval = eval_month
        self.fit = months_back(eval_month, 6)
        self.all = self.fit + [eval_month]
        import calendar
        last = lambda ym: calendar.monthrange(*map(int, ym.split("-")))[1]  # noqa: E731
        self.train_start = f"{self.fit[0]}-01"
        self.train_end = f"{self.fit[-1]}-{last(self.fit[-1]):02d}"
        self.eval_start = f"{eval_month}-01"
        self.eval_end = f"{eval_month}-{last(eval_month):02d}"

    def overrides(self) -> list[str]:
        return [f"dataset.train_date_start={self.train_start}",
                f"dataset.train_date_end={self.train_end}",
                f"dataset.eval_train_date_start={self.train_start}",
                f"dataset.eval_train_date_end={self.train_end}",
                f"dataset.eval_date_start={self.eval_start}",
                f"dataset.eval_date_end={self.eval_end}"]


# ── data ─────────────────────────────────────────────────────────────────────

def hub(fn, *args, **kw):
    """Call a Hub API function, waiting out HTTP 429 and retrying.

    THE HUB ALLOWS 1000 API REQUESTS PER 5 MINUTES PER ACCOUNT, logged in or
    not, and every file downloaded costs at least one. One evaluation month is
    ~2500 files across the three layouts, so a 429 is expected, not
    exceptional -- and once the window is spent, every call fails, listings
    and whoami included, until it rolls over.
    """
    for attempt in range(1, 13):
        try:
            return fn(*args, **kw)
        except Exception as e:  # HfHubHTTPError and requests' HTTPError alike
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) != 429 or attempt == 12:
                raise
            wait = int(resp.headers.get("Retry-After", 0) or 0) or 300
            print(f"rate-limited by the Hub (attempt {attempt}); "
                  f"resuming in {wait} s", flush=True)
            time.sleep(wait)


def check_login() -> None:
    from huggingface_hub import get_token, whoami
    if get_token() is None:
        print("WARNING: not logged in to Hugging Face; run `hf auth login` or "
              "set HF_TOKEN.", file=sys.stderr)
        return
    print(f"Hugging Face: logged in as {hub(whoami)['name']}")


def download(data: Path, repo: tuple[str, str], months: list[str]) -> Path:
    """Download months of one Market-1T layout, then PROVE nothing is missing.

    A rate-limited snapshot_download can return without error with files
    missing, so the listing is checked against the disk afterwards.
    """
    from huggingface_hub import list_repo_files, snapshot_download
    name, sub = repo
    prefixes = [f"{sub}/{m.replace('-', '/')}/" for m in months]
    want = [f for f in hub(list_repo_files, f"{ORG}/{name}", repo_type="dataset")
            if f.startswith(tuple(prefixes))]
    if not want:
        raise SystemExit(f"{name} has no files for {months}")
    missing = [f for f in want if not (data / f).exists()]
    # snapshot_download skips files already on disk, so each round fetches
    # only what is left -- including after a round that returned cleanly with
    # files missing, which a spent rate-limit window can also cause.
    for rnd in range(6):
        if not missing:
            break
        if rnd:
            print(f"{len(missing)} files still missing; retrying in 300 s", flush=True)
            time.sleep(300)
        print(f"downloading {len(missing)} of {len(want)} files from {name}", flush=True)
        hub(snapshot_download, f"{ORG}/{name}", repo_type="dataset", local_dir=data,
            allow_patterns=[p + "*" for p in prefixes], max_workers=4)
        missing = [f for f in want if not (data / f).exists()]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} files still missing, "
                         f"e.g. {missing[0]}. Rerun to resume.")
    return data / sub


def decompress_daystore(root: Path, months: list[str]) -> Path:
    """The Hub day store ships features.npy.zst only, and the reader memmaps
    features.npy. Reading through hf:// decompresses on the way in; a LOCAL
    copy must be decompressed here or training dies in the dataloader."""
    from stable_finance.dataset.daystore import decompress_day
    days = [d for m in months for d in sorted((root / m.replace("-", "/")).iterdir())
            if (d / "meta.json").is_file()]
    print(f"decompressing {len(days)} day-store days (skips any already done)", flush=True)
    for d in days:
        decompress_day(d, remove_zst=False)
    return root


def build_targets(sparse: Path, out: Path, months: list[str]) -> Path:
    """The per-month cross-sectional target tables (paper: xs_anchor_stats_fwdvwap60)."""
    todo = [m for m in months if not (out / f"{m}.npz").exists()]
    if todo:
        run(["uv", "run", "sf-build-targets", "--mosaic-dir", sparse,
             "--out-dir", out, "--start", todo[0], "--end", todo[-1],
             "--holiday-csv", HOLIDAYS, "--workers", str(max(1, os.cpu_count() // 4))])
    return out


def released(work: Path, slug: str, month: str) -> Path:
    """One month of a released encoder, in a private copy (scoring writes into it)."""
    from huggingface_hub import snapshot_download
    src = Path(hub(snapshot_download, f"{ORG}/tfwm-{slug}", allow_patterns=[f"{month}/*"]))
    dst = work / "released" / slug / month
    if not dst.exists():
        shutil.copytree(src / month, dst)
    return dst


# ── training ─────────────────────────────────────────────────────────────────

def config_diff(ours: Path, theirs: Path) -> list[str]:
    """Training-relevant fields of two train_meta configs that differ."""
    def flat(d, p=""):
        if not isinstance(d, dict):
            return {p: d}
        o = {}
        for k, v in d.items():
            o.update(flat(v, f"{p}.{k}" if p else str(k)))
        return o
    a = flat(json.loads(ours.read_text())["config"])
    b = flat(json.loads(theirs.read_text())["config"])
    # Where the files live, not what was trained; and live-probe settings,
    # which the reported recipe does not use (training.live_eval=false).
    skip = ("machine.", "wandb.", "checkpoint.", "probe_eval.", "skip_if_done",
            "dataset.xs_anchor_stats_dir", "training.max_train_steps",
            "optimizer.warmup_steps")
    out = []
    for k in sorted(set(a) | set(b)):
        if k.startswith(skip) or k not in b:   # absent in the release = added since
            continue
        if a.get(k, "<absent>") != b[k]:
            out.append(f"{k}: ours={a.get(k, '<absent>')!r} released={b[k]!r}")
    return out


def train(work: Path, name: str, overrides: list[str], smoke: bool) -> Path:
    """Train with train.py and return the checkpoint directory."""
    ckpts = work / "checkpoints" / (f"{name}-smoke" if smoke else name)
    # Weights are written last, so a crashed run is never reused.
    finished = lambda: sorted(  # noqa: E731
        (p.parent for p in ckpts.glob("*/*/train_meta.json")
         if (p.parent / "model.pt").exists() or (p.parent / "backbone.pt").exists()),
        key=lambda d: d.stat().st_mtime)
    if finished() and not smoke:
        print(f"reusing {finished()[-1]} (delete it to retrain)")
        return finished()[-1]
    extra = ["training.max_train_steps=50"] if smoke else []
    env = {**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "offline")}
    run(["uv", "run", "train.py", *overrides, *extra,
         f"checkpoint.chkpt_dir={ckpts}", "skip_if_done=false",
         f"wandb.project=replicate-{name}"], env=env)
    if not finished():
        raise SystemExit(f"training wrote no checkpoint under {ckpts}")
    return finished()[-1]


# ── scoring ──────────────────────────────────────────────────────────────────

def _workers(n: int) -> int:
    return n or max(1, min(16, (os.cpu_count() or 4) // 4))


def score_probe(work: Path, ckpts: dict[str, Path], m: Month, mosaic: Path,
                targets: Path, workers: int, smoke: bool) -> dict:
    """The paper's forecasting probe: last-patch readout, ridge (alpha 10) fit
    on the six training months at 36 anchors/day, rank IC on the eval month at
    8. Returns {name: {task: (ic, se)}} at the largest fit size."""
    shards = 4 if smoke else 32
    workers = _workers(workers)
    cache = work / "probe_cache"
    tasks = work / "probe_tasks.txt"
    lines = [f"{ck} {mo} {s} {36 if mo != m.eval else 8}"
             for ck in ckpts.values() for mo in m.all for s in range(shards)]
    tasks.write_text("\n".join(lines) + "\n")
    common = ["--tasks", tasks, "--num-workers", str(workers), "--num-shards",
              str(shards), "--out-dir", cache, "--pool", "last",
              "--mosaic-dir", mosaic, "--xs-anchor-stats-dir", targets,
              "--holiday-csv", HOLIDAYS]
    print(f"+ embedding {len(ckpts)} checkpoint(s) x {len(m.all)} months x "
          f"{shards} shards on {workers} workers", flush=True)
    procs = [subprocess.Popen(["uv", "run", "scripts/eval/probe_fit_size.py",
                               "embed-many", "--worker-index", str(i), *map(str, common)],
                              cwd=ROOT) for i in range(workers)]
    if any(p.wait() for p in procs):
        raise SystemExit("an embedding worker failed; see its output above")
    out = work / "probe_results.json"
    run(["uv", "run", "scripts/eval/probe_fit_size.py", "reduce", "--out-dir", cache,
         "--eval-month-glob", m.eval, "--alphas", "10", "--expect-shards",
         str(shards), "--json", out])
    rows = json.loads(out.read_text())
    res = {}
    for name, ck in ckpts.items():
        mine = [r for r in rows if r["ckpt"] == ck.name and r["alpha"] == 10.0]
        # The cache keys a checkpoint by its folder name: a run id for ours,
        # the eval month for the released copy, so the two never collide.
        top = max(mine, key=lambda r: r["n"])
        res[name] = {t: (top[t], top.get(f"{t}_se")) for t in TASKS}
    return res


def score_head(ckpt: Path, m: Month, mosaic: Path, targets: Path, task: str) -> tuple:
    """The supervised arms' reported number: the trained head's rank IC on the
    eval month (8 anchors/day), written to <ckpt>/xs_ic.json."""
    run(["uv", "run", "scripts/generic/post_train_ic_eval.py", "--ckpt-dir", ckpt,
         "--train-month", m.fit[-1], "--eval-month", m.eval, "--head-only",
         "--no-wandb", "--mosaic-dir", mosaic, "--xs-anchor-stats-dir", targets,
         "--holiday-csv", HOLIDAYS])
    ic = json.loads((ckpt / "xs_ic.json").read_text())
    return ic[f"xs_ic/head:{task}"], ic.get(f"xs_ic_se/head:{task}")


# ── verdict ──────────────────────────────────────────────────────────────────

def compare(title: str, ours: dict, rel: dict, gate: list[str], paper: dict,
            tol: float, out: Path) -> bool:
    print(f"\n{title}\n" + "-" * len(title))
    print(f"{'task':24} {'paper':>8} {'released':>9} {'ours':>9} {'gap':>7}  verdict")
    ok = True
    for t in ours:
        o, r = ours[t][0], rel[t][0]
        gap = (o - r) / abs(r) if r else float("nan")
        gated = t in gate
        passed = abs(gap) <= tol
        ok &= passed or not gated
        verdict = ("PASS" if passed else "FAIL") if gated else "(not gated)"
        print(f"{t:24} {paper.get(t, float('nan')):8.4f} {r:9.4f} {o:9.4f} "
              f"{gap:+7.1%}  {verdict}")
    print(f"\n{'PARITY' if ok else 'NO PARITY'} at {tol:.0%} on {', '.join(gate)}")
    out.write_text(json.dumps({"ours": ours, "released": rel, "paper": paper,
                               "gate": gate, "tolerance": tol, "pass": ok}, indent=1))
    print(f"wrote {out}")
    return ok
