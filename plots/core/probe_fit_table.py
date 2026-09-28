"""Rank IC at the FULL fit pool, one column per task, one row per method.

The companion to plots/core/fixed_panel_table.py: the same rows, in the
same order, under the same family headings and with the same names -- but
asking the forecasting question instead of the organization one. Every number
here is a ridge probe on frozen features, fit on the whole six-month pool the
supervised head itself trained on (~2M rows at 36 anchors/day) and scored on
the reported eval month at 8 anchors/day.

WHY THE LARGEST n AND ONLY THE LARGEST. scripts/eval/probe_fit_size.py walks a
nested ladder (2048 ... 1,152,000, then the pool) so a curve can show where
the probe saturates; that curve is plots/core/probe_fit_breadth.py's
subject. This table is the ENDPOINT of the ladder -- the probe given
everything -- so it takes the max n per (checkpoint, eval month) and nothing
else. Mixing ladder rungs across arms would compare probes fit on different
amounts of data and call it a method difference.

THE SUPERVISED ARMS CARRY TWO NUMBERS ON THEIR OWN TASK. A specialist trained
a head on one target, and the multihead on all three; that head is the model's
actual output, and scoring it by linear probe measures a readout nobody would
deploy. So on a supervised arm's native column the cell is
``probe``/``head`` -- the same frozen features read two ways -- and the head
number comes from the checked-in artifact plots/metrics/supervised_head_ic.json
(build_supervised_ic.py --readout head), NOT from this sweep, which only ever
fits probes. Off its native task a supervised arm is probe-only, like everyone
else.

PARTIAL BY DESIGN. The sweep lands one json per (job, eval month) as its jobs
finish, so this renders whatever exists and says how many months each row got.
A row with n=2 is not a result; the ``n`` column is there so it cannot be read
as one. Pass --min-months to drop the thin rows once the panel fills in.

Usage::

    uv run plots/core/probe_fit_table.py                    # markdown
    uv run plots/core/probe_fit_table.py --latex \\
        --out plots/core/probe_fit_table.tex
    ... --alpha 10 --min-months 5 --no-n --size footnotesize
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "plots/latent_eval/fixed_panel"))
sys.path.insert(0, str(ROOT / "plots/core"))
sys.path.insert(0, str(ROOT / "plots/task_corr"))

# Every input path is declared in paths.py -- see its docstring for why the
# files themselves stay beside their producers instead of moving in here.
from paths import HEAD_JSON, RANKME_JSON, RESULTS, index_rows, result_files  # noqa: E402,F401
import readout as _readout  # noqa: E402

# The three reported targets at h=900 (15 minutes), in the paper's order.
TASKS = [("return_900", "Return"),
         ("volatility_change_900", "Volatility"),
         ("spread_change_900", "Spread")]

FAMILY_ORDER = ["Supervised", "LeJEPA", "SSL"]
FLOOR_LABEL = "Random ViT"

# THE CAPTION LIVES HERE, not on the command line. It used to be passed with
# --caption, which meant a plain re-render emitted a table with no \caption
# and no \label at all -- silently, because LaTeX is happy to typeset a
# floating tabular. Every refresh then had to splice the old caption back in
# by hand. Every ORDERING it asserts is re-tested against the current numbers
# by check_caption_claims() on every run, so this string cannot quietly go
# stale the way a hand-carried one did.
# TWO SENTENCES, BUILT NOT APPENDED. The caption this replaced grew to a
# full paragraph because every new fact was appended to a fixed string: the
# month count, then the RankMe column, then the dagger. Anything conditional
# now goes INSIDE one of the two sentences, so the caption cannot grow a third
# by accretion. What the table means lives here; what the numbers SAY belongs
# in the body text, which is where the orderings check_caption_claims() still
# reports now live.
def default_caption(n_months=None, rankme=False, dagger=False):
    """The table's two sentences: what the cells are, then what RankMe is."""
    span = (f"averaged over {n_months} eval months"
            if n_months else
            "averaged over the eval months each arm shares with the floor")
    one = (r"Forecasting rank IC at the 15-minute horizon from a ridge probe "
           r"on frozen six-month features, " + span
           + r"; a supervised arm's own head IC follows in parentheses, "
             r"\sout{struck} cells fail to beat the untrained Random ViT "
             r"floor, and \textbf{bold} marks the best two "
             r"\emph{self-supervised} arms per target.")
    if not rankme:
        return one
    two = (r" RankMe~\citep{garrido2023rankme} measures how many directions "
           r"the frozen embedding spends its variance on and is reported for "
           r"reference only---it is not floor-relative, so nothing in that "
           r"column is struck or bold")
    two += (r" ($\dagger$: a different embedding width, which bounds RankMe, "
            r"so not on the rest of the column's scale)." if dagger else ".")
    return one + two


# BOLD IS FOR THE SELF-SUPERVISED FIELD ONLY. The supervised multihead and
# the column's own specialist take the top two places on every target, so
# bolding "the best overall" would mark the same two supervised rows three
# times and say nothing. The contest the table is actually reporting is among
# the arms that never saw the label, so that is where the bold goes; the
# caption states the supervised result in words instead.
N_BOLD = 2
BOLD_EXCLUDE_FAMILY = "Supervised"

# WHICH COLUMN EACH SUPERVISED ARM OWNS. A head is trained at one horizon on
# one target (the multihead on all three), and ROSTER's "head" field is the
# same statement -- read from there rather than restated, so the two tables
# cannot disagree about which cell is native.
HEAD_TASK = {"return": "return_900",
             "volatility_change": "volatility_change_900",
             "spread_change": "spread_change_900"}

# WHOSE HEAD IS WHOSE is recorded on each record now, so there is no map here.
# The multihead has a head on all three targets and the specialists one each;
# keying by target alone (as supervised_head_ic.json forces) gave the multihead
# the specialists' values -- on 2013-01 its head column read
# 0.0302/0.1027/0.1715, theirs exactly. probe_head_ic.json carries series_key.


def roster():
    """[(series_key, label, family, native_tasks)] in ROSTER order.

    Imported lazily -- task_rank_corr pulls scipy at module scope and this
    script must run without it. The ``<x>_final`` -> ``<x>_6mo`` mapping is
    the one latex_roster() makes: the probe sweep scored the six-month wave,
    ROSTER is keyed by the one-month generation it was written for, and they
    are one method on two budgets. Keeping the mapping here rather than a
    third spelling of each name is the whole point.
    """
    from task_rank_corr import ROSTER
    order = _model_order()
    rank = {k: i for i, k in enumerate(order)}

    def pos(m):
        # A ROSTER row is ranked by whichever of its names MODEL_ORDER carries
        # (the six-month key for the SSL/LeJEPA arms); anything MODEL_ORDER
        # does not name sorts after, in ROSTER order.
        for k in series_aliases(m["lat"]):
            if k in rank:
                return rank[k]
        return len(rank)

    out = []
    for m in sorted(ROSTER, key=pos):
        key = m["lat"]
        native = {HEAD_TASK[h] for h in m.get("head", ()) if h in HEAD_TASK}
        out.append((key, m["label"], m["fam"], native, m.get("cite", "")))
    return out


def _model_order():
    """MODEL_ORDER from the fixed-panel registry -- the reported row order."""
    try:
        from industry_nn_sweep import MODEL_ORDER
        return list(MODEL_ORDER)
    except Exception:            # registry unavailable: keep ROSTER's order
        return []


def series_aliases(key):
    """Every series_key this ROSTER row may answer to, most specific first.

    ROSTER is keyed by the one-month generation (``<x>_final``); the probe
    sweep scored the SIX-MONTH wave (``<x>_6mo``). Without this the fourteen
    SSL and LeJEPA rows silently find no results and the table renders the
    four supervised arms as if they were the whole panel -- which is exactly
    what it did before 2026-09-15.
    """
    out = [key]
    if key.endswith("_final"):
        out.append(key[:-len("_final")] + "_6mo")
    return out


def family_cites():
    from task_rank_corr import FAMILY_CITE
    return dict(FAMILY_CITE)


def _citep(cite):
    """LATEX ONLY, and appended AFTER _tex_escape -- escaping a \\citep turns
    it into \\textbackslash{}citep and prints the macro instead of running it.
    """
    return (r"~\citep{" + cite + "}") if cite else ""


def load_index():
    """{(run_id, eval_month): series_key} over every manifest index present."""
    return {(r["run_id"], r["eval_month"]): r["series_key"]
            for r in index_rows()}


def _eval_month_of(path: Path) -> str | None:
    m = re.search(r"(\d{4}-\d{2})(?=\.json$)", path.name)
    return m.group(1) if m else None


def load_results(alpha: float):
    """{series_key: {task: {month: ic}}} plus {series_key: {month: n}}.

    Only the LARGEST n per (checkpoint, eval month) survives -- see the module
    docstring. Rows are keyed by the checkpoint's run_id, which the manifest
    index maps to the series.
    """
    idx = load_index()
    # COLLECT EVERY ROW FIRST, THEN CHOOSE THE READOUT. Picking the largest n
    # as we go would mix a mean-pooled row and a last-pooled one for the same
    # checkpoint, which are identical apart from the `readout` stamp -- see
    # readout.py. The choice has to be made per (series, month) over ALL of
    # that key's rows, before n is compared.
    seen: dict[tuple, list] = {}
    unknown: set[str] = set()
    files = result_files(RESULTS)
    for name, rows in files:
        ym = _eval_month_of(Path(name))
        if ym is None:
            continue
        for row in rows:
            if abs(float(row.get("alpha", -1)) - alpha) > 1e-9:
                continue
            series = idx.get((row["ckpt"], ym))
            if series is None:
                unknown.add(row["ckpt"])
                continue
            seen.setdefault((series, ym), []).append(row)

    best: dict[tuple, tuple] = {}          # (series, month) -> (n, row)
    blank: set[str] = set()
    for k, rows in seen.items():
        kept, used = _readout.pick(rows, k[0])
        if used != _readout.PREDICT_READOUT:
            blank.add(k[0])
        for row in kept:
            n = int(row["n"])
            if k not in best or n > best[k][0]:
                best[k] = (n, row)
    _readout.report(blank, sys.stderr)

    ics: dict[str, dict[str, dict[str, float]]] = {}
    ns: dict[str, dict[str, int]] = {}
    for (series, ym), (n, row) in best.items():
        ns.setdefault(series, {})[ym] = n
        for task, _ in TASKS:
            v = row.get(task)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ics.setdefault(series, {}).setdefault(task, {})[ym] = float(v)
    return ics, ns, len(files), sorted(unknown)


def load_head():
    """{task: {month: ic}} for the supervised heads (specialists only today).

    Keyed by (series_key, target) -- the arm is on the record, so the
    multihead's three heads and the specialists' one each stay separate.
    Written by build_probe_head_ic.py off the sweep's own checkpoints.
    """
    if not HEAD_JSON.exists():
        return {}
    out: dict[tuple, dict[str, float]] = {}
    if not HEAD_JSON.exists():
        return out
    for r in json.loads(HEAD_JSON.read_text()):
        if r.get("ic") is None:
            continue
        out.setdefault((r["series_key"], r["target"]), {})[
            r["eval_month"]] = float(r["ic"])
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def comparable_months(ics, tasks):
    """``{task: months where the floor AND some method both scored}``.

    THE DEFAULT MONTH SET, and the reason the floor can strike anything at
    all. Every cell under the floor row is struck through, so that row is only
    a bar if it was read on the SAME market the rows beneath it were. The two
    waves do not scan together -- pbfloor started at 2008-08 while the method
    rows were 2011-2013 -- and a floor averaged over one era struck out rows
    measured in another: 2008-08 is crisis-era and its spread_change carries
    the mechanical leak a scalar known at t already wins on, so the
    "untrained" bar came in at 0.1871 on spread against method rows that never
    saw that month. That is a property of the month, not of the architecture.

    Both directions are required. Dropping the floor's uncovered months is the
    obvious half; dropping the METHODS' uncovered months matters just as much,
    or the floor averages over a month no row beneath it is scored on and the
    bar again describes a market the table does not show.
    """
    floor_keys = [k for k in ics if k.startswith("randinit_")]
    meth_keys = [k for k in ics if not k.startswith("randinit_")]
    out = {}
    for task, _ in tasks:
        fm = {m for k in floor_keys for m in ics[k].get(task, {})}
        mm = {m for k in meth_keys for m in ics[k].get(task, {})}
        out[task] = fm & mm
    return out


def _keep(months, shared, restrict, task):
    """Months surviving the fair set and the floor-covered set."""
    if shared is not None:
        months = [m for m in months if m in shared]
    if restrict is not None:
        months = [m for m in months if m in restrict.get(task, ())]
    return months


# RankMe rides along from scripts/eval/rankme.py's json. It is NOT an IC and
# nothing in this table's marking applies to it: it is not floor-relative, so
# no RankMe cell is ever struck or bold, and it is not ranked with the three
# target columns.


def load_rankme(path=None, metric="rankme"):
    """``{series_key: (mean, se, n_months, widths)}`` for the RankMe column.

    SEEDS WITHIN A MONTH FIRST, THEN MONTHS -- floor_row()'s order, for
    floor_row()'s reason: a randinit seed is a draw from the same untrained
    architecture, so averaging the three within a month keeps one seed's noise
    out of the month-to-month spread the SE is read from. The seeds collapse
    to one :data:`FLOOR_LABEL` entry, which is the key the floor row looks up.

    Returns ``{}`` when the file is absent. The column is an addition to this
    table, not a precondition for it -- a checkout without the json still
    renders the table it always did.
    """
    p = Path(path) if path else RANKME_JSON
    if not p.exists():
        return {}
    by: dict[str, dict[str, list]] = {}
    dims: dict[str, set] = {}
    for r in json.loads(p.read_text()):
        key = (FLOOR_LABEL if r["series"].startswith("randinit_")
               else r["series"])
        by.setdefault(key, {}).setdefault(r["month"], []).append(
            float(r[metric]))
        dims.setdefault(key, set()).add(int(r["d"]))
    out = {}
    for key, months in by.items():
        per_month = [sum(v) / len(v) for v in months.values()]
        n = len(per_month)
        mu = sum(per_month) / n
        se = None
        if n > 1:
            var = sum((x - mu) ** 2 for x in per_month) / (n - 1)
            se = math.sqrt(var / n)
        out[key] = (mu, se, n, sorted(dims[key]))
    return out


def rankme_for(rankme, series):
    """This row's RankMe entry, resolved through the same aliases as its IC."""
    if not rankme:
        return None
    if series == "__floor__":
        return rankme.get(FLOOR_LABEL)
    key = next((k for k in series_aliases(series) if k in rankme), None)
    return rankme.get(key)


def rankme_modal_width(rankme, rows, floor=None):
    """The embedding width MOST of the displayed arms share.

    RankMe is bounded above by the embedding width, so an arm of a different
    width is not on the same scale as the rest of the column and gets a
    dagger. Computed over the arms actually displayed rather than hardcoded:
    which arm is the odd one out is a property of the roster, not a constant.
    """
    widths = []
    for r in list(rows) + ([floor] if floor else []):
        got = rankme_for(rankme, r[0])
        if got:
            widths.extend(got[3])
    if not widths:
        return None
    return max(set(widths), key=widths.count)


def build_rows(ics, ns, head, *, min_months, fair, restrict=None):
    """[(series, label, family, native, {task: (probe, head, n)})].

    ``fair`` restricts every row to the months EVERY scored row shares, so a
    column is a comparison rather than an average over whichever months each
    arm happened to finish. Off by default while the sweep is filling in --
    with one arm at n=1 the fair set is 1 -- and worth turning on for the
    reported version.
    """
    have = {k for k in ics}
    months_by = {k: {m for t in ics[k].values() for m in t} for k in have}
    shared = set.intersection(*months_by.values()) if fair and months_by else None

    out = []
    for series, label, fam, native, cite in roster():
        cells = {}
        alias = next((k for k in series_aliases(series) if k in ics), series)
        for task, _ in TASKS:
            per = ics.get(alias, {}).get(task, {})
            months = sorted(_keep(list(per), shared, restrict, task))
            if not months:
                cells[task] = (float("nan"), None, 0)
                continue
            probe = _mean([per[m] for m in months])
            h = None
            if task in native:
                hv = [head.get((series, task), {}).get(m) for m in months]
                hv = [x for x in hv if x is not None]
                h = _mean(hv) if hv else None
            cells[task] = (probe, h, len(months))
        if max(c[2] for c in cells.values()) < min_months:
            continue
        out.append((series, label, fam, native, cells, cite))
    return out


def floor_row(ics, *, min_months, shared=None, method_months=None,
              restrict=None):
    """The untrained ViT, averaged over its seeds THEN over months.

    Each (seed, eval month) has its own checkpoint, so a seed is a draw from
    the same untrained architecture and the three are exchangeable; averaging
    them first keeps one seed's noise from setting the bar a whole column is
    struck against.

    THE FLOOR IS DROPPED UNTIL IT COVERS THE MONTHS THE METHODS DO. Every
    value below this row gets struck through, so a floor read on a DIFFERENT
    month set does not set a bar -- it sets a bar for a different market. The
    first floor job back was 2008-08 while every method row was 2011-2013, and
    it struck out all but four cells in the table: 2008-08 is crisis-era, and
    spread_change there carries the mechanical leak a scalar known at t
    already wins on, so the "untrained" bar came in at 0.1871 on spread and
    0.0999 on volatility. Those are properties of the month, not of the
    architecture. ``method_months`` is the per-task month count the method
    rows rest on; the floor draws only where it is at least as well sampled.
    """
    seeds = sorted(k for k in ics if k.startswith("randinit_"))
    if not seeds:
        return None
    cells = {}
    for task, _ in TASKS:
        months = sorted({m for s in seeds for m in ics[s].get(task, {})})
        months = _keep(months, shared, restrict, task)
        vals = [_mean([ics[s].get(task, {}).get(m) for s in seeds])
                for m in months]
        cells[task] = (_mean(vals), None, len(months))
    if max(c[2] for c in cells.values()) < min_months:
        return None
    if method_months:
        need = min(method_months.values())
        have = min(c[2] for c in cells.values())
        if have < need:
            print(f"  floor HELD BACK: {have} month(s) against the methods' "
                  f"{need}; it would strike rows it is not sampled to judge. "
                  f"It returns on its own once the pbfloor wave catches up.",
                  file=sys.stderr)
            return None
    return ("__floor__", FLOOR_LABEL, "Untrained floor", set(), cells, "")


def trim_family(label, fam):
    """``label`` without its family prefix -- fixed_panel_table.py's rule."""
    if not label.lower().startswith(fam.lower()):
        return label
    rest = label[len(fam):].strip()
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()
    if not rest:
        return label
    return rest[0].upper() + rest[1:] if rest[0].isalpha() else rest


def _fmt(v, digits=4):
    return "--" if v is None or math.isnan(v) else f"{v:.{digits}f}"


def check_caption_claims(rows, floor=None, out=sys.stderr):
    """Re-test, on the CURRENT data, the orderings the PROSE relies on.

    These were caption sentences until 2026-09-17, when the caption was cut to
    two and now states only what the cells and marks MEAN, not what they say.
    The claims did not stop mattering -- they moved into the body text, which
    this script cannot see and which no longer moves with the numbers at all.
    So the check stays, and its report is the only thing standing between a
    sweep landing another month and a paragraph in the paper going quietly
    false. Each claim is a statement about an ORDERING, which is exactly what
    a few more months can flip. Printed on every run, loudly when broken.

    Claims, as of 2026-09-22 (31 eval months, six-month pool). The 2026-09-15
    set -- time warp first, at 1.3--2.0x the runner-up -- did not survive the
    full panel: TS2Vec edges it on return, ties it on volatility, and every
    lead in the self-supervised field is ~1%.
      1. the multihead and the target's own specialist are the top two;
      2. the top two self-supervised arms per target are TOP2_SSL (the bold);
      3. no self-supervised arm leads the runner-up by more than LEAD_MAX --
         the prose says there is no clear self-supervised winner;
      4. which self-supervised arms clear the untrained floor (CLEARS).
    """
    NATIVE = {"return_900": "Supervised (return)",
              "volatility_change_900": "Supervised (vol)",
              "spread_change_900": "Supervised (spread)"}
    MULTI = "Supervised (multihead)"
    WARP = "LeJEPA Time Warping"
    TOP2_SSL = {"return_900": {WARP, "TS2Vec"},
                "volatility_change_900": {WARP, "TS2Vec"},
                "spread_change_900": {WARP, "LeJEPA Same Stock, Diff. View"}}
    LEAD_MAX = 1.05
    # Claim 4 is the most fragile -- the floor is a BAR, so a shift of 0.001
    # can hand a row back or take it away.
    CLEARS = {"return_900": {WARP, "TS2Vec", "TimeMAE", "CPC", "MAE",
                             "LeJEPA Same Stock, Diff. View",
                             "LeJEPA Gaussian Noising",
                             "LeJEPA C-S, Same Industry"},
              "volatility_change_900": {WARP, "TS2Vec", "TimeMAE", "CPC",
                                        "LeJEPA Same Stock, Diff. View"},
              "spread_change_900": set()}

    ok = True
    ratios = []
    print("\n-- caption claim check --", file=out)
    for task, name in TASKS:
        live = [(r[4][task][0], r[1], r[2]) for r in rows
                if not math.isnan(r[4][task][0])]
        if not live:
            continue
        live.sort(reverse=True)
        top2 = {l for _, l, _ in live[:2]}
        c1 = top2 == {MULTI, NATIVE[task]}
        field = [(v, l) for v, l, f in live if f != "Supervised"]
        ssl2 = {l for _, l in field[:2]}
        c2 = ssl2 == TOP2_SSL[task]
        ratio = (field[0][0] / field[1][0]) if len(field) > 1 and field[1][0] > 0 else float("nan")
        if not math.isnan(ratio):
            ratios.append(ratio)
        c3 = not math.isnan(ratio) and ratio <= LEAD_MAX
        ok &= c1 and c2 and c3
        print(f"   {name:11s} top2={'OK ' if c1 else 'BROKEN'} ({', '.join(sorted(top2))})"
              f"  ssl_top2={'OK ' if c2 else 'BROKEN'} ({', '.join(sorted(ssl2))})"
              f"  lead={ratio:.2f}x {'OK' if c3 else f'ABOVE {LEAD_MAX}x'}",
              file=out)
    if ratios:
        print(f"   measured self-supervised lead {min(ratios):.2f}--{max(ratios):.2f}x "
              f"(prose says none above {LEAD_MAX}x)", file=out)
    if floor is not None:
        for task, name in TASKS:
            bar = floor[4][task][0]
            if math.isnan(bar):
                continue
            got = {r[1] for r in rows if r[2] != "Supervised"
                   and not math.isnan(r[4][task][0]) and r[4][task][0] > bar}
            c4 = got == CLEARS[task]
            ok &= c4
            print(f"   {name:11s} clears floor ({bar:.4f}): "
                  f"{'{' + ', '.join(sorted(got)) + '}' if got else 'none'}"
                  f"  {'OK' if c4 else 'BROKEN -- caption names '
                     + (', '.join(sorted(CLEARS[task])) or 'none')}", file=out)
    print("   ALL CAPTION CLAIMS HOLD" if ok else
          "   *** CAPTION IS NOW WRONG -- REWORD IT BEFORE USING THIS TABLE ***",
          file=out)
    return ok


def check_rankme_months(rows, floor, rankme, out=sys.stderr):
    """Report arms whose RankMe rests on a different month set than their IC.

    The two columns come from different pipelines -- IC from the probe-breadth
    reduce, RankMe from the rescued embeddings -- and an arm can have one
    without the other. The ``n`` column reports the IC months, so an arm whose
    RankMe covers fewer is quietly showing two different panels in one row.
    That is acceptable (RankMe is reference, not a ranked column) but it must
    not be silent.
    """
    bad = []
    for r in list(rows) + ([floor] if floor else []):
        got = rankme_for(rankme, r[0])
        if got is None:
            continue
        ic_n = max(c[2] for c in r[4].values())
        if got[2] != ic_n:
            bad.append((r[1], got[2], ic_n))
    if bad:
        print("  RankMe month coverage differs from IC for "
              f"{len(bad)} row(s):", file=out)
        for label, rn, icn in bad:
            print(f"    {label}: RankMe {rn} month(s), IC {icn}", file=out)
    return not bad


def render_latex(rows, floor, *, show_n, caption, label, size, tabcolsep,
                 rankme=None):
    from style import render_latex_table, tex_escape as _tex_escape
    from task_rank_corr import latex_size_label
    fcites = family_cites()

    # Bold the best N per column among TRAINED arms, on the PROBE number: the
    # head is a different readout and ranking the two together would put a
    # supervised arm above the field for having a head rather than a feature.
    field = [r for r in rows if r[2] != BOLD_EXCLUDE_FAMILY]
    topN = {}
    for task, _ in TASKS:
        cand = [(r[4][task][0], r[0]) for r in field
                if not math.isnan(r[4][task][0])]
        topN[task] = {k for _, k in sorted(cand, reverse=True)[:N_BOLD]}

    fl = floor[4] if floor else None

    def cell(r, task):
        probe, h, n = r[4][task]
        if math.isnan(probe):
            return "--"
        txt = _fmt(probe)
        if r[0] in topN[task]:
            txt = r"\mathbf{" + txt + "}"
        txt = f"${txt}$"
        if (fl and r[0] != "__floor__" and not math.isnan(fl[task][0])
                and probe <= fl[task][0]):
            txt = r"\sout{" + txt + "}"
        if h is not None and not math.isnan(h):
            # THE HEAD GOES IN PARENTHESES, not after a slash. "a/b" reads as a
            # ratio or as an either/or; the two numbers here are the same
            # frozen features read two ways, and the probe is the one the
            # column is about. Parentheses say "and its head as well".
            ht = f"${h:.4f}$"
            if fl and not math.isnan(fl[task][0]) and h <= fl[task][0]:
                ht = r"\sout{" + ht + "}"
            txt = txt + r"\,(" + ht + ")"
        return txt

    # THE RANKME COLUMN IS NOT MARKED. Bold is "best two self-supervised on
    # this target" and \sout is "does not beat the floor"; neither is defined
    # for a spectrum statistic, and a wide spectrum is not a good one (on this
    # panel it is anti-correlated with IC). It prints as a plain number.
    modal_d = rankme_modal_width(rankme, rows, floor) if rankme else None

    def rcell(r):
        got = rankme_for(rankme, r[0])
        if got is None:
            return "--"
        mu, _se, _n, widths = got
        txt = f"{mu:.1f}"
        # Bounded by the embedding width, so an off-width arm is not on the
        # same scale as the column it sits in.
        if modal_d is not None and widths and modal_d not in widths:
            txt += r"^{\dagger}"
        return f"${txt}$"

    n_rank = 1 if rankme else 0
    ncol = 1 + n_rank + len(TASKS) + (1 if show_n else 0)
    body = []
    by_fam: dict[str, list] = {}
    for r in rows:
        by_fam.setdefault(r[2], []).append(r)
    order = FAMILY_ORDER + [f for f in by_fam if f not in FAMILY_ORDER]
    for fam in order:
        if fam not in by_fam:
            continue
        body.append([r"\addlinespace \multicolumn{" + str(ncol)
                     + r"}{l}{\emph{" + _tex_escape(fam) + r"}"
                     + _citep(fcites.get(fam, "")) + r"}"])
        for r in by_fam[fam]:
            nm = trim_family(r[1], fam)
            line = [latex_size_label(_tex_escape(nm), nm) + _citep(r[5])]
            if rankme:
                line.append(rcell(r))
            line += [cell(r, t) for t, _ in TASKS]
            if show_n:
                line.append(str(max(c[2] for c in r[4].values())))
            body.append(line)
    if floor:
        body.append([r"\addlinespace \multicolumn{" + str(ncol)
                     + r"}{l}{\emph{Untrained floor}}"])
        line = [_tex_escape(FLOOR_LABEL)]
        if rankme:
            line.append(rcell(floor))
        line += [cell(floor, t) for t, _ in TASKS]
        if show_n:
            line.append(str(max(c[2] for c in floor[4].values())))
        body.append(line)

    hdr = ([""] + (["RankMe"] if rankme else [])
           + [h for _, h in TASKS] + (["n"] if show_n else []))
    return (r"% requires \usepackage[normalem]{ulem} for \sout" + "\n"
            + render_latex_table(
                body, header_rows=[hdr],
                col_spec="l" + "c" * (n_rank + len(TASKS)
                                      + (1 if show_n else 0)),
                caption=caption, label=label if caption else None,
                caption_above=True,
                # ``--size none`` means NO size macro -- body text. Four
                # columns fit at full size, and \footnotesize on a table that
                # does not need it just makes it harder to read.
                small=bool(size),
                size=size or None,
                # Likewise a tabcolsep of 0 or less emits nothing and leaves
                # LaTeX's own 6pt alone.
                pre=([r"\setlength{\tabcolsep}{" + f"{tabcolsep:g}" + "pt}"]
                     if tabcolsep > 0 else []),
            ))


def render_markdown(rows, floor, *, show_n, rankme=None):
    hdr = (["model"] + (["RankMe"] if rankme else [])
           + [h for _, h in TASKS] + (["n"] if show_n else []))
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "---|" * len(hdr)]

    def cell(r, task):
        probe, h, _ = r[4][task]
        if math.isnan(probe):
            return "--"
        return _fmt(probe) + (f" ({h:.4f})" if h is not None
                              and not math.isnan(h) else "")
    modal_d = rankme_modal_width(rankme, rows, floor) if rankme else None
    for r in rows + ([floor] if floor else []):
        line = [r[1]]
        if rankme:
            got = rankme_for(rankme, r[0])
            if got is None:
                line.append("--")
            else:
                mu, se, n, widths = got
                mark = ("*" if modal_d is not None and widths
                        and modal_d not in widths else "")
                # The markdown view is the working one -- it carries the SE
                # and the month count the LaTeX column has no room for.
                line.append(f"{mu:.2f}{mark} +/-"
                            f" {se:.2f} ({n})" if se is not None
                            else f"{mu:.2f}{mark} (n={n})")
        line += [cell(r, t) for t, _ in TASKS]
        if show_n:
            line.append(str(max(c[2] for c in r[4].values())))
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out)


def main():
    global RESULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--alpha", type=float, default=10.0,
                   help="ridge alpha to read; the sweep fits 1/10/100")
    p.add_argument("--min-months", type=int, default=1,
                   help="drop rows with fewer scored eval months than this")
    p.add_argument("--fair", action="store_true",
                   help="restrict every row to the months all rows share "
                        "(off while the sweep fills in)")
    p.add_argument("--all-months", dest="floor_months", action="store_false",
                   help="render every scored month, including those the "
                        "untrained floor has not reached. The floor row is "
                        "then held back rather than struck across months it "
                        "cannot judge -- see comparable_months().")
    p.add_argument("--latex", action="store_true")
    # AUTOMATIC BY DEFAULT. The column answers "are these rows comparable?",
    # and once every row is on the same panel it answers it the same way in
    # every cell -- a column of one repeated number is furniture. It comes
    # back on its own the moment a row lands on a different month set, which
    # is exactly when the reader needs it. Either flag pins it.
    p.add_argument("--no-n", dest="show_n", action="store_const", const=False,
                   default=None, help="force the month-count column off")
    p.add_argument("--show-n", dest="show_n", action="store_const",
                   const=True, help="force the month-count column on even "
                                    "when every row shares a month count")
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default="tab:probe_fit")
    p.add_argument("--size", default="",
                   help="LaTeX size macro without the backslash (small, "
                        "footnotesize, scriptsize). Empty (the default) "
                        "emits none: four columns fit at body size.")
    p.add_argument("--tabcolsep", type=float, default=0.0,
                   help="column gap in pt; 0 (the default) leaves LaTeX's own")
    p.add_argument("--out", default=None)
    # OFF BY DEFAULT since 2026-09-17. RankMe is not floor-relative and nothing
    # in the column was ever struck or bold, so it sat in a table whose every
    # other number is read against the Random ViT floor and invited being read
    # the same way -- which is backwards here: on this panel RankMe is
    # significantly NEGATIVELY correlated with probe IC (return -0.66,
    # vol -0.51, spread -0.52). It also cost the caption a whole second
    # sentence to disclaim. The numbers are not lost: scripts/eval/rankme.py
    # still writes them and plots/core/rankme_table.py still renders them as a
    # table of their own, where they are not read against an IC floor.
    p.add_argument("--rankme", default=None,
                   help=f"rankme.py json for the RankMe column; passing this "
                        f"implies --with-rankme "
                        f"(default file: {RANKME_JSON.name})")
    p.add_argument("--with-rankme", dest="rankme_on", action="store_true",
                   default=False,
                   help="add the RankMe column (off by default: it is not "
                        "floor-relative, so it does not belong beside columns "
                        "that are)")
    a = p.parse_args()

    RESULTS = Path(a.results)

    ics, ns, nfiles, unknown = load_results(a.alpha)
    if not ics:
        print(f"no results yet under {RESULTS} "
              f"({nfiles} json file(s) seen, alpha={a.alpha:g})",
              file=sys.stderr)
        return 1
    if unknown:
        print(f"WARNING: {len(unknown)} checkpoint(s) in the results are not "
              f"in any manifest index and were skipped: "
              f"{', '.join(unknown[:6])}", file=sys.stderr)

    head = load_head()
    # THE FLOOR'S MONTHS ARE THE DEFAULT PANEL. Restricting here rather than
    # in the renderer means the month counts, the caption claim check and the
    # bolding all rest on the same set the floor is read on.
    restrict = comparable_months(ics, TASKS) if a.floor_months else None
    if restrict is not None:
        sizes = {t: len(v) for t, v in restrict.items()}
        if not any(sizes.values()):
            print("  no month has BOTH a floor and a method result yet; "
                  "falling back to every scored month (--all-months)",
                  file=sys.stderr)
            restrict = None
        else:
            print(f"  floor-covered panel: "
                  + ", ".join(f"{t.replace('_900','')} {n}"
                              for t, n in sizes.items()) + " month(s)",
                  file=sys.stderr)
    rows = build_rows(ics, ns, head, min_months=a.min_months, fair=a.fair,
                      restrict=restrict)
    shared = None
    if a.fair:
        mb = {k: {m for t in ics[k].values() for m in t} for k in ics}
        shared = set.intersection(*mb.values()) if mb else None
    # The per-task month count the METHOD rows rest on -- the bar the floor
    # has to clear before it is allowed to strike anything.
    method_months = {}
    for r in rows:
        for task, c in r[4].items():
            method_months[task] = max(method_months.get(task, 0), c[2])
    floor = floor_row(ics, min_months=a.min_months, shared=shared,
                      method_months=method_months, restrict=restrict)

    allns = [c[2] for r in rows for c in r[4].values()]
    poolsz = [n for d in ns.values() for n in d.values()]
    print(f"# {len(rows)} method row(s) from {nfiles} result file(s); "
          f"months per row {min(allns)}-{max(allns)}; "
          f"fit pool {min(poolsz):,d}-{max(poolsz):,d} rows; alpha {a.alpha:g}"
          + ("; FAIR months" if a.fair else "")
          + ("; floor-covered months only" if restrict else "; ALL months"),
          file=sys.stderr)

    check_caption_claims(rows, floor)

    # An explicit --rankme path is itself a request for the column.
    rankme_on = a.rankme_on or a.rankme is not None
    rankme = load_rankme(a.rankme) if rankme_on else {}
    if rankme_on and not rankme:
        print(f"  no RankMe json at {a.rankme or RANKME_JSON} -- column "
              f"omitted (scripts/eval/rankme.py --json writes it)",
              file=sys.stderr)
    elif rankme:
        check_rankme_months(rows, floor, rankme)

    # The n a row would print, over the rows the table actually shows.
    ns_seen = sorted({max(c[2] for c in r[4].values())
                      for r in list(rows)
                      + ([floor] if floor else [])})
    show_n = a.show_n if a.show_n is not None else len(ns_seen) > 1
    if a.show_n is None:
        print(f"  month counts {ns_seen} -- n column "
              f"{'kept' if show_n else 'dropped'} "
              f"(--show-n / --no-n to pin it)", file=sys.stderr)

    # The dagger clause is emitted only if a dagger is: an off-width arm is a
    # property of the roster, and explaining a mark the table does not carry
    # is how a caption starts drifting from its table.
    modal_d = rankme_modal_width(rankme, rows, floor) if rankme else None
    dagger = bool(rankme) and any(
        (got := rankme_for(rankme, r[0])) and modal_d is not None
        and got[3] and modal_d not in got[3]
        for r in list(rows) + ([floor] if floor else []))
    caption = a.caption or default_caption(
        n_months=ns_seen[0] if len(ns_seen) == 1 else None,
        rankme=bool(rankme), dagger=dagger)

    txt = (render_latex(rows, floor, show_n=show_n, rankme=rankme,
                        caption=caption,
                        label=a.label, size=a.size, tabcolsep=a.tabcolsep)
           if a.latex else render_markdown(rows, floor, show_n=show_n,
                                            rankme=rankme))
    if a.out:
        Path(a.out).write_text(txt + "\n")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
