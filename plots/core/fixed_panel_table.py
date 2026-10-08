"""All-models latent-structure table from the fixed_panel_P3S2 JSONs.

Two renderers over the same numbers:

MARKDOWN (default) — one row per model, T1-T4 each shown as top-1 ratio
over chance / mean-rank percentile, with an n column: models are pooled
over the months they have, which differ while a wave is still filling.
Bold = top two per column (ratio
and percentile ranked independently); strikethrough = worse than the
Random ViT row (ratio below / percentile above it).

LATEX (``--latex``) — the paper table: six columns, T1-T4 and the two
factor columns F1/F2, one row per METHOD, grouped by family (LeJEPA /
Supervised / SSL) with the Random ViT floor in its own block at the bottom
(it is the evidential floor, not a method being compared). Each of the four
TASK cells holds BOTH readouts, stacked: the top-1 ratio over chance above,
the target's mean-rank percentile below (``--top1`` and ``--rank`` swap
either for its raw form). F1 and F2 have one readout each. In the LaTeX,
green (and bold) is the top N_BOLD trained methods per column -- each half
ranked on its own -- and anything worse than the floor is RED, which needs
``\\usepackage{xcolor}`` (the emitted .tex says so on line 1). The markdown
still strikes. No t statistics unless ``--with-t``.

It emits NO caption and no column descriptions: the paper's caption is
written by hand and a generated one would fight it. ``--caption`` supplies
one if you want it, and ``--label`` rides along with it. The per-method n
is markdown-only, so read that before writing the caption -- while a wave
is still filling, the rows do not all rest on the same number of months.

Rows come from industry_nn_sweep.MODEL_ORDER, so the table follows the
registry rather than a second hand-maintained list that can drift from it.
The LaTeX labels and family grouping come from task_rank_corr.ROSTER, the
existing paper-facing roster, for the same reason: a method must not
answer to a third name here. A ``<x>_6mo`` key takes the label and family
of its ``<x>_final`` twin -- same method, a different training budget --
so the six-month wave lands in the LeJEPA and SSL blocks rather than in an
"Other" heap, and the span is stated once in the caption instead of being
suffixed onto fourteen labels.

``--tags`` names which result files to read, MERGED into one table -- both
the fixed-panel JSONs (fixed_panel/) and the two factor JSONs (factors/)
under each tag. One wave per tag is the normal shape now, so the REPORTED
TABLE IS THE DEFAULT MERGE (DEFAULT_TAGS) and a bare run is the whole thing:

    uv run python plots/core/fixed_panel_table.py
    uv run python plots/core/fixed_panel_table.py --latex \
        --out plots/core/fixed_panel_table.tex

Later tags win a repeated key; the shared `random` floor is written by the
same seeded model on the same panel, so a collision is bit-identical.
A tag that carries no floor row simply renders without the strike marks.

Pass ``--tags`` to read something else -- one wave on its own, the archived
campaign, or ``--tags ""`` for the untagged file a bare run_eval.sh writes.
An explicitly named tag with no result file is an error; one of the default
roster that has not been produced yet is skipped with a note.
"""
import argparse
import json
import math
import sys
from pathlib import Path

# MOVED UP OUT OF fixed_panel/ 2026-09-15. It renders the WHOLE latent table --
# the four fixed-panel tasks AND the two factor-structure columns -- so living
# inside one of the two stages' directories had it importing its sibling stage
# through `..`. HERE is now plots/latent_eval and both stages are below it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import FACTORS, PANELS  # noqa: E402

# ROSTER, FAMILY_CITE and latex_size_label come from task_rank_corr.
TASK_CORR = HERE.parent / "task_corr"

sys.path.insert(0, str(PANELS))

from industry_nn_sweep import MODEL_ORDER, MODEL_SPECS  # noqa: E402

METRICS = ["metric1", "metric2", "metric3", "metric4"]
HDRS = ["T1 pooled NN", "T2 day centroid", "T3 own-firm centroid",
        "T4 partner firm centroid"]

# The four CLEAN_TITLES of metric_diagram.py, verbatim — a talk shows the
# diagram and then this table, so the column must not change its name in
# between. Stacked with \shortstack (core LaTeX, no extra package) to keep
# the columns narrow enough for a two-column page.
LATEX_HDRS = [("T1", "Partner", "Matched View"), ("T2", "Own-Day", "Centroid"),
              ("T3", "Own-Firm", "Centroid"), ("T4", "Partner Firm", "Centroid")]

FLOOR = "random"
# How many trained methods per column get the bold. One place, both renderers.
N_BOLD = 3

# WHAT A BARE RUN RENDERS. The reported table is no longer one run's output:
# each wave is scored under its own tag and merged here, so the default is the
# roster rather than the untagged file.
#   _meanpool  the supervised span specialists, the frozen TSFM layers and the
#              random-init floor, all re-read at the mean (2026-09-14)
#   _6mo       the SSL and LeJEPA arms retrained on the supervised budget
#   _multi     the full-history supervised multihead trunk
#   _cmean     the frozen TSFMs again, nine channels AVERAGED (the prediction
#              evals' readout; plots/latent_eval/run_cmean_eval.sh) -- the
#              TSFM rows this table reports (TSFM_ROWS)
# A DEFAULT TAG WHOSE FILE IS NOT THERE YET IS SKIPPED, with a note -- a wave
# still running must not stop the other two from rendering. A tag named
# EXPLICITLY is still an error if it is missing: that one was asked for.
DEFAULT_TAGS = ("_meanpool", "_6mo", "_multi", "_cmean")

# The two factor-structure stages, read by the SAME tags as the panel file.
# They live one directory over and are keyed by SERIES name, which for every
# manifest-resolved model IS the registry key -- except the floor, which is
# five random-init seeds here and one in-process model there.
FLOOR_SEEDS = [f"randvit_s{i}" for i in range(5)]
# The figure's left panel is the mean over the four TOTAL loadings; r2_k is a
# different quantity (fit quality, not exposure) and is not averaged in.
DECODE_COLS = ["tot1", "tot2", "tot3", "tot4"]
FAC_HDRS = ["mean decode r", "factor space captured"]

# Family block order for the LaTeX table; ROSTER supplies each method's
# family, these are the headings and the order they print in. Supervised
# leads: it is the reference the other two families are read against.
FAMILY_ORDER = ["Supervised", "LeJEPA", "SSL"]

# THE FROZEN TSFMs, ONE ROW EACH, AT THEIR LAST LAYER. Every other arm in this
# table is read at its output layer with nothing tuned per model or per task,
# so a TSFM gets the same: its last hidden state, not the layer a sweep would
# pick. The sweep itself is plots/tsfm_layers/latent_sweep.py, whose two
# figures the block heading points to. They come from the _cmean wave (the
# latent protocol's mean over valid patches, the nine per-channel states
# AVERAGED into one d_model vector -- the prediction evals' readout), are
# struck against the floor like every row, and compete for the bold like
# every row: the bold is the top three methods in a column, frozen TSFMs
# included, and only the untrained floor is kept out of it.
TSFM_BLOCK = "Frozen TSFMs"
TSFM_ROWS = [("tsfm_chronos2_cmean_l12", "Chronos-2"),
             ("tsfm_kronos_cmean_l12", "Kronos"),
             ("tsfm_timesfm3_cmean_l20", "TimesFM 3.0")]
TSFM_FIG_REF = "fig:tsfm_latent_sweep"
# Their RankMe, every layer, from the same ff_fullday_cache embeddings
# (scripts/eval/rankme_fullday.py --tsfm). Only the three TSFM_ROWS keys are
# merged into the column: the file also holds 44 other layers. Each TSFM is
# d_model wide (768 / 832 / 1280), never 384, which is why RankMe has its own
# table (plots/latent_eval/rankme/rankme_table.py) that prints the width.
TSFM_RANKME = HERE.parent / "tsfm_layers" / "rankme_tsfm.json"


def _tex_escape(s):
    """style.tex_escape, imported lazily so the markdown path stays light."""
    from style import tex_escape
    return tex_escape(s)


def load_geo(tags, required=True):
    """Merged {model_key: {metric: stats}} over the named result files.

    Later tags win on a repeated key. The shared rows (random) are written
    by the same seeded in-process model on the same panel, so a collision is
    bit-identical rather than a choice.

    ``required=False`` (the default roster) skips a tag with no file yet and
    says so; with an explicitly named tag a missing file is an error.
    """
    geo, seen = {}, []
    for tag in tags:
        f = PANELS / f"fixed_panel_P3S2{tag}.json"
        if not f.is_file():
            if required:
                raise SystemExit(f"no such result file: {f}")
            print(f"note: no {f.name} yet — skipping tag {tag!r}",
                  file=sys.stderr)
            continue
        geo.update(json.load(open(f))["models"])
        seen.append(tag)
    if not geo:
        raise SystemExit(
            f"none of the tags {list(tags)} has a result file in {PANELS}")
    return geo, seen


def _mean(xs):
    xs = [x for x in xs if not math.isnan(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def load_factors(tags):
    """{series: {"r", "rhobar", "n_r", "n_rho"}} over the named factor files.

    Two numbers per model, one from each factor stage:

      * ``r`` -- decode_loadings' out-of-sample cross-firm Pearson r for a
        ridge from the month-mean embedding to the stock's loadings on total
        factors 1-4, averaged over the four loadings and then over months.
        Can a stock's factor exposure be read off its embedding.
      * ``rhobar`` -- subspace_alignment's sum(rho^2)/K_hat between the top-10
        PCs of the ticker-centroid embeddings and the estimated loading space:
        how much of the factor space the embedding geometry spans.

    Both average over the months a SERIES actually has, not over the panel --
    an arm still filling is short some months in the embedding cache too -- so
    each carries its own n and the caller checks those against the panel's.

    Missing files are not an error: a tag whose factor stages have not run
    yet simply contributes no rows, and those cells print as a dash.
    """
    fac: dict = {}
    for tag in tags:
        f = FACTORS / f"decode_loadings{tag}.json"
        if f.is_file():
            d = json.load(open(f))
            used = d.get("months_used", {})
            for key in d["series"]:
                cell = d["results"].get(f"{key}|emb")
                if not cell:
                    continue
                fac.setdefault(key, {})["r"] = _mean(
                    [_mean(cell[t]) for t in DECODE_COLS])
                fac[key]["n_r"] = len(used.get(key, cell[DECODE_COLS[0]]))
        f = FACTORS / f"subspace_alignment{tag}.json"
        if f.is_file():
            d = json.load(open(f))
            used = d.get("months_used", {})
            for key in d["series"]:
                cell = d["results"].get(key)
                if not cell or not cell["rhobar"]:
                    continue
                fac.setdefault(key, {})["rhobar"] = _mean(cell["rhobar"])
                fac[key]["n_rho"] = len(used.get(key, cell["rhobar"]))

    # THE FLOOR IS FIVE SEEDS HERE AND ONE MODEL IN THE PANEL FILE. The panel's
    # `random` is one seeded in-process ViT; the factor stages score the five
    # randvit seeds instead (they need a cached embedding, which `random` has
    # none of). Average the seeds into the floor row, which is what
    # factor_summary already reports as "Random ViT (5 seeds)".
    seeds = [fac[k] for k in FLOOR_SEEDS if k in fac]
    if seeds:
        fac[FLOOR] = {
            "r": _mean([d["r"] for d in seeds if "r" in d]),
            "rhobar": _mean([d["rhobar"] for d in seeds if "rhobar" in d]),
            "n_r": max([d.get("n_r", 0) for d in seeds]),
            "n_rho": max([d.get("n_rho", 0) for d in seeds]),
        }
    return fac


def n_months(cell):
    """Eval months this cell actually observed.

    NOT ``len(month_rates)``. A model with no checkpoint for a month is
    recorded as NaN rather than omitted (stage 1 keeps the row so the month
    log shows which arm was short), so the series carries months the model
    never ran. Count the real ones. ``n_months`` written by _repool agrees
    with this; recomputing here keeps the column right for a cell that never
    went through a merge.
    """
    return sum(1 for v in cell.get("month_rates", {}).values()
               if not math.isnan(float(v)))


def select_rows(geo, tags):
    """(key, registry label) for every MODEL_ORDER entry the run scored.

    A model missing a freshly-added metric is skipped rather than crashing,
    and so is one with no month it actually observed -- an arm whose
    checkpoints have not landed yet is absent from the table rather than a
    row of NaN. Everything with at least one month is IN, at whatever n it
    has; the n column is what tells the two apart.
    """
    rows = [(k, MODEL_SPECS[k]["label"]) for k in MODEL_ORDER
            if k in geo and all(mk in geo[k] for mk in METRICS)
            and n_months(geo[k]["metric1"])]
    if not rows:
        raise SystemExit(
            f"none of MODEL_ORDER is in {tags} — pass the tags whose "
            "runs produced the models you want tabulated")
    return rows


def render_markdown(geo, rows, fac=None):
    ratio = {mk: {k: geo[k][mk]["rate"] / geo[k][mk]["chance"]
                  for k, _ in rows} for mk in METRICS}
    pctile = {mk: {k: geo[k][mk]["mean_pctile"] for k, _ in rows}
              for mk in METRICS}
    top_r = {mk: set(sorted(ratio[mk], key=lambda k: -ratio[mk][k])[:N_BOLD])
             for mk in METRICS}
    top_p = {mk: set(sorted(pctile[mk], key=lambda k: pctile[mk][k])[:N_BOLD])
             for mk in METRICS}

    # n VARIES PER ROW and the header must not pretend otherwise: an arm
    # still filling is pooled over the months it has, so its cells are a
    # mean over fewer months than the row above it and the two are not
    # equally resolved. One n per model, in its own column.
    # A TAG NEED NOT CARRY THE FLOOR. Re-scoring one arm under its own tag is
    # a normal thing to do and the floor is already in the tag it was built
    # with, so the strike marks are simply omitted rather than KeyError'd when
    # `random` is not among the rows -- merge that tag in to get them back.
    has_floor = any(k == FLOOR for k, _ in rows)
    n_mo = {k: n_months(geo[k]["metric1"]) for k, _ in rows}
    span = (f"{min(n_mo.values())}-{max(n_mo.values())}"
            if len(set(n_mo.values())) > 1 else f"{max(n_mo.values())}")
    # NO t IN A CELL. Two numbers per column is what a reader can hold; the
    # t belonged to the LaTeX table, where a caption says what it tests. The
    # rank cell's t was never the ratio's anyway (rank_t vs t), so a cell
    # carrying one of them beside both readouts was quoting a statistic at
    # half its row. --latex has none either unless --with-t asks, and then
    # each half takes its own.
    # The two factor columns are on the same footing as T1-T4: higher is
    # better, bold the top two, strike what the floor beats. They come from a
    # different pair of stages, so a tag whose factor run has not happened
    # leaves them out entirely rather than printing an empty column.
    fac = fac or {}
    fkeys = [f for f in ("r", "rhobar")
             if any(f in fac.get(k, {}) for k, _ in rows)]
    fhdr = [h for h, f in zip(FAC_HDRS, ("r", "rhobar")) if f in fkeys]
    top_f = {f: set(sorted((k for k, _ in rows if f in fac.get(k, {})),
                           key=lambda k: -fac[k][f])[:N_BOLD]) for f in fkeys}

    out = [f"top-1 ratio over chance / mean-rank pctile, n={span} months "
           f"per model (n column); bold top-{N_BOLD} per column, "
           "struck < Random ViT",
           "| model | n | " + " | ".join(HDRS + fhdr) + " |",
           "|" + "---|" * (len(HDRS) + len(fhdr) + 2)]
    for k, lbl in rows:
        cells = [str(n_mo[k])]
        for mk in METRICS:
            r_txt = f"{ratio[mk][k]:.2f}x"
            p_txt = f"{pctile[mk][k]:.1%}"
            if has_floor and k != FLOOR and ratio[mk][k] < ratio[mk][FLOOR]:
                r_txt = f"~~{r_txt}~~"
            if has_floor and k != FLOOR and pctile[mk][k] > pctile[mk][FLOOR]:
                p_txt = f"~~{p_txt}~~"
            if k in top_r[mk]:
                r_txt = f"**{r_txt}**"
            if k in top_p[mk]:
                p_txt = f"**{p_txt}**"
            cells.append(f"{r_txt} / {p_txt}")
        for f in fkeys:
            v = fac.get(k, {}).get(f)
            if v is None or math.isnan(v):
                cells.append("--")
                continue
            txt = f"{v:+.3f}" if f == "r" else f"{v:.3f}"
            if k != FLOOR and FLOOR in fac and v < fac[FLOOR][f]:
                txt = f"~~{txt}~~"
            if k in top_f[f]:
                txt = f"**{txt}**"
            cells.append(txt)
        out.append(f"| {lbl} | " + " | ".join(cells) + " |")

    # THE TWO n's MUST AGREE. A model's factor months come from the embedding
    # cache and its panel months from the manifest; they are the same coverage
    # and a divergence means one stage saw checkpoints the other did not. Say
    # so under the table rather than let one n stand for both.
    odd = [(k, n_mo[k], fac[k].get("n_r"), fac[k].get("n_rho"))
           for k, _ in rows if k in fac and k != FLOOR
           and {n_mo[k]} != {fac[k].get("n_r"), fac[k].get("n_rho")}]
    for k, n_p, n_r, n_rho in odd:
        out.append(f"<!-- {k}: panel n={n_p} but decode n={n_r}, "
                   f"subspace n={n_rho} — the n column is the panel's -->")
    return "\n".join(out)


# RANKME, FROM THE LATENT SUITE'S OWN EMBEDDINGS. Written by
# scripts/eval/rankme_fullday.py off ff_fullday_cache, which is mean-pooled for
# EVERY arm including the floor -- deliberately not plots/metrics/rankme_6mo.json,
# whose supervised arms and floor are read at the LAST token because it is built
# from the predictive panel. Mixing the two would put spectra taken at two
# readouts in one column.
RANKME_JSON = HERE / "rankme_latent.json"


def load_rankme(path=None):
    """``{series: (mean, se, n_months, widths)}`` for the RankMe column.

    SEEDS WITHIN A MONTH, THEN MONTHS. The floor is five randvit seeds; pooling
    all 5*N (seed, month) cells in one pass would weight a month carrying five
    seeds the same as one carrying two. Collapse the seeds per month first, then
    average those monthly values -- the order the floor is built with everywhere
    else in this file.

    Returns ``{}`` when the json is absent, so the column simply does not render.
    """
    import statistics as _st
    from collections import defaultdict

    p = Path(path) if path else RANKME_JSON
    if not p.exists():
        return {}
    per_month = defaultdict(lambda: defaultdict(list))   # series -> month -> [v]
    widths = defaultdict(set)
    for r in json.loads(p.read_text()):
        s = r["series"]
        key = FLOOR if s.startswith("randvit_s") else s
        per_month[key][r["month"]].append(float(r["rankme"]))
        widths[key].add(int(r["d"]))
    out = {}
    for s, months in per_month.items():
        vals = [_st.mean(v) for _, v in sorted(months.items())]
        se = (_st.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out[s] = (_st.mean(vals), se, len(vals), widths[s])
    return out


def rankme_modal_width(rankme):
    """The embedding width most arms share -- what the column's scale means."""
    from collections import Counter

    c = Counter(w for _, _, _, ws in rankme.values() for w in ws)
    return c.most_common(1)[0][0] if c else None


def latex_roster():
    """{lat key: (paper label, family)} from task_rank_corr.ROSTER.

    Imported lazily: the markdown path must not pull scipy in, and a tag
    whose models are outside the reported roster still renders (those rows
    fall back to the registry label under an 'Other' block).
    """
    sys.path.insert(0, str(TASK_CORR))
    from task_rank_corr import ROSTER  # noqa: E402
    r = {m["lat"]: (m["label"], m["fam"]) for m in ROSTER}
    # THE SIX-MONTH WAVE ANSWERS TO THE SAME NAMES. `pair_rrc_6mo` and
    # `pair_rrc_final` are one method on two training budgets, and ROSTER is
    # keyed by the one-month key because that is the generation it was written
    # for. Mapping the suffix onto its twin is what puts these rows in the
    # LeJEPA and SSL blocks instead of an 'Other' heap -- and it keeps the
    # paper label in ONE place, rather than adding a third spelling here.
    # The span is not in the label because the whole table is one budget: every
    # row, supervised included, is a six-month span (see the caption).
    for key, (lbl, fam) in list(r.items()):
        if key.endswith("_final"):
            r.setdefault(key[:-len("_final")] + "_6mo", (lbl, fam))
    return r


def latex_cites():
    """({lat key: cite key}, {family: cite key}) from task_rank_corr.

    LATEX ONLY. A ``\\citep`` in the shared label would land in the markdown
    and in every figure axis that reads ROSTER, so the citation is attached
    here, at the point of rendering, instead of being baked into the name.
    """
    sys.path.insert(0, str(TASK_CORR))
    from task_rank_corr import ROSTER, FAMILY_CITE  # noqa: E402
    r = {m["lat"]: m.get("cite", "") for m in ROSTER}
    for key, c in list(r.items()):
        if key.endswith("_final"):
            r.setdefault(key[:-len("_final")] + "_6mo", c)
    return r, dict(FAMILY_CITE)


def _citep(cite):
    return (r"~\citep{" + cite + "}") if cite else ""


def trim_family(label, fam):
    """``label`` with the family name dropped, for a row under a family block.

    The block heading already says LeJEPA or Supervised, so a row that repeats
    it spends the widest column in the table saying it twice -- and the part
    that actually distinguishes the rows ("+noise", "cross-stock ind") is the
    part pushed to the right. The markdown keeps the full names: it has no
    family blocks, so there the prefix is the only thing carrying the family.

    A label that does not start with its family is left alone (every SSL row),
    and so is one that would be emptied by the trim.
    """
    if not label.lower().startswith(fam.lower()):
        return label
    rest = label[len(fam):].strip()
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()
    if not rest:
        return label
    # "+time warp" must keep its sign; "crops" reads better capitalized.
    return rest[0].upper() + rest[1:] if rest[0].isalpha() else rest


# ONE FLOAT, ONE CAPTION. The diagrams and the table were two floats with two
# captions, and the table's had to cross-reference the figure's to say what
# T1-T4 and F1/F2 even were. They are one float now: the two diagrams sit above
# the tabular and a single caption defines the tasks AND reads the table. Pass
# --no-figure to drop the diagrams (the caption then still reads the table, but
# nothing defines the tasks, so only do that where the paper defines them).
#
# PLACEMENT IS [tp]. Two full-width diagrams plus a 19-row table is taller than
# most columns, so a plain [t] can drift pages away from the text or overflow.
# [tp] lets LaTeX give it a float page, which is what "all on one page" needs.
#
# The \includegraphics paths are relative to the PAPER's tree (figures/), not
# to this repo. The only copies of these two PDFs in the repo are under
# plots/latent_eval/_archive/pre_meanpool_readout/, which is the retired
# readout -- do not regenerate the paper's copies from there.
TASKS_GRAPHICS = (
    r"\includegraphics[width=1\linewidth]"
    r"{figures/fixed_panel_metric_diagram_clean.pdf}" + "\n"
    r"\includegraphics[width=1\linewidth]"
    r"{figures/factor_diagram_clean.pdf}" + "\n"
    r"\vspace{0.5em}"
)

# THE CAPTION LIVES HERE, not in regen.sh. It used to be a shell variable the
# driver passed through --caption, which meant the one description of what
# these numbers mean sat outside the file that computes them and could not be
# checked against them.
_CAPTION_BASE = (
    r"Evaluating the organization of learned market state. \textbf{Top:} We "
    r"embed observations from six firms across three randomly selected "
    r"industries within the same evaluation month and test whether proximity "
    r"in representation space reflects shared day, firm, and industry "
    r"structure, scored as T1--T4. \textbf{Bottom:} We estimate statistical "
    r"factors from realized returns and measure both how well individual "
    r"factor loadings can be decoded from the embeddings (F1) and how much of "
    r"the factor subspace is captured by their principal directions (F2). "
    r"\textbf{Table:} T1--T4 give the top-1 rate as a multiple of chance and the mean "
    r"percentile rank of the target (lower is better), while F1 and F2 give a "
    r"single score for which higher is better; "
    r"\textcolor{green!45!black}{\textbf{green}} marks the top three trained "
    r"methods in a column, and \textcolor{red!75!black}{red} values are no "
    r"better than the untrained Random ViT."
)


def default_caption(rankme=False, dagger=False):
    """The caption, BUILT rather than concatenated.

    The RankMe sentence is a property of whether that column renders, so it is
    assembled here instead of being appended by the caller -- which is how the
    probe table's caption grew to a paragraph, one bolted-on clause at a time.
    """
    if not rankme:
        return _CAPTION_BASE
    two = (r" RankMe~\citep{garrido2023rankme} counts how many directions the "
           r"frozen embedding spends its variance on, read at the same mean "
           r"pooling as every other column, and is reported for reference "
           r"only---it is not floor-relative, so nothing in it is coloured")
    two += (r" ($\dagger$: a different embedding width, which bounds RankMe, so "
            r"not on the rest of the column's scale)." if dagger else ".")
    return _CAPTION_BASE + two


def render_latex(geo, rows, *, top1="ratio", rank="pctile", show_t=False,
                 caption=None, label=None, fac=None,
                 size="small", tabcolsep=4.0, figure=True, rankme=None,
                 tsfm=()):
    sys.path.insert(0, str(HERE.parent))
    from style import render_latex_tabular, tex_mark  # noqa: E402
    sys.path.insert(0, str(TASK_CORR))
    from task_rank_corr import latex_size_label  # noqa: E402

    roster = latex_roster()
    cites, fam_cites = latex_cites()
    ratio = {mk: {k: geo[k][mk]["rate"] / geo[k][mk]["chance"]
                  for k, _ in list(rows) + list(tsfm)} for mk in METRICS}

    # EACH TASK CELL CARRIES BOTH READOUTS. They answer different questions --
    # top-1 is "how often is the right one first", the rank is "where does it
    # sit when it is not" -- and a method can win one and lose the other (TF-C
    # on T1 is the case in point). One number per cell made that invisible and
    # left the choice to a flag nobody set twice.
    #
    # HIGHER IS BETTER for the top-1 half, LOWER for the rank half, so each
    # half gets its own green top-N and its own floor comparison.
    HALVES = ((top1, -1), (rank, +1))

    def value(k, mk, which):
        return {"ratio": ratio[mk][k],
                "rate": geo[k][mk]["rate"],
                "rank": geo[k][mk]["mean_rank"],
                "pctile": geo[k][mk]["mean_pctile"]}[which]

    # The floor is a reference level, not a method: it is excluded from the
    # bold-two so a task the floor happens to win cannot spend a bold on it.
    # The frozen TSFMs are in the contest too (see TSFM_ROWS).
    trained = [k for k, _ in rows if k != FLOOR] + [k for k, _ in tsfm]
    topN = {(mk, w): set(sorted(trained,
                                key=lambda k: g * value(k, mk, w))[:N_BOLD])
            for mk in METRICS for w, g in HALVES}

    def half(k, mk, which, sgn, size=""):
        v = value(k, mk, which)
        # THE UNIT MARK IS SET SMALL. Every task column carries two numbers,
        # so the width of a cell is set by its widest one and the \times / \%
        # glyphs cost a column of text width they carry no information in.
        # \scriptstyle, not \scriptsize: these are inside $...$ below, and a
        # text-size command in math mode either does nothing or needs \text{}.
        mul = r"{\scriptstyle\times}"
        pct = r"{\scriptstyle\%}"
        txt = {"ratio": f"{v:.2f}{mul}",
               "rate": f"{100 * v:.1f}{pct}",
               "rank": f"{v:.2f}",
               "pctile": f"{100 * v:.1f}{pct}"}[which]
        # BELOW THE FLOOR IS RED, the same rule as the markdown's strike -- a
        # reader should not have to decode a dagger to see that a number is
        # not evidence of anything. Red beats green (see style.tex_mark).
        txt = tex_mark(txt, best=k in topN[(mk, which)],
                       worse=(k != FLOOR and sgn * value(k, mk, which)
                              >= sgn * value(FLOOR, mk, which)))
        if show_t:
            # The two readouts carry DIFFERENT t: ``t`` tests the top-1 rate
            # against chance, ``rank_t`` the mean rank against the chance rank.
            # Each half takes its own or it would be quoting a statistic that
            # was not computed from the number beside it.
            t_key = "rank_t" if which in ("rank", "pctile") else "t"
            txt += r"{\scriptsize(" + f"{geo[k][mk][t_key]:+.1f}" + r")}"
        return f"{{{size} {txt}}}" if size else txt

    def fmt(k, mk):
        # INLINE "a/b", one line, both halves at the body size. This was a
        # \shortstack until 2026-09-15 -- stacking keeps a cell as wide as its
        # wider half rather than their sum, which is the narrower table -- but
        # a two-line cell in every one of four task columns reads as eight
        # rows of numbers instead of four, and the eye has to pair them up
        # again. The colours still apply to each half separately, so a half
        # below the floor is red on its own.
        return half(k, mk, top1, -1) + "/" + half(k, mk, rank, +1)

    # THE TWO FACTOR COLUMNS ride alongside T1-T4 but carry ONE number each:
    # decode r and rhobar have a single readout and higher is better for both,
    # so there is no second half to stack. Own green top-N, own red, no t.
    fac = fac or {}
    fkeys = [f for f in ("r", "rhobar")
             if any(f in fac.get(k, {}) for k, _ in rows)]
    topNf = {f: set(sorted((k for k in trained if f in fac.get(k, {})),
                           key=lambda k: -fac[k][f])[:N_BOLD]) for f in fkeys}

    def ffmt(k, f):
        v = fac.get(k, {}).get(f)
        if v is None or math.isnan(v):
            return "--"
        # NO SIGN ON r. It is a correlation whose sign is never in doubt here
        # -- every arm decodes loadings positively, floor included -- so the
        # "+" was a column of noise. A negative would still print its "-".
        return tex_mark(f"{v:.3f}", best=k in topNf[f],
                        worse=(k != FLOOR and FLOOR in fac
                               and v <= fac[FLOOR].get(f, float("nan"))))

    # NO n COLUMN. The markdown keeps n per row; the paper table is six
    # numbers wide and the hand-written caption carries the counts. Arms DO
    # cover different numbers of months while a wave is filling -- read the
    # markdown before writing that caption.

    n_cols = len(METRICS) + len(fkeys) + 1 + (1 if rankme else 0)
    body: list[list[str]] = []

    def group(name):
        # The family heading carries the family's own citation when it has one
        # (LeJEPA: five rows, one paper), so the cite is stated once instead of
        # five times down the name column.
        body.append([r"\addlinespace \multicolumn{" + str(n_cols)
                     + r"}{l}{\emph{" + _tex_escape(name) + r"}"
                     + _citep(fam_cites.get(name, "")) + r"}"])

    by_fam: dict[str, list] = {}
    # RANKME IS NOT FLOOR-RELATIVE, so no RankMe cell is ever struck or bold:
    # it is a property of the spectrum, not a score against the Random ViT, and
    # marking it as if it were is how a reader concludes high rank is good. On
    # this panel the opposite holds -- the four supervised arms have the
    # NARROWEST spectra and the floor one of the widest.
    modal_d = rankme_modal_width(rankme) if rankme else None

    def rcell(k):
        got = rankme.get(k) if rankme else None
        if not got:
            return ""
        mu, _se, _n, ws = got
        # A different width BOUNDS RankMe differently, so that cell is not on
        # the rest of the column's scale. Mark it rather than drop it.
        off = modal_d is not None and ws and modal_d not in ws
        return f"${mu:.1f}" + (r"^{\dagger}" if off else "") + "$"

    for k, reg_lbl in rows:
        if k == FLOOR:
            continue
        lbl, fam = roster.get(k, (reg_lbl, "Other"))
        by_fam.setdefault(fam, []).append((k, trim_family(lbl, fam)))
    for fam in FAMILY_ORDER + [f for f in by_fam if f not in FAMILY_ORDER]:
        if fam not in by_fam:
            continue
        group(fam)
        for k, lbl in by_fam[fam]:
            # ESCAPE THE NAME, THEN APPEND THE CITE. _tex_escape would turn
            # the \citep into \textbackslash{}citep and its underscores into
            # \_, i.e. print the macro instead of running it.
            body.append([latex_size_label(_tex_escape(lbl), lbl)
                         + _citep(cites.get(k, ""))]
                        + ([rcell(k)] if rankme else [])
                        + [fmt(k, mk) for mk in METRICS]
                        + [ffmt(k, f) for f in fkeys])
    if tsfm:
        # After the trained families, before the floor; the heading carries
        # the pointer to the layer-wise sweep (see TSFM_ROWS).
        body.append([r"\addlinespace \multicolumn{" + str(n_cols)
                     + r"}{l}{\emph{" + TSFM_BLOCK + r"} (last layer, channels "
                     r"averaged); see "
                     r"Figure~\ref{" + TSFM_FIG_REF
                     + r"} for the layer-wise sweep.}"])
        for k, lbl in tsfm:
            body.append([_tex_escape(lbl)]
                        + ([rcell(k) or "--"] if rankme else [])
                        + [fmt(k, mk) for mk in METRICS]
                        + [ffmt(k, f) for f in fkeys])
    if FLOOR in geo and any(k == FLOOR for k, _ in rows):
        group("Untrained floor")
        body.append([_tex_escape(MODEL_SPECS[FLOOR]["label"])]
                    + ([rcell(FLOOR)] if rankme else [])
                    + [fmt(FLOOR, mk) for mk in METRICS]
                    + [ffmt(FLOOR, f) for f in fkeys])

    # A GUARD, not a display input: nothing in the table quotes these any more
    # (the caption is hand-written), but a campaign whose models disagree about
    # the panel they were scored on is a broken table however it is rendered,
    # and this is the only place that would notice. Chance is a property of the
    # panel, not of the model -- except that T2's is estimated per model by
    # day-label permutation (see summarize.py), which is what makes the check
    # worth keeping.
    # n_candidates IS NOT FIXED GEOMETRY EITHER. T1 and T2 pool a whole
    # panel-month, so their candidate count is a mean over the months a model
    # was scored on (103.4, 20.7 on the full panel) -- two models on different
    # month sets differ for the same innocent reason chance does. Both fields
    # are therefore checked the same way: within a month set, never across one.
    for mk in METRICS:

        # A PER-MONTH LEVEL AVERAGED OVER A MODEL'S OWN MONTHS: two arms
        # pooled over DIFFERENT months differ for an innocent reason, and a
        # blanket refusal fires on every ragged panel. What is still an error
        # is two models on the SAME months disagreeing -- that would mean T2's
        # per-model permutation baseline had really diverged. Group by month
        # set; refuse inside a group, never across one.
        for field in ("chance", "n_candidates"):
            groups: dict = {}
            for k, _ in rows:
                ms = frozenset(geo[k][mk].get("month_rates", {}))
                groups.setdefault(ms, {})[k] = round(geo[k][mk][field], 3)
            for ms, vals in groups.items():
                if len(set(vals.values())) > 1:
                    raise SystemExit(
                        f"{mk}: models on the same {len(ms)} months disagree "
                        f"on {field} ({sorted(set(vals.values()))}) -- "
                        f"{sorted(vals)}")

    # BARE COLUMN NAMES, NO CAPTION. What each task is, what the cells hold and
    # what the marks mean are the caption's job, and the caption is written by
    # hand in the paper -- this file must not put a second description in the
    # way of it. Pass --caption to have one emitted; \label rides with it,
    # since a \label in a float with no \caption points at the wrong counter.
    hdr = [""] + (["RankMe"] if rankme else []) + [t for t, _, _ in LATEX_HDRS]
    hdr += [{"r": "F1", "rhobar": "F2"}[f] for f in fkeys]
    # THE FLOAT IS BUILT HERE, not by style.render_latex_table, because this
    # one is a compound float and that helper cannot express it: it hardcodes
    # [t], and it emits the size macro BEFORE the caption, which would set this
    # long caption at \footnotesize. The size has to scope the tabular only.
    #
    # CAPTION BELOW, because the diagrams are inside the float: a caption set
    # above them would say "Top:"/"Bottom:" before the reader has seen either.
    lines = [r"\begin{table}[tp]", r"\centering"]
    if figure:
        lines.append(TASKS_GRAPHICS)
    # SIZED TO FIT, because the cells went inline: a row is four "a/b" pairs
    # plus two factor numbers plus a row name as long as "Cross Stock, Same
    # Industry". \small with a 4pt column gap is what holds that inside
    # one text width; --size/--tabcolsep tune it without editing the .tex,
    # which is REGENERATED and would lose the edit. The group closes after
    # \end{tabular} so neither reaches the caption.
    lines.append("{" + ("\\" + size if size else r"\small"))
    lines.append(r"\setlength{\tabcolsep}{" + f"{tabcolsep:g}" + "pt}")
    lines.append(render_latex_tabular(
        body, header_rows=[hdr],
        col_spec="l" + "c" * (len(METRICS) + len(fkeys) + (1 if rankme else 0)),
    ))
    lines.append("}")
    if caption:
        lines += ["", r"\vspace{-0.5em}", ""]
        lines.append(r"\caption{" + caption + "}")
        # \label rides with the caption: a label in a float with no caption
        # points at the wrong counter.
        if label:
            lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return (r"% requires \usepackage{xcolor}" + "\n"
            + "\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="*", default=None,
                   help="result-file suffixes to merge; default is the "
                        f"reported roster {' '.join(DEFAULT_TAGS)} (a tag "
                        "with no file yet is skipped). Pass \"\" for the "
                        "untagged file a bare run_eval.sh writes.")
    p.add_argument("--latex", action="store_true",
                   help="render the paper LaTeX table instead of markdown")
    p.add_argument("--size", default="small",
                   help="LaTeX size macro for the table body, without the "
                        "backslash (small, footnotesize, scriptsize, tiny)")
    p.add_argument("--tabcolsep", type=float, default=4.0,
                   help="column gap in pt; lower packs the columns tighter")
    p.add_argument("--top1", default="ratio", choices=["ratio", "rate"],
                   help="LaTeX only: the TOP half of each task cell -- the "
                        "top-1 hit rate as a ratio over chance (default) or "
                        "the raw rate")
    p.add_argument("--rank", default="pctile", choices=["pctile", "rank"],
                   help="LaTeX only: the BOTTOM half -- the target's mean "
                        "rank as a percentile of the pool (default, and the "
                        "only one comparable across columns) or the raw rank")
    p.add_argument("--with-t", dest="show_t", action="store_true",
                   help="LaTeX only: add each half's t statistic (the top-1 "
                        "rate's t on top, the rank's rank_t below)")
    p.add_argument("--caption", default=None,
                   help="LaTeX only: caption text. Defaults to the merged "
                        "caption that both defines the six tasks and reads "
                        "the table; pass an empty string for none")
    p.add_argument("--label", default="tab:latent_results",
                   help="LaTeX only: \\label key, emitted only with --caption "
                        "(a label in a float with no caption points at the "
                        "wrong counter)")
    p.add_argument("--no-factors", action="store_true",
                   help="markdown only: drop the two factor-structure columns "
                        "(decode r and rhobar) even where their files exist")
    p.add_argument("--rankme", default=None,
                   help=f"rankme_fullday.py json for the RankMe column "
                        f"(default: {RANKME_JSON.name} beside this script)")
    p.add_argument("--with-rankme", dest="no_rankme", action="store_false",
                   default=True,
                   help="put the RankMe column back in the latent table (it "
                        "has its own appendix table now: "
                        "plots/latent_eval/rankme/rankme_table.py)")
    p.add_argument("--no-figure", dest="figure", action="store_false",
                   help="omit the Figure~\\ref{fig:tasks} block that defines "
                        "the six tasks; the table's caption references it, so "
                        "dropping it leaves a dangling \\ref")
    p.add_argument("--no-tsfm", dest="tsfm", action="store_false",
                   help="LaTeX only: omit the frozen-TSFM block")
    p.add_argument("--out", help="write to this file instead of stdout")
    args = p.parse_args()

    explicit = args.tags is not None
    tags = args.tags if explicit else list(DEFAULT_TAGS)
    geo, tags = load_geo(tags, required=explicit)
    rows = select_rows(geo, tags)
    fac = {} if args.no_factors else load_factors(tags)
    want_rk = not args.no_rankme
    rankme = load_rankme(args.rankme) if want_rk else {}
    tsfm = []
    if args.tsfm and rankme and TSFM_RANKME.is_file():
        trk = load_rankme(TSFM_RANKME)
        rankme.update({k: trk[k] for k, _ in TSFM_ROWS if k in trk})
    if args.tsfm:
        tsfm = [(k, l) for k, l in TSFM_ROWS
                if k in geo and all(mk in geo[k] for mk in METRICS)]
        missing = [k for k, _ in TSFM_ROWS if (k, dict(TSFM_ROWS)[k]) not in tsfm]
        if missing:
            print(f"  frozen-TSFM rows missing from {tags}: {missing}",
                  file=sys.stderr)
    _modal = rankme_modal_width(rankme) if rankme else None
    _dagger = bool(rankme) and any(
        ws and _modal is not None and _modal not in ws
        for _, _, _, ws in rankme.values())
    caption = args.caption if args.caption is not None else default_caption(
        rankme=bool(rankme), dagger=_dagger)
    if want_rk and not rankme:
        print(f"  no RankMe json at {args.rankme or RANKME_JSON} -- column "
              f"omitted (scripts/eval/rankme_fullday.py --json writes it)",
              file=sys.stderr)
    txt = (render_latex(geo, rows, top1=args.top1, rank=args.rank,
                        show_t=args.show_t, caption=caption,
                        label=args.label, fac=fac,
                        size=args.size, tabcolsep=args.tabcolsep,
                        figure=args.figure, rankme=rankme, tsfm=tsfm)
           if args.latex else render_markdown(geo, rows, fac))
    if args.out:
        Path(args.out).write_text(txt + "\n")
        print(f"wrote {args.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
