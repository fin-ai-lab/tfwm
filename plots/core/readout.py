"""Which token the predictive results were read at, and which rows to believe.

THE PROTOCOL. Prediction is scored at the LAST token for every arm; the latent
suite reads everything at the mean (panel_lib.LATENT_POOL). Until 2026-09-17
the probe-breadth sweep had neither: probe_fit_size.py loaded each checkpoint
with its own config, so the 14 SSL/LeJEPA arms came back MEAN-pooled while the
four supervised arms and the three random-init floor seeds were read at LAST --
and every struck SSL cell was an arm measured against a floor read at a
different token. xs_ic_eval.py:997 names that exact failure.

WHY A SELECTOR AND NOT JUST A GLOB. Results are keyed by (ckpt, n, alpha) and
nothing else, so a mean-pooled row and a last-pooled row for the same
checkpoint are indistinguishable. The two readers then disagree by accident:
probe_fit_table keeps the FIRST file at the largest n and probe_fit_breadth
keeps the LAST, and `sorted()` puts "pball-*" before "pblast-*" -- so the table
would have shown the old mean number while the figure showed the new last one,
with nothing raising. Rows are now stamped with `readout` by
probe_fit_size.cmd_reduce, and this module is the single rule for choosing.

UNSTAMPED ROWS PREDATE THE STAMP and carry the checkpoint's OWN training pool:
last for the supervised arms and the floor, mean for the SSL arms. The first
group is already at the predictive readout, so its rows are accepted -- that is
what LAST_BY_CONSTRUCTION names, and it is a fact about how those checkpoints
were trained, not a guess. The second group is NOT, so those rows are DROPPED
and the cell renders blank.

NO SILENT SUBSTITUTION. There is deliberately no fallback to "whatever readout
we happen to have": a mean-pooled number in a last-pooled column is a wrong
number that looks like a right one, and this table's whole marking scheme reads
every cell against the floor. A hole is recoverable; a plausible wrong value is
not. Series rendered blank are reported on stderr every run.
"""
PREDICT_READOUT = "last"

# CoST IS THE ONLY ARM THAT CANNOT BE RE-POOLED. Its encode() returns
# cat(trend[:, -1], season[:, -1]) taken straight off backbone.forward_patches(),
# so it never consults backbone.pool -- and it is ALREADY reading the last valid
# patch, which is the readout the predictive protocol asks for. It is therefore
# in LAST_BY_CONSTRUCTION below, not blanked.
#
# TS2Vec, TF-C and TimeMAE were briefly listed here and that was WRONG: setting
# only .backbone.pool moved TS2Vec and TF-C by 0.0, because their readouts live
# on a second module (.swa_backbone, .freq_backbone). Setting every
# sub-backbone moves them by 0.71 / 1.60 / 2.75. They are re-poolable and are
# in the sweep.
ENCODE_ONLY = ("cost_6mo",)


# TRAINED AT THE PREDICTIVE READOUT ALREADY, so their unstamped rows need no
# re-run: the four supervised arms train pool="last", and the random-init floor
# backbones are built at "last" (they are backbone.pt checkpoints whose pool
# comes from the cfg, which xs_ic_eval pins). Verified 2026-09-17 by loading
# every arm in manifest_all.index.json and reading backbone.pool.
LAST_BY_CONSTRUCTION = frozenset({
    "sup_return_w8", "sup_vol_w8", "sup_spread_w8", "sup_multi_w8",
    "randinit_s42", "randinit_s43", "randinit_s44",
    # CoST's published readout IS the last valid patch (see ENCODE_ONLY), so
    # its existing rows already satisfy the protocol and must not be blanked.
    "cost_6mo",
})


def pick(rows, series):
    """The rows to believe for one (series, month), and the readout they are at.

    ``rows`` is every result row found for that key, across every result file.
    Returns ``([], None)`` when nothing is at the predictive readout -- the
    caller renders that blank rather than substituting another readout.
    """
    stamped = [r for r in rows if r.get("readout") == PREDICT_READOUT]
    if stamped:
        return stamped, PREDICT_READOUT
    if series in LAST_BY_CONSTRUCTION:
        return [r for r in rows if not r.get("readout")], PREDICT_READOUT
    return [], None


def report(blank, out):
    """Print which series render blank for want of the predictive readout."""
    if not blank:
        return
    exempt = sorted(s for s in blank if s in ENCODE_ONLY)
    other = sorted(s for s in blank if s not in ENCODE_ONLY)
    if exempt:
        print(f"  readout: {len(exempt)} arm(s) BLANK and un-repoolable: "
              f"{', '.join(exempt)}", file=out)
    if other:
        print(f"  readout: {len(other)} series BLANK -- no result at "
              f"'{PREDICT_READOUT}' yet; re-run them with "
              f"probe_fit_size.py --pool {PREDICT_READOUT}: "
              f"{', '.join(other)}", file=out)
