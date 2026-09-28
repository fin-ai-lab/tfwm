"""Target discretization for the supervised binned head.

A :class:`Discretizer` carries everything needed to turn a raw forward target
into a TARGET DISTRIBUTION over ``k`` ordered bins. It is fitted once, from
calibration batches drawn off the training dataloader, and then applied per
batch: ``fit`` returns the object, the supervised model stores it, and
``.apply(y)`` labels every subsequent batch against the same edges.

EQUAL-COUNT BINS, AND WHY THERE IS NO ZERO BIN
----------------------------------------------
Bins are pure equal-count quantiles of the calibration sample. There is no
special bin for "the return was zero" and no 5bp threshold constant.

The scheme this replaces carved out a sticky zero bin (``|y| <= 5bp``) because
returns have a large atom at exactly zero. That fixed the atom but broke the
bin geometry: the zero bin held ~9% of rows at EVERY k, so at k=21 it was
roughly twice the width of its neighbours, and the bin INDEX stopped being a
linear function of the quantile. That matters because the head is read out as
an expected bin and scored by a rank correlation — the index is being used as
a quantile estimate, so a nonlinear index-to-quantile map is a biased one.

TIES ARE SPREAD, NOT SNAPPED
----------------------------
Prices are tick-quantized, so the target has atoms at many values; zero is
merely the biggest. An atom occupies a quantile INTERVAL ``[F(y-), F(y)]``.
When that interval straddles a bin edge the rows in it are exchangeable — no
model can tell them apart — so the target mass is split across the bins the
interval covers, in proportion to the OVERLAP. That is exactly the expected
label under random tie-breaking, and it is what keeps every bin at 1/k mass.

  * atom inside one bin              -> one-hot, as before
  * atom covering 30% of bin 9 and
    70% of bin 10                    -> 0.3 / 0.7
  * atom covering four bins whole    -> 0.25 each

``apply`` therefore returns an ``(N, k)`` distribution rather than integer
labels. Cross-entropy takes it directly; the expected-bin losses take the
mass-weighted mean index ``sum_c c * t_c``, which for an atom is the centre of
its quantile interval.

Probe eval does not discretize at all: it fits one ridge per column onto the
z-score and scores Spearman rank IC.
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Fitted discretizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Discretizer:
    """Fitted bin assignment for a single target column.

    Construct via :func:`fit`; never instantiate directly.

    Attributes:
        k: Number of bins.
        edges: The ``k-1`` interior quantile edges. May contain duplicates
            when an atom is wider than a bin — every value that lands on an
            edge is routed through ``atom_*`` instead, so the duplicates are
            never used to assign a bin.
        atom_vals: Sorted values whose quantile interval STRADDLES at least
            one edge, i.e. the ties that have to be spread. Usually a handful;
            often just ``0.0``.
        atom_lo / atom_hi: ``F(v-)`` and ``F(v)`` for each entry of
            ``atom_vals`` — the quantile interval that value occupies.
    """

    k: int
    edges: np.ndarray
    atom_vals: np.ndarray
    atom_lo: np.ndarray
    atom_hi: np.ndarray

    def apply(self, y: np.ndarray) -> np.ndarray:
        """Target distribution ``(len(y), k)``, rows summing to 1.

        One-hot for ordinary values; overlap-split for the stored atoms.
        NaN rows come back as all-zero, which the caller is expected to mask.
        """
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        out = np.zeros((len(y), self.k), dtype=np.float32)
        finite = np.isfinite(y)
        if not finite.any():
            return out

        # Default: one-hot at the bin the value falls in. `side="right"` puts a
        # value exactly on an edge into the upper bin; for the values where
        # that choice would matter (atoms) the atom branch below overwrites it.
        idx = np.searchsorted(self.edges, y, side="right")
        np.clip(idx, 0, self.k - 1, out=idx)
        rows = np.nonzero(finite)[0]
        out[rows, idx[rows]] = 1.0

        if len(self.atom_vals) == 0:
            return out

        # Atoms: split by overlap of [F(v-), F(v)] with each bin's [c/k,(c+1)/k].
        pos = np.searchsorted(self.atom_vals, y)
        np.clip(pos, 0, len(self.atom_vals) - 1, out=pos)
        is_atom = finite & (self.atom_vals[pos] == y)
        if not is_atom.any():
            return out

        a = self.atom_lo[pos[is_atom]][:, None]
        b = self.atom_hi[pos[is_atom]][:, None]
        cuts = np.arange(self.k + 1, dtype=np.float64) / self.k
        ov = np.clip(np.minimum(b, cuts[None, 1:]) - np.maximum(a, cuts[None, :-1]),
                     0.0, None)
        tot = ov.sum(axis=1, keepdims=True)
        # A zero-width interval cannot happen for a stored atom (it straddles an
        # edge by construction), but guard rather than divide by zero.
        safe = tot[:, 0] > 0
        ov[safe] /= tot[safe]
        out[np.nonzero(is_atom)[0][safe]] = ov[safe].astype(np.float32)
        return out

    def expected_index(self, y: np.ndarray) -> np.ndarray:
        """``sum_c c * t_c`` — the scalar bin target for the expected-bin losses.

        For an ordinary value this is its integer bin; for an atom it is the
        centre of the atom's quantile interval expressed in bin units, which is
        the best single number available for a set of tied rows.
        """
        t = self.apply(y)
        return t @ np.arange(self.k, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit(y: np.ndarray, k: int) -> Discretizer | None:
    """Fit a :class:`Discretizer` from a 1-D array of training values.

    NaN entries are dropped. Returns ``None`` when the sample is too small to
    define ``k`` bins (fewer than ``k`` finite values) or is a single constant
    value. Probe-eval treats ``None`` as "skip this column"; supervised
    calibration treats it as a hard error.
    """
    if k < 2:
        raise ValueError(f"n_bins must be at least 2, got {k}")

    v = np.asarray(y, dtype=np.float64).reshape(-1)
    v = np.sort(v[np.isfinite(v)])
    n = len(v)
    if n < k or v[0] == v[-1]:
        return None

    # Empirical CDF limits per distinct value.
    vals, counts = np.unique(v, return_counts=True)
    hi = np.cumsum(counts) / n
    lo = hi - counts / n

    # Interior edges at the k-1 equal-count quantiles.
    q = np.arange(1, k, dtype=np.float64) / k
    edges = vals[np.searchsorted(hi, q, side="left")]

    # A value must be spread iff its quantile interval strictly contains an
    # edge quantile — that is precisely "this tie straddles a bin boundary".
    at = np.searchsorted(hi, q, side="left")
    straddles = (lo[at] < q) & (q < hi[at])
    keep = np.unique(at[straddles])
    return Discretizer(
        k=k,
        edges=edges,
        atom_vals=vals[keep],
        atom_lo=lo[keep],
        atom_hi=hi[keep],
    )
