"""The defaults ARE the recipe the paper reports.
import pytest

Every knob below was chosen on a holdout panel and then passed by hand in a
sweep script, which meant a bare ``mode=supervised`` run reproduced none of it:
it trained "mse" -- the collapse control, minimized by a constant on a
standardized target -- at the cross-entropy learning rate, with no information
token, against the mid-to-mid return retired on 2026-08-22. These pin the
defaults so that drifts back are a failing test rather than a silent one.

They are deliberately literal. A test that recomputed the values from the same
source would pass no matter what that source said.
"""

from __future__ import annotations

import dataclasses
import re
import tempfile

import pytest
from pathlib import Path

from market_jepa.modeling.backbones.transformer import (
    TransformerConfig as RuntimeTransformerConfig,
)
from market_jepa.schemas import (
    DatasetConfig, LeJEPAModeConfig, SupervisedModeConfig,
    TransformerBackboneConfig, TransformerInnerConfig,
)


def test_the_two_transformer_configs_cannot_drift_apart():
    """TransformerInnerConfig is what HYDRA builds; TransformerConfig is what a
    directly-constructed backbone gets (tests, eval loaders, from_pretrained).

    They are two hand-maintained copies of one parameter set, so a knob added
    or moved in one and not the other produces a run whose config says one
    thing and whose weights say another -- the failure mode 396939b and 9d8bbf5
    are both named after. Compare every shared field.
    """
    schema = {f.name: getattr(TransformerInnerConfig(), f.name)
              for f in dataclasses.fields(TransformerInnerConfig)
              if f.name != "_target_"}
    runtime = {f.name: getattr(RuntimeTransformerConfig(), f.name)
               for f in dataclasses.fields(RuntimeTransformerConfig)}
    shared = sorted(set(schema) & set(runtime))
    assert shared, "the two configs share no fields; one of them moved"
    # THREE KNOBS ARE DELIBERATELY None IN THE SCHEMA. pool, pos_embed and
    # cls_pos differ BY MODE -- LeJEPA trains mean_sin, the supervised head
    # last_rope -- so hydra spells "not set" as None and each mode states the
    # value on its OWN backbone field, with an explicit pin still winning.
    # The runtime dataclass keeps a real default for anything constructed
    # directly (tests, eval loaders, from_pretrained), and treats None as
    # "use mine". They are allowed to differ ONLY in that direction.
    mode_resolved = {"pos_embed", "cls_pos"}
    for k in mode_resolved:
        assert schema[k] is None, f"{k} must be None in the schema to be mode-resolvable"
        assert runtime[k] is not None, f"{k} needs a real runtime default"
    mismatched = {k: (schema[k], runtime[k])
                  for k in shared if k not in mode_resolved
                  and schema[k] != runtime[k]}
    assert not mismatched, f"schema vs runtime default disagree: {mismatched}"
    # And nothing may exist on only one side without a deliberate decision.
    assert not set(runtime) - set(schema), (
        f"runtime-only knobs are unreachable from hydra: "
        f"{sorted(set(runtime) - set(schema))}")


def test_the_information_token_is_on_and_the_state_token_is_not():
    d = DatasetConfig()
    assert d.info_norm_stats is True
    assert d.info_window is True
    # The state token was never varied on its own -- it moved with the norm
    # stats in all 295 runs that had either -- so its effect was never
    # attributed. Off is the measured configuration.
    assert TransformerBackboneConfig().state_token is False


def test_the_supervised_arm_defaults_to_its_reported_recipe():
    m = SupervisedModeConfig()
    assert m.loss_fn == "pairwise", "mse is the collapse control, not the recipe"
    o = m.training_overrides
    # The batch is CELLS per step (16 cells x 16 stocks = 256 views). The old
    # (1e-5, 256, 100) was measured on a recipe whose backbone never left its
    # init -- 1,800 steps, ~74 pairs a step -- and its LR grid rewarded the
    # LR that perturbed the random features least. See the 2026-09-11 audit.
    # 256 cells a step over 16-cell micro-batches, 12 passes over a six-month
    # span at 2e-4 (2026-09-13, the passes x LR grid on the day-major store):
    # return +0.013 +- 0.004 over the random-init floor, vol +0.024 +- 0.004
    # at 6/6 months. Twelve passes is where return stops gaining -- 20 adds
    # +0.001 +- 0.002 for double the drift -- and 2e-4 beats 1e-4 in every
    # row while 5e-4 leaves the probe AT the floor. Passes and batch inside
    # one month did nothing; see SupervisedModeConfig for the full grid.
    assert (o.blr, o.per_device_train_batch_size, o.effective_batch_size, o.num_epochs) == (2e-4, 16, 256, 12)
    d = m.dataset_overrides
    assert (d.name, d.n_stocks) == ("cross_stock", 16), "the specialist trains on cells"
    b = m.backbone
    # last_sin since 2026-09-11: the pool stays, the position encoding is now
    # uniform with every other mode so the two arms share one floor. RoPE's
    # margin was measured on the flat surrogate that predates 00fbb35.
    assert (b.pool, b.config.pos_embed) == ("last", "sinusoidal"), \
        "last + sinusoidal is the arm"


def test_training_and_scoring_both_default_to_the_uniform_target():
    d = DatasetConfig()
    assert d.xs_target == "uniform"
    assert d.xs_eval_target == "uniform"

    import inspect

    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    default = inspect.signature(StreamingMarketDataset).parameters["xs_target"].default
    assert default == "uniform"


def test_every_mode_owns_its_backbone_and_states_its_readout():
    """One place per mode, and the two arms split the way the grid split them.

    The 2026-09 grid put SSL on mean_sin and the supervised head on last_rope;
    the head moved to last + sinusoidal on 2026-09-11 so that one position
    encoding spans every method and the arms share a random-init floor.
    so there is no ONE readout and the shared schema must not pick one. Since
    2026-09-10 every mode OWNS a ``backbone`` field and states the pair there;
    the parallel ``ModeBackboneOverrides`` mechanism is gone, so a knob cannot
    be described in two places that disagree.

    A mode added without a readout is the failure this guards: it would
    inherit the cls/learned runtime fallback silently, which is exactly what
    DINO and BYOL did until this was made universal.
    """
    import market_jepa.schemas as S

    assert TransformerBackboneConfig().pool is None, "no global readout default"
    assert not hasattr(S.LeJEPAModeConfig(), "backbone_overrides"), \
        "the overrides mechanism was removed; state it on mode.backbone"

    ssl = ["LeJEPA", "IJEPA", "MAE", "CPC", "DINO", "BYOL",
           "TS2Vec", "CoST", "TFC", "TimeMAE"]
    for name in ssl:
        m = getattr(S, f"{name}ModeConfig")()
        assert (m.backbone.pool, m.backbone.config.pos_embed) == \
            ("mean", "sinusoidal"), f"{name} is not mean_sin"
    for name in ("Supervised", "MultiTaskSupervised"):
        m = getattr(S, f"{name}ModeConfig")()
        assert (m.backbone.pool, m.backbone.config.pos_embed) == \
            ("last", "sinusoidal"), f"{name} is not last + sinusoidal"


def test_no_script_still_falls_back_to_the_retired_anchor_tables():
    """XS_STATS_DIR lives in shell, not in schemas.py, and its fallback was
    ``xs_anchor_stats`` -- the mid-to-mid return retired on 2026-08-22, which
    a 4-number statistic reaches +0.0279 on. A run that forgot to export the
    variable trained against it and looked fine doing so.
    """
    repo = Path(__file__).resolve().parents[1]
    stale = sorted(
        str(p.relative_to(repo))
        for p in (repo / "scripts").rglob("*.sh")
        if "XS_STATS_DIR:-xs_anchor_stats}" in p.read_text()
    )
    assert not stale, f"still defaulting to the retired anchor tables: {stale}"


def test_the_deprecated_info_token_spellings_are_gone_from_the_schema():
    """``norm_stats_channels``/``time_info`` were None-defaulted aliases kept
    alive only so jobs queued before the 2026-08-25 rename would not die on a
    hydra struct-mode error. Both cluster queues were empty on 2026-09-06 and
    they were deleted.

    A live config must now use the current spelling. This is the deletion half
    of the contract; the reading half is
    ``test_the_old_spellings_are_still_read_off_old_checkpoints``.
    """
    fields = {f.name for f in dataclasses.fields(DatasetConfig)}
    assert "norm_stats_channels" not in fields and "time_info" not in fields, (
        "the deprecated aliases are back; a live config must use "
        "info_norm_stats / info_window")
    assert {"info_norm_stats", "info_window"} <= fields


def test_the_old_spellings_are_still_read_off_old_checkpoints():
    """Deleting the CONFIG aliases must not stop us READING them.

    128 supervised checkpoints record the old names in their train_meta, and a
    checkpoint is a historical record: the scorer has to keep understanding
    what it says or those models silently rebuild without their info token.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
    from xs_ic_eval import panel_kwargs_for

    old = panel_kwargs_for({"dataset": {"norm_stats_channels": True,
                                        "time_info": True}})
    assert old["info_norm_stats"] is True and old["info_window"] is True


def test_hydra_composes_no_lambda_at_all():
    """The two lambda defaults disagreed for months -- 0.01 in the schema
    against 0.02 in LeJEPA.__init__ -- so a run built through hydra and a model
    built in a notebook trained two different objectives while both reported
    "the default". Only the schema value was ever composed, which is why nobody
    noticed.

    They cannot drift now because the schema carries no value to drift FROM:
    lambda is per-pairing, so hydra composes None and pretrain.py resolves it
    against LAMBDA_BY_PAIRING or raises. LeJEPA.__init__'s default survives for
    directly-constructed models (tests, notebooks) and is reachable no other
    way.
    """
    from market_jepa.schemas import LeJEPAModeConfig

    assert LeJEPAModeConfig().lamb is None, (
        "lambda's optimum is a property of the pairing; a numeric default here "
        "is a value that is right for no arm in LAMBDA_BY_PAIRING")


def test_every_swept_lambda_is_reachable_from_the_pairing():
    """The table is the whole output of a 117-run sweep, and it is now DATA
    rather than a comment -- pretrain.py resolves an unpinned run through it.

    Guard both halves: the values, and the config shape each one is keyed by.
    ``cross_stock`` splits on two knobs the run name does not carry, so k2 and
    k2ind are distinguishable only here.
    """
    from market_jepa.schemas import (
        LAMBDA_BY_PAIRING, STANDARD_INDUSTRY_TABLE, lejepa_pairing_key)

    assert LAMBDA_BY_PAIRING == {
        "k2": 0.2, "k2ind": 0.1, "random_resized_crop": 0.05,
        "time_warp": 0.001, "gaussian_noise": 0.3,
    }
    cs = {"name": "cross_stock", "n_stocks": 2}
    assert lejepa_pairing_key(
        [{**cs, "industry_table": STANDARD_INDUSTRY_TABLE}]) == "k2ind"
    assert lejepa_pairing_key([{**cs, "industry_table": None}]) == "k2"
    for name in ("random_resized_crop", "time_warp", "gaussian_noise"):
        assert lejepa_pairing_key([{"name": name}]) == name


def test_an_untuned_pairing_resolves_to_nothing_rather_than_to_a_guess():
    """K != 2 and the same-stock augs outside the swept three have no tuned
    lambda, so there is no value to fall back to and pretrain.py must raise.

    Returning some other arm's optimum would be the 0.01 failure again with a
    different number: a value nothing was measured at, applied silently.
    """
    from market_jepa.schemas import LAMBDA_BY_PAIRING, lejepa_pairing_key

    for augs in ([{"name": "cross_stock", "n_stocks": 4}],
                 [{"name": "volume_noise"}],
                 [{"name": "price_jitter"}],
                 [{"name": "channel_drop"}],
                 []):
        key = lejepa_pairing_key(augs)
        assert key is None or key not in LAMBDA_BY_PAIRING, (
            f"{augs} resolved to {key!r}, which would hand it an untuned "
            "lambda instead of demanding an explicit pin")


def test_every_lejepa_sweep_pins_lambda():
    """With no numeric default, a sweep that does not pin lambda either resolves
    from the table -- training a DIFFERENT value than it used to -- or dies.

    Nine sweeps inherited 0.01 and were pinned to it on 2026-09-06 so they keep
    training what they trained. A new LeJEPA sweep must make the same choice
    deliberately.
    """
    repo = Path(__file__).resolve().parents[1]
    unpinned = sorted(
        p.name for p in (repo / "scripts" / "sweeps").glob("*.sh")
        if "mode=lejepa" in (txt := p.read_text()) and "mode.lamb=" not in txt
    )
    assert not unpinned, (
        f"LeJEPA sweeps with no mode.lamb pin: {unpinned}. Lambda is "
        "per-pairing (see LAMBDA_BY_PAIRING); pin it explicitly.")


def test_warp_strength_defaults_to_what_every_run_actually_used():
    """0.75, matching the pin in lejepa_samestock_lambda.sh.

    It defaulted to 0.25 until 2026-08-27 while every scored time_warp run
    pinned 0.75, so the default was a value nobody had measured and an unpinned
    run silently differed from the whole result set. This is alignment with
    what runs, NOT a claim that 0.75 is optimal -- strength has never been
    swept under rank IC.
    """
    from market_jepa.schemas import AugmentationConfig
    assert AugmentationConfig().warp_strength == 0.75


def test_no_live_sweep_restates_a_schema_default():
    """Sweep files name the method and the swept axis, and nothing else.

    REPLACES a test that did the opposite. It used to parse
    sweeps/lejepa_samestock_lambda.sh and assert the schema AGREED with what
    the sweep pinned -- enforcing exactly the duplication that turned out to be
    the problem. Pinning a default in a sweep does not prevent drift; it makes
    drift silent, because the sweep goes on training the old recipe after
    schemas.py has moved and nothing errors. See scripts/sweeps/README.md.

    The live example: full_data_supervised.sh carried training.live_eval=false
    for so long that the switch was assumed to be the default. It was not --
    the value had been set on LiveEvalConfig.enabled, a dataclass nothing
    reads, while pretrain.py reads TrainingConfig.live_eval. Stripping the line
    silently turned live eval back ON.

    A knob a sweep genuinely SWEEPS is passed as a shell variable
    (optimizer.blr=${BLR}), not as a literal, so it does not trip this.
    """
    from pathlib import Path
    from market_jepa.schemas import (
        DatasetConfig, SupervisedModeConfig, TrainingConfig)

    sup = SupervisedModeConfig()
    o, d = sup.training_overrides, DatasetConfig()
    redundant = {
        "mode.loss_fn": sup.loss_fn,
        "optimizer.blr": o.blr,
        "training.per_device_train_batch_size": o.per_device_train_batch_size,
        "training.num_epochs": o.num_epochs,
        "training.live_eval": str(TrainingConfig().live_eval).lower(),
        "dataset.xs_target": d.xs_target,
        "dataset.xs_eval_target": d.xs_eval_target,
        "dataset.n_pairs_per_obs": d.n_pairs_per_obs,
        "backbone.pool": sup.backbone.pool,
        "backbone.config.pos_embed": sup.backbone.config.pos_embed,
    }
    sweeps = Path(__file__).resolve().parents[1] / "scripts" / "sweeps"
    offenders = []
    for f in sorted(sweeps.rglob("*.sh")):
        body = "\n".join(ln for ln in f.read_text().splitlines()
                         if not ln.lstrip().startswith("#"))
        for knob, default in redundant.items():
            if default is None:
                continue
            for m in re.finditer(rf"{re.escape(knob)}=(\S+)", body):
                val = m.group(1).rstrip("\\").strip()
                if "$" in val:
                    continue          # a swept axis, not a restated default
                if val.lower() == str(default).lower():
                    offenders.append(f"{f.name}: {knob}={val} IS the default")
    assert not offenders, (
        "sweep files must not restate schema defaults:\n  "
        + "\n  ".join(offenders))


def _resolve(cfg_epochs, cfg_steps, cfg_batch, ov_epochs, ov_batch):
    """The exact precedence pretrain.py implements, in isolation."""
    from market_jepa.schemas import TrainingConfig
    fb = TrainingConfig()
    batch = cfg_batch
    if batch is None:
        batch = ov_batch
    if batch is None:
        batch = fb.fallback_train_batch_size
    epochs, steps = cfg_epochs, cfg_steps
    if epochs is None and steps is None:
        epochs = ov_epochs
    if epochs is None and steps is None:
        epochs = fb.fallback_num_epochs
    return epochs, steps, batch


def test_an_explicit_budget_pin_beats_the_mode_override():
    """The whole point of the 2026-08-27 precedence fix.

    batch_size_stability puts batch on its sweep axis and variance_decomp asks
    for 100 epochs at batch 128; under the old order the supervised override
    silently gave both 200 at 256, so one sweep measured a batch it never
    trained at and the other measured a budget it did not choose.
    """
    # variance_decomp's ask, against the supervised override
    assert _resolve(100, None, 128, 200, 256) == (100, None, 128)
    # batch_size_stability sweeping batch
    for bs in (64, 128, 256, 512):
        assert _resolve(None, None, bs, 200, 256)[2] == bs


def test_a_step_budget_survives_the_mode_override():
    """`training.num_epochs=null` beside max_train_steps is how six lejepa and
    four supervised sweeps ask for a STEP budget.

    Resurrecting num_epochs there tripped the mutual-exclusion check and killed
    the job at startup, which is how this class of bug announces itself.
    """
    epochs, steps, _ = _resolve(None, 10800, None, 200, 256)
    assert epochs is None and steps == 10800, (
        "the override resurrected num_epochs beside max_train_steps")


def test_the_mode_override_still_supplies_the_recipe_when_nothing_is_pinned():
    """It has to remain useful, or the defaults are back to a configuration
    nobody runs -- which is what 4a85582 was fixing in the first place."""
    assert _resolve(None, None, None, 200, 256) == (200, None, 256)


def test_a_mode_with_no_override_trains_the_campaign_budget():
    """Every method gets the SAME data budget: the six months before the eval
    month (DatasetConfig.train_span_months) at twelve passes, settled
    2026-09-13. A method that trained a different number of passes would make
    an IC difference unreadable -- representation or budget, no way to tell --
    so the fallback moved from the old single-month 100 to 12 with it. Batch
    is unchanged: it is a method's own choice, not part of the budget."""
    from market_jepa.schemas import DatasetConfig, TrainingConfig
    fb = TrainingConfig()
    assert _resolve(None, None, None, None, None) == (
        fb.fallback_num_epochs, None, fb.fallback_train_batch_size)
    assert (fb.fallback_num_epochs, fb.fallback_train_batch_size) == (12, 128)
    assert DatasetConfig().train_span_months == 6


def test_unset_means_none_so_a_pin_is_distinguishable():
    """If these defaulted to 100/128 again, an explicit 128 would be
    indistinguishable from silence and the override would have to win."""
    from market_jepa.schemas import TrainingConfig
    t = TrainingConfig()
    assert t.num_epochs is None and t.per_device_train_batch_size is None


def test_run_recipe_filters_read_the_config_not_the_run_name():
    """A pre-``eb22cac`` ``k2_lamb0.005_bs256_s42`` survives under the same glob
    and the same run name as a current one while carrying a superseded recipe.
    Because analysis cells are keyed by month it simply BECAME 2009-06's
    lambda=0.005 number, moving that cell from +0.0000 to -0.0019 and flipping
    a significance call on spread.

    The keys are written only when they differ from what a reader assumes, so
    absence is a value: no ``xs_target`` means zscore, no ``xs_anchor_stats``
    means the pre-stamp table set.
    """
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
    from style import iter_ic_runs

    root = Path(tempfile.mkdtemp())
    for name, meta in (
        ("stale", {"run_name": "k2_lamb0.005_bs256_s42"}),           # zscore
        ("ranked", {"run_name": "k2_lamb0.005_bs256_s42",
                    "xs_target": "rank"}),
        ("fwdvwap", {"run_name": "k2_lamb0.005_bs256_s42",
                     "xs_target": "rank",
                     "xs_anchor_stats": "xs_anchor_stats_fwdvwap60"}),
    ):
        d = root / "k2-lambda-abc123-2009-06-01-2009-06-30" / name
        d.mkdir(parents=True)
        (d / "train_meta.json").write_text(json.dumps(meta))
        (d / "xs_ic.json").write_text("{}")

    glob = "k2-lambda-*"
    assert len(iter_ic_runs(glob, ckpt_root=root)) == 3, "unfiltered sees all"

    # The name pins nothing; the config does.
    assert {r.run_id for r in iter_ic_runs(
        glob, ckpt_root=root, xs_target="rank")} == {"ranked", "fwdvwap"}
    assert {r.run_id for r in iter_ic_runs(
        glob, ckpt_root=root, xs_target="zscore")} == {"stale"}
    assert {r.run_id for r in iter_ic_runs(
        glob, ckpt_root=root, xs_target="rank",
        xs_stats="xs_anchor_stats_fwdvwap60")} == {"fwdvwap"}

    # And the month filter, for sweeps that share run names across panels.
    assert iter_ic_runs(glob, ckpt_root=root, train_months={"2009-06"})
    assert not iter_ic_runs(glob, ckpt_root=root, train_months={"2011-04"})


def test_lambda_resolves_from_the_pairing_or_refuses_to_guess():
    """The three branches of the real resolver, not a re-implementation of it.

    An explicit pin always wins, an unpinned run gets its own pairing's swept
    optimum, and an unmapped pairing raises rather than borrowing another arm's
    number -- which would be the 0.01 failure again with a different constant.
    """
    from omegaconf import OmegaConf
    from market_jepa.schemas import LeJEPAModeConfig, STANDARD_INDUSTRY_TABLE
    from market_jepa.training.pretrain import resolve_lejepa_lambda

    def cfg_with(lamb):
        m = OmegaConf.structured(LeJEPAModeConfig())
        m.lamb = lamb
        return OmegaConf.create({"mode": m})

    k2ind = [{"name": "cross_stock", "n_stocks": 2,
              "industry_table": STANDARD_INDUSTRY_TABLE}]

    # An explicit pin wins even where the table has an entry.
    cfg = cfg_with(0.01)
    assert resolve_lejepa_lambda(cfg, k2ind) == 0.01
    assert cfg.mode.lamb == 0.01, "the resolved value must be written back"

    # Unpinned resolves to THIS pairing's optimum, not to some global default.
    cfg = cfg_with(None)
    assert resolve_lejepa_lambda(cfg, k2ind) == 0.1
    assert cfg.mode.lamb == 0.1
    assert resolve_lejepa_lambda(
        cfg_with(None), [{"name": "time_warp"}]) == 0.001

    # An untuned pairing has no value to fall back to.
    for augs in ([{"name": "cross_stock", "n_stocks": 4}],
                 [{"name": "volume_noise"}]):
        with pytest.raises(ValueError, match="property of the PAIRING"):
            resolve_lejepa_lambda(cfg_with(None), augs)


def test_latent_evals_read_the_mean_not_the_training_readout():
    """The latent suite must not inherit the supervised arm's `last` readout.

    Supervised trains pool="last", so reading its latent there would make
    every structure number a statement about ONE patch of the day -- and would
    confound each supervised-vs-SSL comparison, since LeJEPA is mean-pooled by
    construction. Pooling is applied after the last block, so overriding it
    changes the readout and no weight.
    """
    import inspect
    import sys

    from market_jepa.eval.checkpoints import load_backbone

    assert "pool" in inspect.signature(load_backbone).parameters, (
        "load_backbone lost its pool override; the latent evals depend on it")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "plots" / "latent_eval" / "fixed_panel"))
    import panel_lib

    assert panel_lib.LATENT_POOL == "mean"

    # load_encoder IS A LOADER TOO, and scanning only _load_backbone missed it
    # for six days. Every key in MODEL_ORDER is manifest-resolved and a
    # manifest row goes through load_encoder, so that path -- not the glob one
    # this test was written for -- is how the entire reported roster is built.
    from market_jepa.eval.checkpoints import load_encoder

    assert "pool" in inspect.signature(load_encoder).parameters, (
        "load_encoder lost its pool override; the manifest-resolved roster "
        "(i.e. every reported encoder) is read through it")

    # Every loader in the latent suite must pass it. A call site that forgets
    # silently falls back to the checkpoint's own readout, which is exactly
    # the bug this guards.
    root = Path(__file__).resolve().parents[1] / "plots" / "latent_eval"
    offenders = []
    for path in root.rglob("*.py"):
        src = path.read_text()
        for opener in ("_load_backbone(", "load_encoder("):
            for chunk in src.split(opener)[1:]:
                call = chunk[:chunk.index(")")] if ")" in chunk else chunk
                if "pool=" not in call:
                    offenders.append(f"{path.name}:{opener.rstrip('(')}")
    assert not offenders, (
        f"latent-eval loaders that do not pin the readout: {sorted(set(offenders))}")


def test_the_random_init_floor_is_read_like_the_model_it_floors():
    """The delta-IC subtrahend must use the readout its numerator uses.

    PREDICTION IS THE LAST TOKEN FOR BOTH ARMS. LeJEPA computes its invariance
    loss on the mean, but its reported prediction comes from a last-token
    probe -- worth +0.055 / +0.083 / +0.112 over the mean on holdout-2 -- and
    the supervised arm trains "last" outright. So the floor is last for both,
    and the two share one architecture signature and one floor computation.

    The latent suite is the deliberate exception: it reads every encoder at the
    mean, so its floor takes the override rather than the config.
    """
    import sys

    import torch

    from market_jepa.eval.checkpoints import (
        architecture_signature, build_untrained_encoder)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "scripts" / "eval"))
    from xs_ic_eval import PREDICT_POOL, _predict_readout

    assert PREDICT_POOL == "last"

    def cfg_for(pool, target):
        return {"n_features": 9, "mode": {"_target_": target}, "backbone": {
            "_target_": "market_jepa.modeling.backbones.transformer."
                        "TransformerBackbone",
            "d_embedding": 64, "pool": pool,
            "config": {"hidden_size": 64, "num_hidden_layers": 2,
                       "num_attention_heads": 4, "intermediate_size": 128,
                       "pos_embed": "sinusoidal"}}}

    lejepa = _predict_readout(
        cfg_for("mean", "market_jepa.modeling.modes.lejepa.LeJEPA"))
    supervised = _predict_readout(
        cfg_for("last", "market_jepa.modeling.modes.supervised.SupervisedModel"))

    assert lejepa["backbone"]["pool"] == "last", "a mean-trained arm must still be SCORED at last"
    assert supervised["backbone"]["pool"] == "last"

    dev = torch.device("cpu")
    assert build_untrained_encoder(lejepa, dev).backbone.pool == "last"
    assert build_untrained_encoder(supervised, dev).backbone.pool == "last"

    # Same readout and same shape -> one floor serves both arms.
    assert architecture_signature(lejepa) == architecture_signature(supervised)

    # But the signature still SEPARATES readouts, so a differently-read model
    # cannot silently inherit this floor.
    other = cfg_for("mean", "market_jepa.modeling.modes.lejepa.LeJEPA")
    assert architecture_signature(other) != architecture_signature(lejepa)

    # The latent suite forces the mean whatever the model trained with.
    assert build_untrained_encoder(
        supervised, dev, pool="mean").backbone.pool == "mean"
