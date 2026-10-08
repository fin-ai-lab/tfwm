"""ONE place that answers "which backbone block does this run use?".

THE RULE, AND IT IS ONE LINE: a mode that declares its own ``backbone``
owns it, and the top-level ``cfg.backbone`` is ignored entirely. Twelve of
the fourteen modes declare one (SSL at ``mean`` pooling with sinusoidal
positions, the supervised family at ``last`` under RoPE); only
FinanceBaseline and PretrainedTSFM do not, and neither builds a ViT.

WHY THIS MODULE EXISTS. That one-line rule was re-implemented at ten call
sites -- the trainer, eight modes, and the eval loader -- and two of them got
it wrong in ways that cost real runs:

  * the eval loader tested ``"IJEPA" in mode_target``, naming ONE of the modes
    that own a backbone. For every other owning mode it read the top-level
    block, which is not empty -- it carries the composed default, ``pool="cls"``
    with learned positions. For the supervised family that built a cls-pooled
    2049-position model against a last/rope 2048-position checkpoint and every
    post-train IC eval died in ``load_state_dict``. For SSL it was SILENT: the
    checkpoint merely got signed into the wrong architecture class, and so was
    scored against the wrong random-init floor.

  * ``architecture_signature`` shared that bug, which is what made the floors
    wrong rather than merely missing.

A rule expressed in ten places is a rule that will be wrong in one of them.
This is the tenth place, and the other nine defer to it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Modes that legitimately declare no backbone. Anything NOT here is expected to
# own one, and tests/test_backbone_ownership.py fails if a new mode quietly
# joins this list by omission.
MODES_WITHOUT_A_BACKBONE = frozenset({
    "FinanceBaselineModeConfig",   # classical factors; no encoder at all
    "PretrainedTSFMModeConfig",    # loads a foreign pretrained model wholesale
})


def _is_block(value) -> bool:
    """Is this a mapping-like config block rather than a scalar or sequence?"""
    if value is None or isinstance(value, (int, float, str, bytes, bool)):
        return False
    if isinstance(value, (list, tuple, set)):
        return False
    return isinstance(value, dict) or hasattr(value, "keys") or hasattr(value, "__dict__")


def _has(container, key: str) -> bool:
    """Membership that works for dict, DictConfig and dataclass alike."""
    if container is None:
        return False
    if isinstance(container, dict):
        return container.get(key) is not None
    try:
        return key in container
    except TypeError:
        return getattr(container, key, None) is not None


def _get(container, key: str, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    try:
        return container[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(container, key, default)


def backbone_block(cfg):
    """The backbone config this run ACTUALLY trains.

    Accepts a plain dict (a checkpoint's ``train_meta.json``), a DictConfig
    (the live Hydra config) or anything attribute-addressable, so the trainer
    and the scorer can ask the same question of different objects and get the
    same answer.

    Returns ``{}`` rather than raising when nothing is resolvable: callers are
    scoring loops that must report a bad checkpoint, not crash on one.
    """
    mode = _get(cfg, "mode") or {}
    # TRUTHINESS, not membership: a mode carrying `backbone: {}` has declared
    # nothing and must fall through to the top level, exactly as before.
    owned = _get(mode, "backbone")
    if owned:
        return owned
    # Some runs logged cfg.backbone as a bare PARAMETER COUNT rather than a
    # block, which is why this is a type test and not just a lookup.
    top = _get(cfg, "backbone")
    # ALLOWLIST, not a denylist of the scalars we happen to have seen. The
    # contract is that this returns a block or {}, and a denylist lets anything
    # unanticipated (a list, say) escape to callers that will index it.
    if _is_block(top):
        return top
    return {}


def owns_backbone(cfg) -> bool:
    """Does the selected mode declare its own backbone?

    An empty block is not a declaration -- such a mode uses the top level, so
    an override there is legitimate and must not be rejected.
    """
    return bool(_get(_get(cfg, "mode") or {}, "backbone"))


def ignored_top_level_overrides() -> list[str]:
    """Hydra overrides the user TYPED that target the ignored top-level block.

    ``backbone=transformer`` is a GROUP selection: every sweep passes it, it is
    conventional, and it is harmless. What this finds is a FIELD override --
    ``backbone.config.pos_embed=sinusoidal``, ``backbone.pool=mean`` -- which
    composes cleanly, logs plausibly, and trains something else entirely when
    the mode owns its backbone. The ``_target_`` is identical either way, so no
    comparison of the two blocks can catch it; only the literal override list
    can.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        task = list(HydraConfig.get().overrides.task)
    except Exception:  # noqa: BLE001 - no Hydra context (tests, notebooks)
        return []
    found = []
    for item in task:
        text = str(item)
        bare = text.lstrip("+~")
        # "backbone.x=y" is a field override; "backbone=x" is a group choice.
        if bare.startswith("backbone."):
            found.append(text)
    return found


def assert_no_ignored_backbone_overrides(cfg) -> None:
    """Refuse to train a backbone the caller did not ask for.

    A field override on the top-level block, for a mode that owns its own, is
    ALWAYS a mistake: the run trains the mode's defaults while the command line
    and the logged config both read as though it did not. It has to be an error
    rather than a warning, because the place this happens is a sweep file whose
    output nobody reads until the figures look wrong.
    """
    if not owns_backbone(cfg):
        return
    ignored = ignored_top_level_overrides()
    if not ignored:
        return
    mode_target = str(_get(_get(cfg, "mode") or {}, "_target_", "") or "")
    mode_name = mode_target.rsplit(".", 1)[-1] or "this mode"
    corrected = ", ".join(f"mode.{o.lstrip('+~')}" for o in ignored)
    raise ValueError(
        f"{mode_name} owns its backbone, so the top-level backbone block is "
        f"ignored -- but the command line overrides it: {' '.join(ignored)}.\n"
        f"That would train {mode_name}'s OWN backbone defaults while the "
        f"config logs your value, which is how a sweep silently measures the "
        f"wrong thing.\n"
        f"Set it on the mode instead: {corrected}"
    )
