"""Hydra entry point for market-jepa training.

Usage:
    # LeJEPA + transformer on the released Market-1T data. The months the run
    # needs are downloaded to <repo>/market1t/ first (with the target tables).
    uv run train.py

    # Supervised mode on the day store
    uv run train.py mode=supervised dataset.backend=days

    # Stream from the Hub instead of downloading
    uv run train.py machine=hub

`machine=market1t` is the default (set on Config.defaults in market_jepa/schemas.py).
The lab's own machines (bll01, pythia, ...) are selected the same way; their
cluster launchers live in scripts/pythia/ and scripts/generic/.
"""

import hydra
from hydra.core.config_store import ConfigStore
from market_jepa.schemas import (
    Config,
    MACHINE_CONFIGS,
    # Backbone groups
    TransformerBackboneConfig,
    ResNetBackboneConfig,
    EfficientNetBackboneConfig,
    ConvNeXtBackboneConfig,
    InceptionBackboneConfig,
    PatchTSTBackboneConfig,
    # Mode groups
    LeJEPAModeConfig,
    IJEPAModeConfig,
    MAEModeConfig,
    CPCModeConfig,
    DINOModeConfig,
    BYOLModeConfig,
    TS2VecModeConfig,
    CoSTModeConfig,
    TFCModeConfig,
    TimeMAEModeConfig,
    FinanceBaselineModeConfig,
    PretrainedTSFMModeConfig,
    SupervisedModeConfig,
    MultiTaskSupervisedModeConfig,
)

cs = ConfigStore.instance()
# Register Config itself as the root — replaces what conf/config.yaml used to do.
cs.store(name="config", node=Config)

# Backbone group
cs.store(group="backbone", name="transformer", node=TransformerBackboneConfig)
cs.store(group="backbone", name="resnet", node=ResNetBackboneConfig)
cs.store(group="backbone", name="efficientnet", node=EfficientNetBackboneConfig)
cs.store(group="backbone", name="convnext", node=ConvNeXtBackboneConfig)
cs.store(group="backbone", name="inception", node=InceptionBackboneConfig)
cs.store(group="backbone", name="patchtst", node=PatchTSTBackboneConfig)

# Mode group
cs.store(group="mode", name="lejepa", node=LeJEPAModeConfig)
cs.store(group="mode", name="ijepa", node=IJEPAModeConfig)
cs.store(group="mode", name="mae", node=MAEModeConfig)
cs.store(group="mode", name="cpc", node=CPCModeConfig)
cs.store(group="mode", name="dino", node=DINOModeConfig)
cs.store(group="mode", name="byol", node=BYOLModeConfig)
cs.store(group="mode", name="ts2vec", node=TS2VecModeConfig)
cs.store(group="mode", name="cost", node=CoSTModeConfig)
cs.store(group="mode", name="tfc", node=TFCModeConfig)
cs.store(group="mode", name="timemae", node=TimeMAEModeConfig)
cs.store(group="mode", name="finance_baseline", node=FinanceBaselineModeConfig)
cs.store(group="mode", name="pretrained_tsfm", node=PretrainedTSFMModeConfig)
cs.store(group="mode", name="supervised", node=SupervisedModeConfig)
cs.store(group="mode", name="multi_supervised", node=MultiTaskSupervisedModeConfig)

# Machine group — registered from schemas.MACHINE_CONFIGS
for _name, _node in MACHINE_CONFIGS.items():
    cs.store(group="machine", name=_name, node=_node)


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def main(cfg: Config):
    # Optional: skip if W&B run already finished. Crashed/killed/failed runs
    # are NOT skipped — they need to be retried (e.g. after a node-down event).
    if cfg.skip_if_done and cfg.wandb.run_name:
        import wandb

        api = wandb.Api()
        try:
            done = api.runs(
                f"{cfg.wandb.entity}/{cfg.wandb.project}",
                filters={
                    "display_name": cfg.wandb.run_name,
                    "state": "finished",
                },
            )
            if len(done) > 0:
                print(
                    f"Skipping '{cfg.wandb.run_name}' -- already finished on W&B."
                )
                return
        except Exception as exc:
            # NOTHING here may cost a job. This guard is an optimization -- it
            # avoids redoing finished work -- but api.runs() reaches
            # api.wandb.ai, and the old `except ValueError` let a network error
            # straight through: a 33-job wave on 2026-08-29 lost jobs to
            # `requests.exceptions.ConnectTimeout` raised out of this call,
            # each one dying before a single training step.
            #
            # Falling through to training is the right failure mode. The worst
            # case is repeating a run that was already finished; the
            # alternative is losing one that was not.
            print(
                f"skip_if_done: could not reach W&B "
                f"({type(exc).__name__}: {exc}); training anyway."
            )

    from market_jepa.training.pretrain import train

    train(cfg)


if __name__ == "__main__":
    main()
