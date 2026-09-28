"""Async probe eval worker — fits ridge probes on frozen backbone embeddings.

Spawned as a detached subprocess by pretrain.py. Logs probe metrics directly
to the same W&B run via shared mode.

Each (target, horizon) column gets ONE ridge regression onto the configured
target (empirical-uniform rank by default), scored by pooled Spearman rank IC.
This replaced the k ∈ {2,3,5} logistic-classifier grid the project reported
before the switch to rank IC: three classifier fits per column became one
closed-form solve, which matters because these nodes run 8 CPUs against an H100
and the probe used to be the CPU bottleneck.

The IC here is POOLED — in-training probe samples are independent draws, not
synchronized cross-sections. The reported per-cell IC comes from the separate
synchronized cross-section eval.

Usage (called by dispatch_probe_eval, not directly):
    python -m market_jepa.eval.probe_eval_worker \
        --npz_path /path/to/probe_100_abc123.npz \
        --obs_seen 6400 \
        --wandb_run_id <run_id> \
        --wandb_project market-jepa \
        --wandb_entity boothai \
        --temporal_train_frac 0.5
"""

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from market_jepa.eval.metrics import rank_ic

# Ridge penalty for the probe. Embeddings are standardized first, so this is
# scale-free; a single fixed value keeps the probe deterministic and avoids a
# per-column CV sweep that would put the CPU bottleneck straight back.
RIDGE_ALPHA = 1.0

# Below this many usable rows a column is skipped rather than scored.
MIN_ROWS = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--obs_seen", type=int, required=True)
    parser.add_argument("--wandb_run_id", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--temporal_train_frac", type=float, default=0.5)
    parser.add_argument("--ridge_alpha", type=float, default=RIDGE_ALPHA)
    return parser.parse_args()


def main():
    args = parse_args()

    data = np.load(args.npz_path, allow_pickle=False)
    target_names = data["target_names"].tolist()

    if "X_train" in data:
        # Pre-split mode: separate probe train/val date windows
        X_train = data["X_train"]
        y_train_all = data["y_train"]
        X_val = data["X_val"]
        y_val_all = data["y_val"]
    else:
        # Legacy temporal split mode
        X = data["X"]
        y = data["y"]
        n_samples = len(X)
        if n_samples < 20:
            print(f"Too few samples ({n_samples}), skipping probe eval")
            return
        split_idx = int(n_samples * args.temporal_train_frac)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train_all, y_val_all = y[:split_idx], y[split_idx:]

    if len(X_train) < 10 or len(X_val) < 10:
        print(f"Insufficient train ({len(X_train)}) or val ({len(X_val)}) samples")
        return

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    metrics = {}

    for ti, target_name in enumerate(target_names):
        y_tr = y_train_all[:, ti]
        y_va = y_val_all[:, ti]

        # NaN carries the no-clamp rule (anchors with < h seconds of session
        # left) as well as ordinary missing quotes — drop per column, since
        # which rows are NaN differs by horizon.
        ok_tr = np.isfinite(y_tr)
        ok_va = np.isfinite(y_va)
        if ok_tr.sum() < MIN_ROWS or ok_va.sum() < MIN_ROWS:
            continue

        try:
            model = Ridge(alpha=args.ridge_alpha)
            model.fit(X_train_scaled[ok_tr], y_tr[ok_tr])
            pred = model.predict(X_val_scaled[ok_va])
        except Exception as e:  # noqa: BLE001
            print(f"  Probe failed: {target_name}: {e}")
            continue

        ic = rank_ic(pred, y_va[ok_va])
        if np.isfinite(ic):
            metrics[f"probe/ridge_ic_{target_name}"] = float(ic)

        # R^2 on the configured target, as a scale-aware companion to rank IC:
        # IC can look healthy while the head is badly miscalibrated, and the
        # supervised loss is trained on this scale.
        resid = pred - y_va[ok_va]
        var = float(np.var(y_va[ok_va]))
        if var > 0:
            metrics[f"probe/ridge_r2_{target_name}"] = float(
                1.0 - float(np.mean(resid ** 2)) / var
            )

    if metrics:
        import wandb
        from wandb.sdk.wandb_settings import Settings

        wandb.init(
            settings=Settings(mode="shared"),
            id=args.wandb_run_id,
            resume="must",
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        wandb.log({**metrics, "obs_seen": args.obs_seen})
        wandb.finish()
        print(f"Logged {len(metrics)} probe metrics at obs_seen={args.obs_seen}")
    else:
        print("No probe metrics to log")

    try:
        os.remove(args.npz_path)
        print(f"Cleaned up {args.npz_path}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
