"""Utility functions for running and tuning EF21-family experiments."""

import argparse
import gc
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

import torch
import wandb
from torch.nn import CrossEntropyLoss

from models import resnet18
from quant import top_k_wrap
from train import tune_step_size
from utils import create_exp, myrepr


def default_gen_config(**overrides) -> Dict:
    """Return the baseline experiment configuration used in the paper."""
    cfg = {
        "model_architecture": resnet18,
        "model_architecture_str": "resnet18",
        "dataset": "cifar10",
        "epochs": 90,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def tuned_presets() -> Dict[str, Dict]:
    """Hyper-parameters tuned for the released runs (Figure 1–4)."""
    return deepcopy(
        {
            "EF21": {"lrs": [1.0], "etas": [None], "p_exps": [None], "q_exps": [None], "cuda_device": "cuda:1"},
            "ECONTROL": {"lrs": [1.0], "etas": [0.1], "p_exps": [None], "q_exps": [None], "cuda_device": "cuda:1"},
            "EF21_SGDM": {"lrs": [0.1], "etas": [0.1], "p_exps": [None], "q_exps": [None], "cuda_device": "cuda:1"},
            "EF21_SGDM_NORM": {"lrs": [0.1], "etas": [1.0], "p_exps": [None], "q_exps": [0.5], "cuda_device": "cuda:1"},
            "EF21_HM_NORM": {"lrs": [0.1], "etas": [1.0], "p_exps": [None], "q_exps": [0.67], "cuda_device": "cuda:1"},
            "EF21_RHM_NORM": {"lrs": [0.1], "etas": [1.0], "p_exps": [None], "q_exps": [0.67], "cuda_device": "cuda:1"},
            "EF21_MVR_NORM": {"lrs": [0.1], "etas": [1.0], "p_exps": [None], "q_exps": [0.67], "cuda_device": "cuda:1"},
            "EF21_IGT_NORM": {"lrs": [0.1], "etas": [1.0], "p_exps": [None], "q_exps": [0.57], "cuda_device": "cuda:1"},
        }
    )


def _prepare_loop_lists(values: Optional[List]) -> List:
    return values if values is not None else [None]


def run_methods(
    methods: Iterable[str],
    *,
    gen_cfg: Optional[Dict] = None,
    presets: Optional[Dict[str, Dict]] = None,
    project_name: str = "EF21_SOM",
    topk_ratio: float = 0.1,
    n_workers: int = 10,
    batch_size: int = 64,
    seed: int = 42,
) -> None:
    """
    Execute tune_step_size for the requested EF methods.

    Parameters mirror the blocks we used in the tmux automation:
      - methods: iterable of keys present in ``presets``.
      - gen_cfg: overrides for model/dataset/epochs/seed (default ResNet18 on CIFAR10 for 90 epochs).
      - presets: dict mapping method -> {lrs, etas, p_exps, q_exps, cuda_device}.
      - project_name: wandb project slug.
      - topk_ratio: k/d used by the Top-K compressor.
      - n_workers, batch_size, seed: distributed training settings.
    """

    if not methods:
        raise ValueError("Please specify at least one EF method to run.")

    base_cfg = default_gen_config(seed=seed)
    if gen_cfg:
        base_cfg.update(gen_cfg)

    presets = presets or tuned_presets()
    missing = [m for m in methods if m not in presets]
    if missing:
        raise KeyError(f"Methods not found in presets: {missing}")

    model_architecture = base_cfg["model_architecture"]
    model_name = base_cfg["model_architecture_str"]
    dataset = base_cfg["dataset"]
    epochs = base_cfg["epochs"]
    criterion = CrossEntropyLoss()

    for method in methods:
        cfg = presets[method]
        lrs = _prepare_loop_lists(cfg.get("lrs"))
        etas = _prepare_loop_lists(cfg.get("etas"))
        p_exps = _prepare_loop_lists(cfg.get("p_exps"))
        q_exps = _prepare_loop_lists(cfg.get("q_exps"))
        cuda_device = cfg.get("cuda_device", "cuda:0")

        print(f"Running {method} on {cuda_device} (lrs={lrs}, etas={etas})")

        for lr in lrs:
            for eta in etas:
                for p_exp in p_exps:
                    lr_schedule = "poly" if p_exp is not None else None
                    for q_exp in q_exps:
                        eta_schedule = "poly" if q_exp is not None else None

                        exp_name = (
                            f"{model_architecture.__name__}_{dataset}_{method}"
                            f"_topk-{myrepr(topk_ratio)}"
                            f"_lr-{myrepr(lr)}"
                            f"_eta-{myrepr(eta)}"
                            f"_p-{myrepr(p_exp)}"
                            f"_q-{myrepr(q_exp)}"
                        )

                        exp = create_exp(
                            name=exp_name,
                            dataset=dataset,
                            net=model_architecture,
                            n_workers=n_workers,
                            epochs=epochs,
                            seed=base_cfg.get("seed", seed),
                            batch_size=batch_size,
                            lrs=[lr],
                            etas=[eta],
                            lr_schedule=lr_schedule,
                            eta_schedule=eta_schedule,
                            compression={"wrapper": False, "compression": top_k_wrap(h=topk_ratio)},
                            error_feedback=method,
                            criterion=criterion,
                            cuda_device=cuda_device,
                            master_compression=None,
                            momentum=0,
                            weight_decay=0,
                            p=p_exp,
                            q=q_exp,
                        )

                        wandb.init(
                            project=project_name,
                            name=f"{exp_name}_{cuda_device}",
                            config={**exp, "lr": lr, "eta": eta, "cuda_device": cuda_device},
                            tags=[exp["error_feedback"], model_name, dataset],
                        )

                        tune_step_size(exp, suffix=exp_name, schedule=lr_schedule, device=cuda_device)
                        wandb.finish()
                        torch.cuda.empty_cache()
                        gc.collect()


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EF21-family experiments.")
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        help="Method to run (repeat flag to queue multiple). Examples: EF21, EF21_RHM_NORM.",
    )
    parser.add_argument("--project", default="EF21_SOM", help="wandb project name.")
    parser.add_argument("--topk", type=float, default=0.1, help="Top-K ratio (K/d).")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers.")
    parser.add_argument("--batch-size", type=int, default=64, help="Worker batch size.")
    parser.add_argument("--epochs", type=int, default=90, help="Training epochs.")
    parser.add_argument("--dataset", default="cifar10", help="Dataset identifier (string label).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print available tuned presets and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    presets = tuned_presets()

    if args.list_methods:
        print("Available methods:", ", ".join(sorted(presets.keys())))
        return

    if not args.methods:
        raise SystemExit("Please supply at least one --method (or use --list-methods).")

    gen_cfg = default_gen_config(dataset=args.dataset, epochs=args.epochs, seed=args.seed)
    run_methods(
        args.methods,
        gen_cfg=gen_cfg,
        presets=presets,
        project_name=args.project,
        topk_ratio=args.topk,
        n_workers=args.workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
