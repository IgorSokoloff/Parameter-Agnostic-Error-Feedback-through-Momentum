import glob
import os
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Styling shared across plots
ARG = dict(label_fontsize=20, marker_size=15)
METRICS: Dict[str, Tuple[str, str]] = {
    "train_loss": ("Train loss", "Train loss"),
    "test_loss": ("Test loss", "Test loss"),
    "test_acc": ("Test accuracy", "Accuracy"),
}
TIME_BASED = {"time", "gpu_seconds"}
AUTO_COLORS = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
AUTO_MARKERS = ["o", "s", "^", "v", "D", "*", "P", "X", "<", ">", "p", "h", "H", "8"]


def configure_style() -> None:
    """Apply figure-wide style settings."""
    plt.rcParams.update(
        {
            "lines.linewidth": 2,
            "xtick.labelsize": ARG["label_fontsize"],
            "ytick.labelsize": ARG["label_fontsize"],
            "legend.fontsize": ARG["label_fontsize"],
            "axes.titlesize": ARG["label_fontsize"],
            "axes.labelsize": ARG["label_fontsize"],
            "figure.figsize": [10, 8],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "font.family": "serif",
        }
    )


def build_curve_specs(folder: str = "_release_data/") -> List[Dict]:
    """Return the canonical list of experiment pickles to plot."""
    folder = folder if folder.endswith("/") else f"{folder}/"
    return [
        {
            "pattern": folder + "resnet18_cifar10_EF21_SGDM_topk-0,1_lr-0,1_eta-0,1_p-None_q-None.pickle",
            "label": "EF21-SGDM",
            "color": "orange",
            "marker": "v",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_ECONTROL_topk-0,1_lr-1,0_eta-0,1_p-None_q-None.pickle",
            "label": "ECONTROL",
            "color": "yellowgreen",
            "marker": "p",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_topk-0,1_lr-1,0_eta-None_p-None_q-None.pickle",
            "label": "EF21",
            "color": "blue",
            "marker": "o",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_MVR_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,67.pickle",
            "label": r"$\|\text{EF21-MVR}\|$",
            "color": "#8c564b",
            "marker": "<",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_RHM_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,67.pickle",
            "label": r"$\|\text{EF21-RHM}\|$",
            "color": "#e377c2",
            "marker": ">",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_SGDM_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,5.pickle",
            "label": r"$\|\text{EF21-SGDM}\|$",
            "color": "darkgreen",
            "marker": "s",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_IGT_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,57.pickle",
            "label": r"$\|\text{EF21-IGT}\|$",
            "color": "#9467bd",
            "marker": "^",
            "linestyle": "-",
        },
        {
            "pattern": folder + "resnet18_cifar10_EF21_HM_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,67.pickle",
            "label": r"$\|\text{EF21-HM}\|$",
            "color": "red",
            "marker": "*",
            "linestyle": "-",
        },
    ]


def load_many(pattern: str) -> List[Dict]:
    runs = []
    for fname in glob.glob(pattern):
        with open(fname, "rb") as f:
            runs.append(pickle.load(f))
    return runs


def mean_curve(runs: List[Dict], key: str):
    if not runs:
        return None
    arrs = [r.get(key) for r in runs if key in r]
    if not arrs:
        return None
    min_len = min(len(a) for a in arrs)
    stack = [np.asarray(a[:min_len]) for a in arrs]
    return np.mean(stack, axis=0)


def auto_label_from_pattern(pattern: str) -> str:
    base = os.path.basename(pattern).replace(".pickle", "")
    parts = base.split("_")
    matches = [p for p in parts if p.startswith("EF21")]
    return matches[0] if matches else base


def ensure_style(spec: Dict, idx: int) -> Dict:
    spec = deepcopy(spec)
    if "label" not in spec:
        spec["label"] = auto_label_from_pattern(spec["pattern"])
    if "color" not in spec:
        spec["color"] = AUTO_COLORS[idx % len(AUTO_COLORS)]
    if "marker" not in spec:
        spec["marker"] = AUTO_MARKERS[idx % len(AUTO_MARKERS)]
    if "linestyle" not in spec:
        spec["linestyle"] = "-"
    return spec


def _cum_gpu_seconds(arr: List[float]) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    c = np.cumsum(arr)
    return np.concatenate([[0.0], c])


def pick_x_series(runs: List[Dict], x_mode: str):
    key_map = {
        "epoch": None,
        "gpu_seconds": "gpu_seconds",
        "bp_total": "cum_bp_eq_total",
        "exbp_total": "cum_ex_bp_total",
        "time": "wall_seconds",
    }
    labels = {
        "epoch": "Epoch",
        "gpu_seconds": "Cumulative GPU seconds (per-epoch CUDA timing)",
        "bp_total": "Cumulative backprop equivalents",
        "exbp_total": "Cumulative examples × backprops",
        "time": "Cumulative wall-clock seconds",
    }

    key = key_map.get(x_mode)
    if key is None:
        return None, labels["epoch"]

    series = []
    for run in runs:
        if key not in run:
            continue
        arr = run[key]
        arr = _cum_gpu_seconds(arr) if x_mode == "gpu_seconds" else np.asarray(arr, dtype=float)
        series.append(arr)

    if not series:
        return None, labels["epoch"]

    min_len = min(len(a) for a in series)
    data = np.mean([a[:min_len] for a in series], axis=0)
    return data, labels[x_mode]


def markers_on(n_points: int, curve_idx: int, total_curves: int) -> np.ndarray:
    freq = max(1, n_points // 10)
    shift = (freq // max(1, total_curves)) * curve_idx
    return np.arange(shift, n_points, freq)


def plot_metrics(curves: List[Dict] = None, x_mode: str = "time", truncate_to_fastest: bool = False) -> None:
    """Render all metric panels with optional timeline truncation."""
    configure_style()
    curves = curves or build_curve_specs()
    styled = [ensure_style(spec, idx) for idx, spec in enumerate(curves)]

    for metric_key, (title, ylabel) in METRICS.items():
        prepared = []
        for spec in styled:
            runs = load_many(spec["pattern"])
            if not runs:
                print(f"⚠  no runs match '{spec['pattern']}' – skipping")
                continue

            y = mean_curve(runs, metric_key)
            if y is None:
                print(f"⚠  runs for '{spec['pattern']}' lack '{metric_key}' – skipping")
                continue

            x, xlabel = pick_x_series(runs, x_mode)
            if x is None:
                x = np.arange(0, len(y))
                xlabel = "Epoch"

            min_len = min(len(x), len(y))
            prepared.append(
                {
                    "spec": spec,
                    "x": x[:min_len],
                    "y": y[:min_len],
                    "xlabel": xlabel,
                }
            )

        if not prepared:
            continue

        if truncate_to_fastest and x_mode in TIME_BASED:
            common_xmax = min(item["x"][-1] for item in prepared)
        else:
            common_xmax = None

        fig, ax = plt.subplots()
        for idx, item in enumerate(prepared):
            spec, x, y = item["spec"], item["x"], item["y"]

            if common_xmax is not None:
                mask = x <= common_xmax + 1e-12
                x = x[mask]
                y = y[mask]

            ax.plot(
                x,
                y,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                marker=spec["marker"],
                markevery=markers_on(len(y), idx, len(prepared)),
                markersize=ARG["marker_size"],
                markerfacecolor=spec["color"],
                markeredgecolor="black",
            )

        if metric_key in {"train_loss", "test_loss"}:
            ax.set_yscale("log")

        ax.set_title(title)
        ax.set_xlabel(prepared[0]["xlabel"])
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend()

        if common_xmax is not None:
            ax.set_xlim(0, common_xmax)

        fig.tight_layout()
        pdf_name = f"{metric_key}_{x_mode}.pdf"
        fig.savefig(pdf_name, bbox_inches="tight")
        print(f"✓  saved {pdf_name}")
        plt.show()
