# Parameter-Agnostic-Error-Feedback-through-Momentum

Supporting code for **Improved Convergence in Parameter-Agnostic Error Feedback through Momentum (2025)**. The repo contains (i) the exact Conda/pip environment used in the paper, (ii) release pickles for every tuned experiment, (iii) reproducible plotting utilities, and (iv) scripts to rerun the training sweeps.

---

## 1. Recreate the software environment

Two Conda specs ship with the repo (both already embed the pip dependencies, so no extra `pip install` step is required):

| File                   | When to use                                                                                   | Commands                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `environment.yml`      | Portable spec with only the packages we explicitly installed.                                 | `conda env create -f environment.yml && conda activate ef21-hess`      |
| `environment.full.yml` | Bit-for-bit snapshot of the camera-ready environment (channel priorities, CUDA builds, etc.). | `conda env create -f environment.full.yml && conda activate ef21-hess` |

---

## 2. `reproduction.ipynb`: plots & experiments

Open the notebook inside the environment and follow the sections in order:

1. **Runtime sanity check** – prints the detected CUDA devices and versions so you can confirm `ef21-hess` is active.
2. **Figures 1–4** – uses `repro_plots.py` to reload `_release_data/*.pickle` and regenerate every plot from the paper (`X_MODE` toggles the horizontal axis).
3. **Run new experiments** – imports `run_methods` from `repro_training.py`, letting you launch one-off or short sweeps directly from the notebook by editing only the `METHODS` list, epochs, and the `topk_ratio`.
4. **Batch automation** – documents `run_all_experiments.sh`, which sequentially queues every tuned method inside detached `tmux` sessions (`python -m repro_training --method ...`). Adjust the `METHODS`, `CONDA_ENV`, and `PROJECT_NAME` variables near the top of the script before running `./run_all_experiments.sh`.

> Need to refresh the plots after new training runs? Drop the newly produced pickles into `_release_data/` (or update the `folder` argument in `repro_plots.build_curve_specs`) and rerun the plotting cells.

> **W&B login**: runs inherit whatever credentials are cached on the host. Before starting any training session, execute `wandb login --relogin` (or `wandb login` if the machine has never been authenticated) and paste the key from https://wandb.ai/authorize. To keep everything local, run `wandb offline` instead. All commands should be issued in the shell where `conda activate ef21-hess` was executed.

---

## 3. Running/tuning experiments

- `repro_training.py` exposes `default_gen_config`, `tuned_presets`, and `run_methods`. Import them inside notebooks or call them from the CLI (`python -m repro_training --method EF21_RHM_NORM --method EF21_MVR_NORM ...`) to replicate or extend the paper’s sweeps.
- For unattended runs we use `tmux` via `run_all_experiments.sh`. Each method is executed in its own session, logs are saved to `train_<METHOD>_<timestamp>.log`, and the script waits for one job to finish before launching the next—mirroring the methodology described in the manuscript.
- The underlying training pipeline lives in `train.py`, `prep_data.py`, `models.py`, `gen_sgd.py`, and `quant.py`. Feel free to script against those modules directly if you need custom compressors or models; the wrappers above simply capture the combinations we benchmarked.

---

## 4. Release data layout

The `_release_data/` directory stores the metric pickles used to draw every figure (train/test loss, accuracy, cumulative wall/gpu time, etc.). Each filename encodes the network, dataset, EF variant, and hyper-parameters (`resnet18_cifar10_EF21_IGT_NORM_topk-0,1_lr-0,1_eta-1,0_p-None_q-0,57.pickle`). Loading them through `repro_plots.py` or your own scripts is enough to recreate the figures—no retraining required unless you want to explore new settings.

---

Questions or issues? Please open a ticket so we can keep the public release in sync with the paper.
