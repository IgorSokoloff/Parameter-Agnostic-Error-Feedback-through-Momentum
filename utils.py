import numpy as np
import os
import glob
import random
import torch
from pickle import load, dump
from collections import defaultdict

SAVED_RUNS_PATH = 'saved_data/'
EXP_PATH = 'exps_setup/'

int_repr_prec = lambda x, prec: int(x) if x.is_integer() else round(x,prec)
myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x)
intrepr = lambda x: int(x) if x.is_integer() else round(x,8)

# def save_run(suffix, run):
#     if not os.path.isdir(SAVED_RUNS_PATH):
#         os.mkdir(SAVED_RUNS_PATH)

#     file = SAVED_RUNS_PATH + suffix + '.pickle'
#     with open(file, 'wb') as f:
#         dump(run, f)

def save_run(suffix, run):
    if not os.path.isdir(SAVED_RUNS_PATH):
        os.mkdir(SAVED_RUNS_PATH)
    file = SAVED_RUNS_PATH + suffix + '.pickle'

    # convert the internal defaultdict logger to a plain dict for pickling
    if "series" in run:   # new-style run
        flat = {k: list(v) for k, v in run["series"].items()}
        flat.update(run.get("meta", {}))   # harmless if empty
        run_to_save = flat
    else:                 # old-style run
        run_to_save = run

    with open(file, 'wb') as f:
        dump(run_to_save, f)


def read_all_runs(exp, suffix=None):
    if suffix is None:
        suffix = exp['name']

    runs = list()
    runs_files = glob.glob(SAVED_RUNS_PATH + suffix + '_' + '[1-9]*.pickle')  # reads at most first ten runs
    for run_file in runs_files:
        runs.append(read_run(run_file))
    return runs


def read_run(file):
    with open(file, 'rb') as f:
        run = load(f)
    return run


# def create_run():
#     run = {'train_loss': [],
#            'test_loss': [],
#            'test_acc': []
#            }
#     return run

def create_run():
    """
    New-style run object:
      run["series"][key] -> list of values (defaultdict(list))
      run["meta"]        -> free-form metadata (optional)
    """
    return {"series": defaultdict(list), "meta": {}}

def log_series(run, **kwargs):
    """
    Append key=value pairs to run["series"]. Skips keys whose value is None.
    """
    series = run["series"]
    for k, v in kwargs.items():
        if v is not None:
            series[k].append(v)

# def update_run(train_loss, test_loss, test_acc, run):
#     run['train_loss'].append(train_loss)
#     run['test_loss'].append(test_loss)
#     run['test_acc'].append(test_acc)

# Backward-compat: keep the old helper so existing calls don't break
def update_run(train_loss, test_loss, test_acc, run):
    log_series(run, train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)


def save_exp(exp):
    if not os.path.isdir(EXP_PATH):
        os.mkdir(EXP_PATH)

    file = EXP_PATH + exp['name'] + '.pickle'
    with open(file, 'wb') as f:
        dump(exp, f)


def load_exp(exp_name):
    file = EXP_PATH + exp_name + '.pickle'
    with open(file, 'rb') as f:
        exp = load(f)
    return exp


def create_exp(
        name,
        dataset,
        net,
        n_workers,
        epochs,
        seed,
        batch_size,
        lrs,                  # list of candidate γ₀ values
        etas,                 # list of candidate η₀ values
        *,
        lr_schedule=None,     # "poly" or None
        eta_schedule=None,    # "poly" or None
        compression,
        error_feedback,
        criterion,
        cuda_device="cuda:0",
        master_compression=None,
        momentum=0,
        weight_decay=0,
        p=None,               # exponent for LR  decay
        q=None                # exponent for ETA decay
):
    """
    Pack all hyper-parameters into a single dict that gets threaded
    through the training pipeline and logged to wandb.

    Parameters marked with '*' are keyword-only for clarity.
    """

    exp = {
        # ---- bookkeeping ----
        "name":            name,
        "dataset_name":    dataset,
        "net":             net,
        "seed":            seed,

        # ---- data / training ----
        "n_workers":       n_workers,
        "epochs":          epochs,
        "batch_size":      batch_size,

        # ---- optimisation ----
        "lrs":             lrs,
        "lr":              None,          # will be filled by tune_step_size
        "etas":            etas,
        "compression":     compression,
        "master_compression": master_compression,
        "error_feedback":  error_feedback,
        "criterion":       criterion,
        "momentum":        momentum,
        "weight_decay":    weight_decay,

        # ---- scheduling ----
        "lr_schedule":     lr_schedule,   # "poly" or None
        "eta_schedule":    eta_schedule,  # "poly" or None
        "p":               p,             # exponent for LR  decay
        "q":               q,             # exponent for ETA decay

        # ---- hardware ----
        "cuda_device":     cuda_device
    }

    return exp


def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
