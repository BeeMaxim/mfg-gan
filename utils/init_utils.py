import logging
import torch
import numpy as np
import random
import os
import string
import secrets
import subprocess

#from logger.logger import setup_logging
from pathlib import Path


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Args:
        length (int): length of a string.
    Returns:
        run_id (str): base-36 string with an experiment id.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(save_dir):
    """
    Log current git commit and patch to save dir.
    Improves reproducibility by allowing to run the same code version:
        git checkout commit_hash_from_commit_path
        git apply patch_path

    If you created new files and want to have them in patch,
    stage them via git add before running the script.

    Patch can be applied via the following command:
        git apply patch_path

    Args:
        save_dir (Path): directory to save patch and commit in
    """
    print("Logging git commit and patch...")
    commit_path = save_dir / "git_commit.txt"
    patch_path = save_dir / "git_diff.patch"
    with commit_path.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_path.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def setup_saving_and_logging(config):
    """
    Initialize the logger, writer, and saving directory.
    The saving directory is defined by the run_name and save_dir
    arguments of config.writer and config.trainer, respectfully.

    Args:
        config (DictConfig): hydra config for the current experiment.
    Returns:
        logger (Logger): logger that logs output.
    """

    #setup_logging(save_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    return logger
