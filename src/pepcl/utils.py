import gc
import pickle
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def get_device(
    device: Optional[Union[str, torch.device]] = None,
    *,
    prefer_cuda: bool = True,
) -> torch.device:
    """
    Resolve a torch device.

    If device is provided, return that device.
    Otherwise, use CUDA when available and prefer_cuda=True, else CPU.
    """
    if device is not None:
        return torch.device(device)

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve and print basic device information.
    """
    device = get_device(device)

    print("Device:", device)

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print("GPU:", torch.cuda.get_device_name(idx))

    return device


def clear_memory() -> None:
    """
    Clear Python and CUDA memory caches.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def set_seed(seed: int = 42) -> None:
    """
    Set common Python, NumPy, and Torch random seeds.

    This does not force deterministic CUDA kernels, which can reduce performance
    and may not be supported for all operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_pickle(obj, path: Union[str, Path]) -> None:
    """
    Save an object as a pickle file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]):
    """
    Load a pickle file.
    """
    path = Path(path)

    with open(path, "rb") as handle:
        return pickle.load(handle)


def resolve_csv_path(path: Union[str, Path]) -> Path:
    """
    Resolve a CSV path that may exist as either .csv or .csv.gz.

    Examples:
        resolve_csv_path("file.csv.gz") will accept:
            file.csv.gz
            file.csv

        resolve_csv_path("file.csv") will accept:
            file.csv
            file.csv.gz
    """
    path = Path(path)

    if path.exists():
        return path

    # file.csv.gz -> file.csv
    if path.suffixes[-2:] == [".csv", ".gz"]:
        uncompressed = path.with_suffix("")
        if uncompressed.exists():
            return uncompressed

    # file.csv -> file.csv.gz
    if path.suffix == ".csv":
        compressed = path.with_suffix(path.suffix + ".gz")
        if compressed.exists():
            return compressed

    raise FileNotFoundError(
        f"Could not find file: {path}\n"
        "Checked compressed/uncompressed CSV variants. "
        "Make sure the PepCL data bundle is downloaded under pepcl_data/."
    )

def get_default_pepcl_params(**overrides):
    """
    Return default parameters for standard PepCL training.

    Runtime-specific objects such as tokenizer, device, and collate_fn are
    intentionally excluded and should be passed explicitly in the notebook or
    calling script.

    Parameters can be overridden by passing keyword arguments.
    """
    params = dict(
        # steps
        steps=300,
        output_dir=None,
        save_every_n_steps=2000,
        seed=42,
        num_workers=16,

        # sampling
        num_ms_pos=2000,
        num_nonms_pos=2000,
        ms_neg_ratio=1.0,
        nonms_neg_ratio=1.0,

        auto_calibrate=True,
        calibration_steps=1,

        ms_batch_size=3072,
        nonms_batch_size=3072,

        # optimization
        encoder_lr=2e-4,
        decoder_lr=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        update_domain_embeddings=False,

        # continual learning parameters
        primary_task_id=1,
        sequential_backprop=True,

        # losses
        ms_loss_type="logsmoothap",
        ms_loss_alpha=10.0,
        ms_lambda=1.0,
        nonms_loss_type="logsmoothap",
        nonms_loss_alpha=10.0,
        ft_lambda=1.0,

        # drift regularization
        dr_lambda=1e-5,
        dr_loss_type="mse",
    )

    params.update(overrides)
    return params


def get_default_pepcl_micro_params(**overrides):
    """
    Return default parameters for PepCL microbatch training.

    Runtime-specific objects such as tokenizer, device, and collate_fn are
    intentionally excluded and should be passed explicitly in the notebook or
    calling script.

    Parameters can be overridden by passing keyword arguments.
    """
    import torch

    params = dict(
        # execution mode
        execution_mode="microbatch",

        # steps
        steps=300,
        output_dir=None,
        save_every_n_steps=2000,
        seed=42,
        num_workers=4,

        # sampling / logical loss batch size
        num_ms_pos=2000,
        num_nonms_pos=2000,
        ms_neg_ratio=1.0,
        nonms_neg_ratio=1.0,

        auto_calibrate=True,
        calibration_steps=1,

        ms_batch_size=3072,
        nonms_batch_size=3072,

        # physical GPU microbatch size
        ms_microbatch_size=512,
        nonms_microbatch_size=512,

        # optimization
        encoder_lr=2e-4,
        decoder_lr=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        update_domain_embeddings=False,

        # continual learning parameters
        primary_task_id=1,
        sequential_backprop=True,

        # losses
        ms_loss_type="logsmoothap",
        ms_loss_alpha=10.0,
        ms_lambda=1.0,
        nonms_loss_type="logsmoothap",
        nonms_loss_alpha=10.0,
        ft_lambda=1.0,

        # drift regularization
        dr_lambda=1e-5,
        dr_loss_type="mse",

        # PepCL_Micro behavior
        micro_strategy="auto",
        cache_forward_train_mode=True,
        preserve_cache_rng_state=True,

        # AMP
        use_amp=True,
        amp_dtype=torch.float16,

        # memory behavior
        empty_cache_between_streams=False,

        # debugging / safety
        fail_on_nonfinite_loss=True,
    )

    params.update(overrides)
    return params