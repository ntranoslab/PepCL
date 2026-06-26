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