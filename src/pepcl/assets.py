from pathlib import Path


def get_repo_root() -> Path:
    """
    Return the current working directory as the assumed repository root.

    This is intentionally simple because PepCL data are external assets
    downloaded by the user, not package data.
    """
    return Path.cwd()


def get_default_data_dir(base_dir=None) -> Path:
    """
    Default location for downloaded PepCL data assets.
    """
    if base_dir is None:
        base_dir = get_repo_root()
    return Path(base_dir) / "pepcl_data"


def get_default_output_dir(base_dir=None) -> Path:
    """
    Default location for PepCL outputs.
    """
    if base_dir is None:
        base_dir = get_repo_root()
    return Path(base_dir) / "pepcl_outputs"


def get_default_checkpoint_dir(base_dir=None) -> Path:
    """
    Default location for PepCL-updated model checkpoints.
    """
    if base_dir is None:
        base_dir = get_repo_root()
    return Path(base_dir) / "model_checkpoints"


def require_path(path, *, description="required file") -> Path:
    """
    Resolve a path and raise a clear error if it does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n\n"
            "PepCL uses external data assets that are not stored in this GitHub "
            "repository. Download the PepCL data bundle and place it under "
            "'pepcl_data/', or pass an explicit path."
        )

    return path