from .version import __version__

from .pepcl import PepCL, PepCL_Micro
from .preprocessing import prepare_pepcl_training_dataframe
from .utils import (
    clear_memory,
    get_device,
    load_pickle,
    print_device_info,
    resolve_csv_path,
    save_pickle,
    set_seed,
    get_default_pepcl_params,
    get_default_pepcl_micro_params,
)

from .metrics import (
    compute_metric_matrix,
)

from .plotting import (
    plot_training_losses,
    plot_bars_or_boxplots,
    plot_bars_or_boxplots_grid,
)

__all__ = [
    "__version__",

    # core trainers
    "PepCL",
    "PepCL_Micro",

    # preprocessing
    "prepare_pepcl_training_dataframe",

    # utilities
    "clear_memory",
    "get_device",
    "load_pickle",
    "print_device_info",
    "resolve_csv_path",
    "save_pickle",
    "set_seed",
    "get_default_pepcl_params",
    "get_default_pepcl_micro_params",

    # metrics
    "compute_metric_matrix",

    # plotting
    "plot_training_losses",
    "plot_bars_or_boxplots",
    "plot_bars_or_boxplots_grid",
]