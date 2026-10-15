def test_package_imports_and_exports():
    import pepcl

    assert isinstance(pepcl.__version__, str)
    assert len(pepcl.__version__) > 0

    expected_exports = [
        "PepCL",
        "PepCL_Micro",
        "prepare_pepcl_training_dataframe",
        "clear_memory",
        "get_device",
        "load_pickle",
        "print_device_info",
        "resolve_csv_path",
        "save_pickle",
        "set_seed",
        "get_default_pepcl_params",
        "get_default_pepcl_micro_params",
        "compute_metric_matrix",
        "plot_training_losses",
        "plot_bars_or_boxplots",
        "plot_bars_or_boxplots_grid",
    ]

    for name in expected_exports:
        assert hasattr(pepcl, name), f"pepcl is missing expected export: {name}"


def test_default_pepcl_params_have_expected_values():
    from pepcl import get_default_pepcl_params

    params = get_default_pepcl_params()

    assert params["steps"] == 300
    assert params["num_workers"] == 16
    assert params["num_ms_pos"] == 2000
    assert params["num_nonms_pos"] == 2000
    assert params["ms_batch_size"] == 3072
    assert params["nonms_batch_size"] == 3072
    assert params["encoder_lr"] == 2e-4
    assert params["decoder_lr"] == 2e-4
    assert params["ms_loss_type"] == "logsmoothap"
    assert params["nonms_loss_type"] == "logsmoothap"
    assert params["dr_lambda"] == 1e-5
    assert params["dr_loss_type"] == "mse"


def test_default_pepcl_params_allow_overrides():
    from pepcl import get_default_pepcl_params

    params = get_default_pepcl_params(
        steps=3,
        encoder_lr=1e-4,
        custom_runtime_value="allowed",
    )

    assert params["steps"] == 3
    assert params["encoder_lr"] == 1e-4
    assert params["custom_runtime_value"] == "allowed"


def test_default_pepcl_micro_params_have_expected_values():
    import torch
    from pepcl import get_default_pepcl_micro_params

    params = get_default_pepcl_micro_params()

    assert params["execution_mode"] == "microbatch"
    assert params["steps"] == 300
    assert params["num_workers"] == 4
    assert params["num_ms_pos"] == 2000
    assert params["num_nonms_pos"] == 2000
    assert params["ms_batch_size"] == 3072
    assert params["nonms_batch_size"] == 3072
    assert params["ms_microbatch_size"] == 512
    assert params["nonms_microbatch_size"] == 512
    assert params["micro_strategy"] == "auto"
    assert params["cache_forward_train_mode"] is True
    assert params["preserve_cache_rng_state"] is True
    assert params["use_amp"] is True
    assert params["amp_dtype"] == torch.float16
    assert params["empty_cache_between_streams"] is False
    assert params["fail_on_nonfinite_loss"] is True


def test_default_pepcl_micro_params_allow_overrides():
    from pepcl import get_default_pepcl_micro_params

    params = get_default_pepcl_micro_params(
        steps=5,
        ms_microbatch_size=128,
        nonms_microbatch_size=256,
    )

    assert params["steps"] == 5
    assert params["ms_microbatch_size"] == 128
    assert params["nonms_microbatch_size"] == 256