import warnings
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from mhcprime import (
    create_seq_allele_column,
    full_process_data,
    load_default_mhc_pseudosequences,
    print_data_stats,
)


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    df_name: str,
    required_cols: Sequence[str],
) -> None:
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{df_name} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _remove_sequence_padding(
    df: pd.DataFrame,
    *,
    seq_col: str = "seq",
    padding_chars: Sequence[str] = ("$",),
) -> pd.DataFrame:
    """
    Remove explicit sequence padding characters from peptide sequences.

    This is intentionally done before seq_len calculation and before
    MHCPrime processing.
    """
    df = df.copy()
    seq = df[seq_col].fillna("").astype(str)

    for pad_char in padding_chars:
        seq = seq.str.replace(pad_char, "", regex=False)

    df[seq_col] = seq.str.strip()
    return df


def _semi_join_on_allele_length(
    left: pd.DataFrame,
    right_groups: pd.DataFrame,
    *,
    allele_col: str = "allele",
    len_col: str = "seq_len",
) -> pd.DataFrame:
    """
    Keep rows in left whose allele-length pair appears in right_groups.
    """
    if right_groups.empty:
        return left.iloc[0:0].copy()

    keep_keys = right_groups[[allele_col, len_col]].drop_duplicates()
    out = left.merge(
        keep_keys.assign(_pepcl_keep_key=True),
        on=[allele_col, len_col],
        how="left",
    )

    out = out[out["_pepcl_keep_key"].eq(True)].drop(
    columns=["_pepcl_keep_key"]
    )

    return out.reset_index(drop=True)


def prepare_pepcl_training_dataframe(
    ms_train_df: pd.DataFrame,
    new_train_df: pd.DataFrame,
    *,
    mhc_seq_dict: Optional[Dict[str, str]] = None,
    seq_col: str = "seq",
    allele_col: str = "allele",
    label_col: str = "label",
    ms_domain_id: int = 0,
    new_domain_id: int = 1,
    ms_method_group: str = "MS",
    new_method_group: str = "nonMS",
    match_ms_to_new_allele_length: bool = True,
    drop_new_unmatched_allele_length: bool = False,
    process: bool = True,
    create_seq_allele: bool = True,
    remove_flank: bool = True,
    add_pseudo_label: bool = False,
    padding_chars: Sequence[str] = ("$",),
    print_stats: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare a merged two-domain PepCL training dataframe.

    Domain convention:
        MS/replay data     -> domain_id = 0
        New/fine-tune data -> domain_id = 1

    Main workflow:
        1. Copy MS and new dataframes.
        2. Validate required columns.
        3. Remove explicit sequence padding characters from seq.
        4. Assign domain_id and method_group.
        5. Create seq_len from unprocessed/unpadded peptide sequence.
        6. Optionally match the MS domain to allele-length groups present
           in the new domain.
        7. Optionally remove new-domain allele-length groups absent from MS.
        8. Concatenate and reset index.
        9. Optionally run MHCPrime full_process_data().
        10. Optionally create seq_allele.

    Notes:
        - Matching does not downsample counts. It filters MS rows to the set
          of allele-length combinations present in the new data.
        - New allele-length groups absent from MS are kept by default.
        - This function does not remove seq_allele duplicates.
    """
    required_cols = [seq_col, allele_col, label_col]

    _log("[PepCL] Validating input dataframes...", verbose=verbose)
    _validate_required_columns(
        ms_train_df,
        df_name="ms_train_df",
        required_cols=required_cols,
    )
    _validate_required_columns(
        new_train_df,
        df_name="new_train_df",
        required_cols=required_cols,
    )

    ms_df = ms_train_df.copy()
    new_df = new_train_df.copy()

    _log("[PepCL] Removing explicit peptide padding characters...", verbose=verbose)
    ms_df = _remove_sequence_padding(
        ms_df,
        seq_col=seq_col,
        padding_chars=padding_chars,
    )
    new_df = _remove_sequence_padding(
        new_df,
        seq_col=seq_col,
        padding_chars=padding_chars,
    )

    _log("[PepCL] Assigning domain IDs...", verbose=verbose)
    ms_df["domain_id"] = int(ms_domain_id)
    new_df["domain_id"] = int(new_domain_id)

    ms_df["method_group"] = ms_method_group
    new_df["method_group"] = new_method_group

    _log("[PepCL] Creating peptide length column...", verbose=verbose)
    ms_df["seq_len"] = ms_df[seq_col].astype(str).str.len()
    new_df["seq_len"] = new_df[seq_col].astype(str).str.len()

    if match_ms_to_new_allele_length:
        _log(
            "[PepCL] Matching MS data to new-data allele-length groups...",
            verbose=verbose,
        )

        new_groups = (
            new_df[[allele_col, "seq_len"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        ms_groups = (
            ms_df[[allele_col, "seq_len"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        common_groups = new_groups.merge(
            ms_groups,
            on=[allele_col, "seq_len"],
            how="inner",
        )

        n_new_groups = len(new_groups)
        n_ms_groups = len(ms_groups)
        n_common_groups = len(common_groups)
        n_unmatched_new_groups = n_new_groups - n_common_groups

        _log(
            (
                "[PepCL] Allele-length groups: "
                f"MS={n_ms_groups:,}, new={n_new_groups:,}, "
                f"shared={n_common_groups:,}, "
                f"new_without_MS_match={n_unmatched_new_groups:,}"
            ),
            verbose=verbose,
        )

        if n_common_groups == 0:
            warnings.warn(
                "No allele-length groups are shared between MS and new data. "
                "The matched MS domain will be empty. New-domain rows are kept "
                "unless drop_new_unmatched_allele_length=True.",
                UserWarning,
            )

        ms_before = len(ms_df)
        ms_df = _semi_join_on_allele_length(
            ms_df,
            new_groups,
            allele_col=allele_col,
            len_col="seq_len",
        )
        ms_after = len(ms_df)

        _log(
            f"[PepCL] MS rows after allele-length matching: "
            f"{ms_after:,} / {ms_before:,}",
            verbose=verbose,
        )

        if drop_new_unmatched_allele_length:
            _log(
                "[PepCL] Removing new-data allele-length groups absent from MS...",
                verbose=verbose,
            )

            new_before = len(new_df)
            new_df = _semi_join_on_allele_length(
                new_df,
                common_groups,
                allele_col=allele_col,
                len_col="seq_len",
            )
            new_after = len(new_df)

            _log(
                f"[PepCL] New-data rows after removing unmatched groups: "
                f"{new_after:,} / {new_before:,}",
                verbose=verbose,
            )

    else:
        _log(
            "[PepCL] Skipping allele-length matching; merging all MS and new rows.",
            verbose=verbose,
        )

    if len(ms_df) == 0:
        warnings.warn(
            "The MS/replay domain is empty after filtering. This is allowed by "
            "the merge function, but PepCL replay training requires domain_id=0 "
            "rows. Use use_replay=False or adjust matching/filtering settings.",
            UserWarning,
        )

    if len(new_df) == 0:
        warnings.warn(
            "The new/fine-tuning domain is empty after filtering. This is allowed "
            "by the merge function, but PepCL fine-tuning requires domain_id=1 rows.",
            UserWarning,
        )

    _log("[PepCL] Concatenating domains and resetting index...", verbose=verbose)
    merged_df = pd.concat([ms_df, new_df], axis=0, ignore_index=True)
    merged_df.reset_index(inplace=True, drop=True)

    if print_stats:
        _log("[PepCL] Merged dataframe statistics before MHCPrime processing:", verbose=verbose)
        print_data_stats(merged_df)

    if process:
        _log("[PepCL] Running MHCPrime full_process_data()...", verbose=verbose)

        if mhc_seq_dict is None:
            _log("[PepCL] Loading packaged MHCPrime pseudosequences...", verbose=verbose)
            mhc_seq_dict = load_default_mhc_pseudosequences()

        merged_df = full_process_data(
            merged_df,
            mhc_seq_dict,
            remove_flank=remove_flank,
            add_pseudo_label=add_pseudo_label,
        )

        if create_seq_allele:
            _log("[PepCL] Creating seq_allele column...", verbose=verbose)
            merged_df = create_seq_allele_column(merged_df)

        merged_df.reset_index(inplace=True, drop=True)

        if print_stats:
            _log("[PepCL] Merged dataframe statistics after MHCPrime processing:", verbose=verbose)
            print_data_stats(merged_df)

    else:
        _log("[PepCL] Skipping MHCPrime processing.", verbose=verbose)

    return merged_df