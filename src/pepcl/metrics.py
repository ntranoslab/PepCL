import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from scipy.stats import spearmanr

def _raw_auc01(y_true, y_score, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if not np.any(np.isclose(fpr, max_fpr)):
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr = np.append(fpr, max_fpr)
        tpr = np.append(tpr, tpr_at_max)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    mask = fpr <= max_fpr
    return auc(fpr[mask], tpr[mask])


def _ppvn(y_true, y_score):
    n = int(np.sum(y_true == 1))
    if n <= 0:
        return np.nan
    n = min(n, len(y_true))
    top_idx = np.argsort(-y_score, kind="mergesort")[:n]
    return np.mean(y_true[top_idx] == 1)


def compute_metric_matrix(
    df,
    score_cols,
    method,
    label_col="label",
    allele_col="allele",
    reference_col=None,
    *,
    auc01_max_fpr=0.1,
    auc01_normalized=True,
    return_float64=True,
    disable_tqdm=False,
):
    valid_methods = {"ap", "auc", "auc01", "ppvn", "spearman"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got: {method}")

    is_spearman = method == "spearman"

    if is_spearman:
        if reference_col is None:
            raise ValueError("reference_col must be provided when method='spearman'.")
        target_col = reference_col
        score_cols = [c for c in score_cols if c in df.columns and c != reference_col]
    else:
        target_col = label_col
        score_cols = [c for c in score_cols if c in df.columns]

    alleles = sorted(df[allele_col].unique())
    groups = df.groupby(allele_col, sort=False).indices

    target_all = df[target_col].to_numpy()
    score_arrays = [df[c].to_numpy() for c in score_cols]

    dtype = np.float64 if return_float64 else float
    metric_values = np.full((len(alleles), len(score_cols)), np.nan, dtype=dtype)

    for i, allele in enumerate(tqdm(alleles, disable=disable_tqdm)):
        idx = groups.get(allele)
        if idx is None or len(idx) == 0:
            continue

        t = target_all[idx]
        t_notna = ~pd.isna(t)

        for j, s_arr in enumerate(score_arrays):
            s = s_arr[idx]
            valid = t_notna & (~pd.isna(s))

            if valid.sum() <= 1:
                continue

            t_valid = t[valid]
            s_valid = s[valid]

            try:
                if is_spearman:
                    if len(np.unique(t_valid)) < 2 or len(np.unique(s_valid)) < 2:
                        continue
                    metric_values[i, j] = spearmanr(t_valid, s_valid).correlation
                else:
                    if len(np.unique(t_valid)) < 2:
                        continue
                    if method == "ap":
                        metric_values[i, j] = average_precision_score(t_valid, s_valid)
                    elif method == "auc":
                        metric_values[i, j] = roc_auc_score(t_valid, s_valid)
                    elif method == "auc01":
                        if auc01_normalized:
                            metric_values[i, j] = roc_auc_score(t_valid, s_valid, max_fpr=auc01_max_fpr)
                        else:
                            metric_values[i, j] = _raw_auc01(t_valid, s_valid, max_fpr=auc01_max_fpr)
                    elif method == "ppvn":
                        metric_values[i, j] = _ppvn(t_valid, s_valid)
            except Exception:
                metric_values[i, j] = np.nan

    return pd.DataFrame(metric_values, index=alleles, columns=score_cols)