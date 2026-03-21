from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    m = len(p_values)
    return p_values <= (alpha / m)


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Thresholds: alpha / (m - k + 1) for k from 1 to m
    thresholds = alpha / (m - np.arange(m))

    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m):
        if sorted_pvals[k] > thresholds[k]:
            break
        reject_sorted[k] = True

    # Reconstruct original order
    rejections = np.zeros(m, dtype=bool)
    rejections[sorted_indices] = reject_sorted
    return rejections


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Thresholds: (k / m) * alpha for k from 1 to m
    thresholds = (np.arange(1, m + 1) / m) * alpha

    # Find the largest rank k satisfyinf the condition
    valid_indices = np.where(sorted_pvals <= thresholds)[0]

    reject_sorted = np.zeros(m, dtype=bool)
    if len(valid_indices) > 0:
        k_max = valid_indices[-1]
        reject_sorted[:k_max + 1] = True

    rejections = np.zeros(m, dtype=bool)
    rejections[sorted_indices] = reject_sorted
    return rejections


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Harmonic correction term
    c_m = np.sum(1.0 / np.arange(1, m+1))

    # Thresholds: (k / (m * c-m)) * alpha for k from 1 to m
    thresholds = (np.arange(1, m+1) / (m * c_m)) * alpha

    valid_indices = np.where(sorted_pvals <= thresholds)[0]

    reject_sorted = np.zeros(m, dtype=bool)
    if len(valid_indices) > 0:
        k_max = valid_indices[-1]
        reject_sorted[:k_max + 1] = True

    rejections = np.zeros(m, dtype=bool)
    rejections[sorted_indices] = reject_sorted
    return rejections


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    # Fraction of rows with at least one True
    return float(np.mean(np.any(rejections_null, axis=1)))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    total_discoveries = np.sum(rejections)
    if total_discoveries == 0:
        return 0.0

    false_discoveries = np.sum(rejections & is_true_null)
    return float(false_discoveries / total_discoveries)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    total_false_nulls = np.sum(~is_true_null)
    if total_false_nulls == 0:
        return 0.0

    true_rejections = np.sum(rejections & ~is_true_null)
    return float(true_rejections / total_false_nulls)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    # 1. Process FWER using complete-null simulations
    # Pivot to get an [L, M] matrix of p-values
    null_matrix = null_pvalues.pivot(
        index='sim_id', columns='hypothesis_id', values='p_value'
    ).values

    rej_unc_null = null_matrix <= alpha
    rej_bonf_null = np.array([bonferroni_rejections(row, alpha) for row in null_matrix])
    rej_holm_null = np.array([holm_rejections(row, alpha) for row in null_matrix])

    fwer_unc = compute_fwer(rej_unc_null)
    fwer_bonf = compute_fwer(rej_bonf_null)
    fwer_holm = compute_fwer(rej_holm_null)

    # 2. Process FDR and Power using mixed simulations
    fdr_unc_list, fdr_bh_list, fdr_by_list = [], [], []
    pow_unc_list, pow_bh_list, pow_by_list = [], [], []

    for _, group in mixed_pvalues.groupby("sim_id"):
        # Ensure hypotheses are sorted to align p-values and is_true_null arrays
        group_sorted = group.sort_values('hypothesis_id')
        p_vals = group_sorted["p_value"].values
        is_null = group_sorted["is_true_null"].values

        # Uncorrected
        rej_unc = p_vals <= alpha
        fdr_unc_list.append(compute_fdr(rej_unc, is_null))
        pow_unc_list.append(compute_power(rej_unc, is_null))

        # Benhamini-Hochberg
        rej_bh = benjamini_hochberg_rejections(p_vals, alpha)
        fdr_bh_list.append(compute_fdr(rej_bh, is_null))
        pow_bh_list.append(compute_power(rej_bh, is_null))

        # Benjamini-Yekutieli
        rej_by = benjamini_yekutieli_rejections(p_vals, alpha)
        fdr_by_list.append(compute_fdr(rej_by, is_null))
        pow_by_list.append(compute_power(rej_by, is_null))
        
    # Average across all simulations
    return {
        "fwer_uncorrected": float(fwer_unc),
        "fwer_bonferroni": float(fwer_bonf),
        "fwer_holm": float(fwer_holm),
        "fdr_uncorrected": float(np.mean(fdr_unc_list)),
        "fdr_bh": float(np.mean(fdr_bh_list)),
        "fdr_by": float(np.mean(fdr_by_list)),
        "power_uncorrected": float(np.mean(pow_unc_list)),
        "power_bh": float(np.mean(pow_bh_list)),
        "power_by": float(np.mean(pow_by_list)),
    }