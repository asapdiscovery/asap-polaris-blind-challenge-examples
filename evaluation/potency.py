import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from evaluation.bootstrapping import bootstrapping_sampler
from evaluation.cld import add_cld_to_leaderboard
from evaluation.utils import (
    compute_macro_metrics,
    mask_flagged,
    mask_nan,
    scores_to_leaderboards,
)


def evaluate_potency_predictions(
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
    method_label: str,
    n_bootstrap_samples: int = 1000,
) -> pd.DataFrame:
    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}

    scores = pd.DataFrame(columns=["Target Label", "Metric", "Score", "Bootstrap Iteration"])

    for target_label in keys:
        if target_label not in y_pred.keys() or target_label not in y_true.keys():
            raise ValueError("required key not present")

        refs = y_true[target_label]
        pred = y_pred[target_label]

        refs, pred = mask_nan(refs, pred)
        refs, pred = mask_flagged(refs, pred, "potency", target_label)

        for i, ind in enumerate(
            bootstrapping_sampler(refs.shape[0], n_bootstrap_samples)
        ):
            collect = {
                "mean_absolute_error": mean_absolute_error(
                    y_true=refs[ind], y_pred=pred[ind]
                ),
                "mean_squared_error": mean_squared_error(
                    y_true=refs[ind], y_pred=pred[ind]
                ),
                "pearsonr": pearsonr(refs[ind], pred[ind])[0],
                "spearmanr": spearmanr(refs[ind], pred[ind])[0],
                "r2": r2_score(y_true=refs[ind], y_pred=pred[ind]),
                "kendall_tau": kendalltau(refs[ind], pred[ind]).statistic,
            }
            for metric, score in collect.items():
                scores.loc[len(scores)] = [target_label, metric, score, i]

    # Macro records
    macro_scores = compute_macro_metrics(scores)
    scores = pd.concat([scores, macro_scores])

    # Add metadata
    scores["Test Set"] = "test"
    scores["Method"] = method_label
    return scores


def evaluate_all_potency_predictions(
    y_true: dict[str, np.ndarray], all_y_pred: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    """
    Evaluate and rank all submissions

    Parameters
    ----------
    y_true : dict[str, np.ndarray]
        The true values.
    all_y_pred : dict[str, dict[str, np.ndarray]]
        The predictions. The key in the top-level dictionary is a unique identifier for each submission.
    """

    all_scores = pd.DataFrame()

    for method_label, y_pred in all_y_pred.items():
        scores = evaluate_potency_predictions(y_true, y_pred, method_label)
        all_scores = pd.concat([all_scores, scores], ignore_index=True)

    leaderboards = scores_to_leaderboards(
        all_scores, rank_by="mean_absolute_error", ascending=True
    )

    main_leaderboard = add_cld_to_leaderboard(
        leaderboards["aggregated"],
        all_scores,
        "mean_absolute_error",
        "aggregated",
    )

    return main_leaderboard, leaderboards
