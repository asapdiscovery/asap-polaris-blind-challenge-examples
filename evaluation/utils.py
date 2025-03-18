import json
from typing import Literal
import numpy as np
import pandas as pd

def mask_nan(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mask out NaN values.

    Since the challenge includes multi-task tasks, some of the ground truth values can be NaN.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : int
        Predicted labels.
    """
    mask = ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    return y_true, y_pred


def mask_flagged(y_true: np.ndarray, y_pred: np.ndarray, task_label: Literal['admet', 'potency', 'ligand-poses'], target_label: str):
    """
    Mask data points that were flagged during the competition and are to be excluded.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : int
        Predicted labels.
    task_label : str
        The task label to mask for.
    target_label : str
        The target label to mask for.
    """
    with open(f"evaluation/data/exclusions/{task_label}.json") as fd:
        flagged = json.load(fd)
    
    flagged = flagged[target_label]
    flagged = [int(x["idx"]) for x in flagged]

    mask = ~np.isin(np.arange(len(y_true)), flagged)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    return y_true, y_pred


def clip_and_log_transform(y: np.ndarray): 
    """
    Clip to a detection limit and transform to log10 scale.

    Parameters
    ----------
    y : np.ndarray
        The array to be clipped and transformed.
    """
    y = np.clip(y, a_min=0, a_max=None)
    return np.log10(y + 1)


def compute_macro_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the macro metrics.
    """
    df = scores.groupby(["Metric", "Bootstrap Iteration"])["Score"].mean()
    df = df.reset_index()
    df["Target Label"] = "aggregated"
    return df


def scores_to_leaderboards(
    scores: pd.DataFrame, rank_by: str = "mean_absolute_error", ascending: bool = True
):
    """
    Convert scores to leaderboards.
    
    Parameters
    ----------
    scores : pd.DataFrame
        The DataFrame containing the scores.
    rank_by : str
        The metric to rank by.
    ascending : bool
        Whether to rank in ascending or descending order
    """
    leaderboards = {}
    for name, group in scores.groupby("Target Label"):
        raw_leaderboard = (
            group.pivot_table(index=["Method"], columns="Metric", values="Score", aggfunc=["mean", "std"])
            .reset_index()
            .sort_values(by=("mean", rank_by), ascending=ascending)
        )

        metrics = raw_leaderboard["mean"].columns
        leaderboard = pd.DataFrame({"Method": raw_leaderboard["Method"]})

        for metric in metrics:
            leaderboard[metric] = raw_leaderboard[[("mean", metric), ("std", metric)]].apply(
                lambda x: f"{x.iloc[0]:.3f} ± {x.iloc[1]:.3f}", axis=1
            )

        leaderboards[name] = leaderboard

    return leaderboards