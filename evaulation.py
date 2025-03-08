from collections import defaultdict
from typing import Tuple

import numpy as np
import spyrmsd
from rdkit import Chem
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, r2_score


def mask_nan(y_true, y_pred):
    mask = ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    return y_true, y_pred


def mol_has_3D(mol):
    try:
        mol.GetConformer()
    except ValueError:
        raise ValueError("Cannot get conformer for molecule, is likely 2D")

    if not mol.GetConformer().Is3D():
        raise ValueError("Molecule is not 3D")


def eval_poses(preds: list[Chem.Mol], refs: list[Chem.Mol], cutoff=2.0) -> Tuple[np.ndarray, float]:
    """
    Evaluate the poses of the predicted molecules against the reference molecules.

    Calculates the % correct with respect to the RMSD cutoff value.

    Parameters
    ----------
    preds : list[Chem.Mol]
        List of predicted molecules.
    refs : list[Chem.Mol]

    cutoff : float
        The cutoff value for the RMSD value. Default is 2.0.

    Returns
    -------
    Tuple[np.ndarray, float]
        Returns a tuple of the RMSD values and the percentage of RMSD values less than the cutoff

    """

    if len(preds) != len(refs):
        raise ValueError("mismatched lengths in preds vs references")

    # find symm corrected  heavy atom RMSDs
    rmsds = []
    for pred, ref in zip(preds, refs):
        # Check the input
        mol_has_3D(pred)
        mol_has_3D(ref)
        if pred.GetNumHeavyAtoms() != ref.GetNumHeavyAtoms():
            raise ValueError("mismatched number of atoms")

        # Compute RMSD
        pred_spy = spyrmsd.molecule.Molecule.from_rdkit(pred)
        ref_spy = spyrmsd.molecule.Molecule.from_rdkit(ref)
        rmsd = spyrmsd.rmsd.rmsdwrapper(ref_spy, pred_spy, symmetry=True, strip=True)
        rmsds.extend(rmsd)

    rmsds = np.asarray(rmsds)

    # calculate % less than cutoff
    mask = rmsds <= cutoff

    correct = sum(mask)
    prob = (correct / rmsds.shape[0]) * 100

    collect = {}
    collect["rmsd_mean"] = np.mean(rmsds)
    collect["rmsd_min"] = np.min(rmsds)
    collect["rmsd_q1"] = np.quantile(rmsds, 0.25)
    collect["rmsd_median"] = np.quantile(rmsds, 0.5)
    collect["rmsd_q3"] = np.quantile(rmsds, 0.75)
    collect["rmsd_max"] = np.max(rmsds)
    collect["rmsd_coverage"] = prob

    return collect


def eval_potency(preds: dict[str, list], refs: dict[str, list]) -> Tuple[dict[str, float]]:
    """
    Eval pIC50 potency performance with MAE (already log10 transformed)

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.
    refs : dict[str, list]
        Dictionary of reference pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """

    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}
    collect = defaultdict(dict)

    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")

        ref, pred = mask_nan(refs[k], preds[k])

        mae = mean_absolute_error(ref, pred)
        ktau = kendalltau(ref, pred)
        r2 = r2_score(ref, pred)

        # subchallenge statistics
        collect[k]["mean_absolute_error"] = mae
        collect[k]["kendall_tau"] = ktau.statistic
        collect[k]["r2"] = r2

    # compute macro average MAE
    macro_mae = np.mean([collect[k]["mean_absolute_error"] for k in keys])
    collect["aggregated"]["macro_mean_absolute_error"] = macro_mae

    # compute macro average R2
    macro_r2 = np.mean([collect[k]["r2"] for k in keys])
    collect["aggregated"]["macro_r2"] = macro_r2

    return collect


def eval_admet(preds: dict[str, list], refs: dict[str, list]) -> Tuple[dict[str, float], np.ndarray]:
    """
    Eval ADMET targets with MAE for pre-log10 transformed targets (LogD) and MALE  (MAE on log10 transformed dataset) on non-transformed data

    This provides a "relative" error metric that will not be as sensitive to the large outliers with huge errors. This is sometimes known as MALE.

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted ADMET values.
    refs : dict[str, list]
        Dictionary of reference ADMET values.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """
    keys = {
        "MLM",
        "HLM",
        "KSOL",
        "LogD",
        "MDR1-MDCKII",
    }
    # will be treated as is
    logscale_endpts = {"LogD"}

    collect = defaultdict(dict)

    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")

        ref, pred = mask_nan(refs[k], preds[k])

        if k in logscale_endpts:
            # already log10scaled
            mae = mean_absolute_error(ref, pred)
            r2 = r2_score(ref, pred)
        else:
            # clip to a detection limit

            # transform both log10scale
            pred_log10s = np.log10(np.clip(pred, a_min=0, a_max=None) + 1)
            ref_log10s = np.log10(np.clip(ref, a_min=0, a_max=None) + 1)

            # compute MALE and R2 in log space
            mae = mean_absolute_error(ref_log10s, pred_log10s)
            r2 = r2_score(ref_log10s, pred_log10s)

        collect[k]["mean_absolute_error"] = mae
        collect[k]["r2"] = r2

    # compute macro average MAE
    macro_mae = np.mean([collect[k]["mean_absolute_error"] for k in keys])
    collect["aggregated"]["macro_mean_absolute_error"] = macro_mae

    # compute macro average R2
    macro_r2 = np.mean([collect[k]["r2"] for k in keys])
    collect["aggregated"]["macro_r2"] = macro_r2

    return collect
