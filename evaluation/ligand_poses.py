import base64
import pandas as pd
from evaluation.bootstrapping import bootstrapping_sampler
from evaluation.cld import add_cld_to_leaderboard
from evaluation.utils import mask_flagged, scores_to_leaderboards
import spyrmsd.molecule
import spyrmsd.rmsd

import numpy as np
from rdkit import Chem


def mol_has_3D(mol):
    try:
        mol.GetConformer()
    except ValueError:
        raise ValueError("Cannot get conformer for molecule, is likely 2D")

    if not mol.GetConformer().Is3D():
        raise ValueError("Molecule is not 3D")


def evaluate_ligand_poses_predictions(
    y_pred: list[Chem.Mol],
    y_true: list[Chem.Mol],
    method_label: str,
    inputs: pd.DataFrame,
    cutoff: float = 2.0,
    n_bootstrap_samples: int = 1000,
):
    rmsds = []

    y_true, y_pred = mask_flagged(y_true, y_pred, "ligand-poses", "Ligand Pose")

    for pred, refs in zip(y_pred, y_true):
        # Check the input
        mol_has_3D(pred)
        mol_has_3D(refs)

        if pred.GetNumHeavyAtoms() != refs.GetNumHeavyAtoms():
            raise ValueError("mismatched number of atoms")

        # Compute RMSD
        pred_spy = spyrmsd.molecule.Molecule.from_rdkit(pred)
        refs_spy = spyrmsd.molecule.Molecule.from_rdkit(refs)

        rmsd = spyrmsd.rmsd.rmsdwrapper(refs_spy, pred_spy, symmetry=True, strip=True)
        rmsds.extend(rmsd)

    rmsds = np.asarray(rmsds)

    # The test set included some duplicate inputs.
    # In other words: For certain protein, ligand pairs, there were multiple possible poses.
    # For those duplicates, we keep and evaluate the best pose with the minimal RMSD.
    inputs["RMSD"] = rmsds
    grouped = (
        inputs.groupby(
            ["CXSMILES", "Chain A Sequence", "Chain B Sequence", "Protein Label"]
        )["RMSD"]
        .min()
        .reset_index()
    )
    rmsds = grouped["RMSD"].values

    # Evaluate with bootstrapping
    scores = pd.DataFrame(
        columns=["Target Label", "Metric", "Score", "Bootstrap Iteration"]
    )

    for i, ind in enumerate(bootstrapping_sampler(rmsds.shape[0], n_bootstrap_samples)):
        bootstrapped_rmsds = rmsds[ind]
        correct = bootstrapped_rmsds <= cutoff
        n_correct = sum(correct)
        ratio = (n_correct / len(bootstrapped_rmsds)) * 100

        scores.loc[len(scores)] = ["Ligand Pose", "success_rate", ratio, i]
        scores.loc[len(scores)] = [
            "Ligand Pose",
            "mean_rmsd",
            np.mean(bootstrapped_rmsds),
            i,
        ]

    scores["Test Set"] = "test"
    scores["Method"] = method_label

    return scores


def evaluate_all_ligand_pose_predictions(
    y_true: list[Chem.Mol], all_y_pred: dict[str, list[Chem.Mol]]
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

    inputs = pd.read_csv("evaluation/data/ligand_pose_test_inputs.csv", index_col=0)

    all_scores = pd.DataFrame()

    for method_label, y_pred in all_y_pred.items():
        scores = evaluate_ligand_poses_predictions(y_true, y_pred, method_label, inputs)
        all_scores = pd.concat([all_scores, scores], ignore_index=True)

    leaderboards = scores_to_leaderboards(
        all_scores, rank_by="success_rate", ascending=False
    )

    main_leaderboard = add_cld_to_leaderboard(
        leaderboards["Ligand Pose"],
        all_scores,
        "success_rate",
        "Ligand Pose",
    )

    return main_leaderboard, leaderboards
