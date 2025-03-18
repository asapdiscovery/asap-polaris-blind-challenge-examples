from string import ascii_lowercase, ascii_uppercase
import string

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import tqdm


CLD_ALPHABET = list(ascii_lowercase) + list(ascii_uppercase)


def asserts_non_significance(col: list[bool], i: int, j: int):
    return col[i] and col[j]


def can_be_absorbed(new_col: list[bool], ref_col: list[bool]):
    return all(ref_col[i] for i, x in enumerate(new_col) if x)


def insert(column: list[bool], i: int, j: int):
    col_i = column.copy()
    col_j = column.copy()
    col_i[i] = False
    col_j[j] = False
    return col_i, col_j


def absorb(new_column: list[bool], columns: list[list[bool]]):
    if any(can_be_absorbed(new_column, c) for c in columns):
        return columns
    return columns + [new_column]


def cld(comparisons: pd.DataFrame) -> dict[str, str]:
    """
    Compact Letter Display

    Compute the compact letter display using the insert-absorb algorithm. 

    See the following papers for more information: 
    (1) https://doi.org/10.1016/j.csda.2006.09.035
    (2) https://doi.org/10.1198/1061860043515

    Parameters
    ----------
        comparisons : pd.DataFrame 
            A DataFrame containing the pairwise comparisons produced by:
            https://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html 
    """
    unique_groups = set(comparisons["group_t"].unique())
    unique_groups = unique_groups.union(set(comparisons["group_c"].unique()))
    unique_groups = list(unique_groups)
    unique_groups_indices = {g: i for i, g in enumerate(unique_groups)}

    sig_diff = comparisons[comparisons["reject"]]
    print(f"Found {len(sig_diff)} significantly different pairs")

    solution = [[True] * len(unique_groups)]

    for _, row in tqdm.tqdm(sig_diff.iterrows(), total=len(sig_diff)):
        i = unique_groups_indices[row["group_t"]]
        j = unique_groups_indices[row["group_c"]]

        has_changed: bool = True
        while has_changed:
            has_changed = False

            for idx in range(len(solution)):
                if asserts_non_significance(solution[idx], i, j):
                    # Duplicate the column
                    col_i, col_j = insert(solution[idx], i, j)

                    # Remove the old column
                    solution.pop(idx)

                    # Try absorb the column in an old column
                    # Simply add it to the solution otherwise
                    solution = absorb(col_i, solution)
                    solution = absorb(col_j, solution)

                    has_changed = True
                    break

    # Assign letters
    letters = [""] * len(unique_groups)

    for ci, col in enumerate(solution):
        letter = CLD_ALPHABET[ci]
        for idx, has_letter in enumerate(col):
            if has_letter:
                letters[idx] += letter

    return {group: sorted(letter) for group, letter in zip(unique_groups, letters)}



def add_cld_to_leaderboard(
    leaderboard: pd.DataFrame,
    scores: pd.DataFrame,
    metric_label: str,
    target_label: str,
):
    """
    Add the compact letter display to the leaderboard.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        The leaderboard DataFrame, including the aggregated scores.
    scores : pd.DataFrame
        The scores DataFrame, having all replicates of the bootstrapped scores
    metric_label : str
        The metric label.
    target_label : str
        The target label.
    """
    ordered_methods = leaderboard["Method"].values

    scores = scores[scores["Metric"] == metric_label]
    scores = scores[scores["Target Label"] == target_label]
    scores["Score"] = scores["Score"].astype(float)

    # We compared methods using bootstrapping and the Tukey HSD test, presenting results via Compact Letter Display (CLD). 
    # While acknowledging that bootstrapping likely underestimates variance, 
    # we are not aware of better sampling techniques that fit the challenge format.
    stats = pairwise_tukeyhsd(endog=scores["Score"], groups=scores["Method"])
    comparisons = stats.summary_frame()

    # Reassign CLD letters such that the alphabetical ordering 
    # aligns with the ordering of the methods in the leaderboard

    letter_mapping = {}
    letter_code = cld(comparisons)

    cld_column = [""] * len(leaderboard)
    for idx, method in enumerate(ordered_methods):
        letters = letter_code[method]
        for letter in letters:
            if letter not in letter_mapping:
                letter_mapping[letter] = CLD_ALPHABET[len(letter_mapping)]
            cld_column[idx] += letter_mapping[letter]

    leaderboard["CLD"] = cld_column
    leaderboard = leaderboard[["Method", "CLD"] + leaderboard.columns[1:-1].tolist()]

    return leaderboard
