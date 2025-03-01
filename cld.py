from string import ascii_lowercase

import pandas as pd
import tqdm


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


def cld(comparisons: pd.DataFrame):
    """
    Compact Letter Display

    Compute the compact letter display using the insert-absorb algorithm. 
    
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
    alphabet = ascii_lowercase
    for ci, col in enumerate(solution):
        letter = alphabet[ci]
        for idx, has_letter in enumerate(col):
            if has_letter:
                letters[idx] += letter

    return {group: sorted(letter) for group, letter in zip(unique_groups, letters)}
