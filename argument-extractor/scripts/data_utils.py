# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import fire
import pandas as pd
from pathlib import Path

from . import data_ioer


def add_index_to_file(file_path: str, idx_name: str, prefix: str, file_suffix: str = "idx", **kwargs) -> None:
    df = data_ioer.load(file_path)

    indexed_df = add_index(df, idx_name, prefix, **kwargs)

    target_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{file_suffix}.jsonl")
    data_ioer.save(indexed_df, target_path)


def add_index(df: pd.DataFrame, idx_name: str, prefix: str, **kwargs) -> pd.DataFrame:
    # Get index
    if not kwargs.get("group", False):  # both group=False and group=None
        df[idx_name] = range(df.shape[0])
    else:
        df[idx_name] = df.groupby(kwargs["group_name"]).cumcount().astype(int)
    # Post-process index (zfill)
    max_value = df[idx_name].max()
    df[idx_name] = df[idx_name].astype(str).str.zfill(len(str(max_value)))
    df[idx_name] = prefix + df[idx_name].astype(str)
    return df


def filter_file(anchor_file_path: str, filter_file_path: str, column_name: str, file_suffix: str = "filter") -> None:
    anchor_df = data_ioer.load(anchor_file_path)
    filter_df = data_ioer.load(filter_file_path)

    df = filter_by_column(anchor_df, filter_df, column_name)

    target_path = str(Path(filter_file_path).parent / f"{Path(filter_file_path).stem}-{file_suffix}.jsonl")
    data_ioer.save(df, target_path)


def filter_by_column(anchor_df: pd.DataFrame, filter_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df = anchor_df[anchor_df[column_name].isin(filter_df[column_name].to_list())]
    df.reset_index(drop=True, inplace=True)
    return df


def filter_by_column_pairs(anchor_df: pd.DataFrame, filter_df: pd.DataFrame, column_name1: str, column_name2: str) -> (
        pd.DataFrame):
    df = anchor_df[anchor_df[column_name1].isin(filter_df[column_name1].to_list()) &
                   anchor_df[column_name2].isin(filter_df[column_name2].to_list())]
    # using and results in ValueError The truth value of a Series is ambiguous.
    # Use a.empty, a.bool(), a.item(), a.any() or a.all().
    df.reset_index(drop=True, inplace=True)
    return df


def add_dummy_column_to_file(file_path: str, column_name: str, file_suffix: str = "dummy") -> None:
    df = data_ioer.load(file_path)
    modified_df = add_dummy_column(df, column_name)
    target_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{file_suffix}.jsonl")
    data_ioer.save(modified_df, target_path)


def add_dummy_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = 0
    return df


if __name__ == "__main__":
    fire.Fire()
