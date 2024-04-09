# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Data sampler - print sampled data to console

Note:
    - don't use class as there is no data + fn encapsulation necessary
"""

import fire
import pandas as pd
from typing import Any
from wasabi import msg


def sample(data: Any, n: int = 2) -> None:
    if isinstance(data, pd.DataFrame):
        sample_df(data, n)
    if isinstance(data, dict):
        sample_dict(data, n)
    if isinstance(data, list):
        sample_list(data, n)


def sample_df(df: pd.DataFrame, n: int) -> None:
    # get possible number of items to print
    n = min(n, df.shape[0])

    msg.info(f"data-sampler | Found {df.shape[0]} rows")
    for i in range(n):
        msg.info(f"data-sampler | Display example: {i}-th row")
        print(df.iloc[i])


def sample_list(llist: list, n: int) -> None:
    # get possible number of items to print
    n = min(n, len(llist))

    msg.info(f"data-sampler | Found {len(llist)} items")
    for i in range(n):
        item = llist[i]
        msg.info(f"data-sampler | Display example: {i}-th item")
        print(f"  Item:\t {item}\t (type: {type(item).__name__})")


def sample_dict(ddict: dict, n: int) -> None:
    # get possible number of items to print
    n = min(n, len(ddict))

    msg.info(f"data-sampler | Found {len(ddict)} items")
    for i in range(n):
        k, v = list(ddict.items())[i]
        msg.info(f"data-sampler | Display example: {i}-th item")
        print(f"  Key:\t\t {k}\t\t (type: {type(k).__name__})")
        print(f"  Value:\t {v}\t\t (type: {type(v).__name__})")


if __name__ == "__main__":
    fire.Fire()
