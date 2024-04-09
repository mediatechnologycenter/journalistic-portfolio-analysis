# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import pandas as pd
from datasets import load_dataset

from src.cc.utils.data_ioer import DataIOer

text_col = {
    "p2p": "text"
}

for data_type in ["p2p"]:
    dsd = load_dataset(f"slvnwhrl/tenkgnad-clustering-{data_type}")["test"]

    ds_list = list()

    for split in range(len(dsd)):

        data = pd.DataFrame(dsd[split])
        data.rename({"sentences": text_col[data_type], "labels": "gold_cluster_name"}, axis="columns", inplace=True)
        data.drop_duplicates(inplace=True, ignore_index=True) # splits have duplicates
        data.reset_index(drop=True, inplace=True)
        DataIOer().save(data, f"data/10kgnad-{data_type}-{split}.jsonl")

        assert(len(data["gold_cluster_name"].unique()) == 9)