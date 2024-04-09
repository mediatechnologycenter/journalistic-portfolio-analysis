# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Data augmentation helper, with functionalities like
- index augmentation parameter list (for naming file short)
- get augmented data_preprocessing file path
- load augmented data_preprocessing
- create augmented data_preprocessing
"""

import fire
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm.auto import tqdm
from typing import Callable
from wasabi import msg

from ..data_augmentation.augmentation_functions import fns
from ..utils.data_ioer import DataIOer
from ..utils import path_utils

SEED = 10
np.random.seed(SEED)
random.seed(SEED)
tqdm.pandas()


class AugmentationProcessor:
    def __init__(self):
        # Load our modules
        self.dl = DataIOer()

    def process_data(self, data: pd.DataFrame, fn_name: str, target_column: str = "text", lang: str = "de",
                     **kwargs) -> pd.DataFrame:
        """
        Process data_preprocessing
        """
        # Sanity check
        if target_column not in data.columns:
            msg.fail(f"ap | Failed to process due to column not in data_preprocessing. Column: {target_column}")

        fn = fns[fn_name]

        # Duplicate rows
        num_copy = kwargs.get("num_copy", 1)
        data = self.duplicate_rows(data, num_copy)
        # Create pairs
        data = self.create_pairs(data, fn, target_column, lang, **kwargs)
        # # Insert index
        # data_preprocessing = self.add_index(data_preprocessing, augmentation_method=fn_name, name_prefix="aug")

        return data[["doc_idx", "text", "sentence1", "sentence2"]]

    def process_data_from_file(self, processed_path: str, fn_name: str, target_column: str = "text", lang: str = "de",
                               **kwargs):
        # load
        data = self.dl.load(processed_path)
        # process
        augmented_data = self.process_data(data, fn_name, target_column, lang, **kwargs)
        # save
        augmented_path = path_utils.create_augmented_path_from_processed(processed_path, fn_name)
        self.dl.save(augmented_data, augmented_path)
        # save metadata
        metadata_path = Path(augmented_path).parent / f"{Path(augmented_path).stem}_metadata.json"
        self.dl.save(kwargs, metadata_path)
        # return augmented_data

    @staticmethod
    def duplicate_rows(data: pd.DataFrame, num_copy: int) -> pd.DataFrame:
        data = data.loc[data.index.repeat(num_copy)]
        data.reset_index(drop=True, inplace=True)
        msg.good(f"ap | Successfully duplicated to {num_copy} copies of rows")
        return data

    @staticmethod
    def create_pairs(data: pd.DataFrame, fn: Callable, target_column: str, lang: str, **kwargs) -> pd.DataFrame:
        data[["sentence1", "sentence2"]] = data.progress_apply(
            lambda row: fn(row[target_column], lang=lang, **kwargs), axis="columns", result_type="expand")
        msg.good(f"ap | Successfully created pairs from text augmentation")
        return data


if __name__ == "__main__":
    fire.Fire(AugmentationProcessor)
