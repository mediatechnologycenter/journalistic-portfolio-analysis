# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Pre-process dataframe
"""

import fire
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm.auto import tqdm
from wasabi import msg

from ..utils.data_ioer import DataIOer
from ..utils import path_utils

SEED = 10
DetectorFactory.seed = SEED
tqdm.pandas()


class DataPreprocessor:
    def __init__(self):
        # Load our modules
        self.dl = DataIOer()

    def process_data(self, data: pd.DataFrame, dataset_name: str, lang: str = "de") -> pd.DataFrame:
        # Process dataset
        ## INPLACE ##
        ### Rename content columns
        self.rename_content_columns(data, dataset_name)
        ### Drop missing rows
        self.drop_missing_rows(data, dataset_name)
        ### Fill missing rows in column lead from column text
        if ("text" in data.columns) and ("lead" in data.columns):
            data_to_fill = data["text"].str.split("\n").apply(lambda text_list: text_list[0])
            self.fill_missing_column_from_data(data, dataset_name, column_to_fill="lead", data_to_fill=data_to_fill)
        ### Drop index columns
        self.drop_index_columns(data, dataset_name, name_prefix="Unnamed")
        ## END OF INPLACE ##

        ## NOT INPLACE ##
        ### dataset-specific process
        self.process_dataset_specific(data, dataset_name, lang)
        ### Convert integer to datetime
        data = self.convert_to_datetime(data, dataset_name)
        ### Remove HTML elements
        data = self.remove_html_elements(data, dataset_name)
        ### Remove tabs in title
        data = self.convert_whitespaces_to_spaces(data, dataset_name)
        ### Remove non-target language articles
        data = self.remove_non_target_lang(data, dataset_name, lang=lang)
        ### Add doc_idx column
        data = self.add_index(data, dataset_name, name_prefix="doc")
        ## END OF NOT INPLACE ##
        return data

    def process_data_from_file(self, dataset_path: str | Path, lang: str = "de") -> pd.DataFrame:
        # load
        data = self.dl.load(dataset_path)
        dataset_name = path_utils.get_original_name_from_path(dataset_path)
        # process
        processed_data = self.process_data(data, dataset_name, lang)
        # save
        processed_path = path_utils.create_processed_path_from_original(dataset_path)
        self.dl.save(processed_data, processed_path)
        # save original .jsonl if not already
        if not str(dataset_path).endswith(".jsonl"):
            self.dl.save(data, Path(dataset_path).parent / f"{Path(dataset_path).stem}.jsonl")
        # return processed_data

    @staticmethod
    def process_dataset_specific(data: pd.DataFrame, dataset_name: str, lang: str):
        # TODO, in case anything
        return data

    @staticmethod
    def rename_content_columns(data: pd.DataFrame, dataset_name: str) -> None:
        """
        Make content columns (title, lead, text; ts cols) consistent
        """
        fn_dict = {
            "10kgnad-p2p": {"sentences": "text", "labels": "gold_cluster_name"},
            "10kgnad-s2s": {"sentences": "title", "labels": "gold_cluster_name"},
        }
        rename_map = fn_dict.get(dataset_name, dict())
        data.rename(rename_map, axis="columns", inplace=True)
        msg.good(f"dp | Successfully renamed columns: {rename_map}")

    @staticmethod
    def drop_missing_rows(data: pd.DataFrame, dataset_name: str) -> None:
        """
        Drop missing rows that don't have entries in title/lead/text
        """
        fn_dict = {
            "10kgnad-p2p": ["text"],
            "10kgnad-s2s": ["title"],
        }
        target_cols = fn_dict.get(dataset_name, list())
        data.dropna(subset=target_cols, how="any", inplace=True)
        data.reset_index(drop=True, inplace=True)
        msg.good(f"dp | Successfully dropped missing rows in columns: {target_cols}")

    @staticmethod
    def fill_missing_column_from_data(data: pd.DataFrame, dataset_name: str, column_to_fill: str,
                                      data_to_fill: list | pd.Series) -> None:
        """
        Fill missing rows in column lead from column text
        """
        data[column_to_fill] = None
        data[column_to_fill].fillna(data_to_fill, axis="index", inplace=True)
        msg.good(f"dp | Successfully filled column: {column_to_fill}")

    @staticmethod
    def drop_index_columns(data: pd.DataFrame, dataset_name: str, name_prefix: str) -> None:
        """
        Drop index columns
        """
        fn_dict = {
        }
        target_cols = fn_dict.get(dataset_name, list())
        data.drop(axis="columns", columns=target_cols, inplace=True, errors="ignore")
        msg.good(f"dp | Successfully dropped index columns: {target_cols}")

    @staticmethod
    def convert_to_datetime(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Convert integer / string (2023-01-01) to datetime
        """
        msg.info(f"dp | Converting integer to datetime...")
        fn_dict = {
        }
        target_cols = fn_dict.get(dataset_name, list())
        for col in target_cols:
            try:
                data[col] = pd.to_datetime(data[col], utc=True, unit="s") # ts
            except ValueError:
                data[col] = pd.to_datetime(data[col], utc=True) # date
        msg.good(f"dp | Successfully converted int timestamp rows to datetime: {target_cols}")
        return data

    @staticmethod
    def remove_html_elements(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Remove HTML elements such as <...> and keep only plain text
        """
        fn_dict = {
        }
        target_cols = fn_dict.get(dataset_name, list())
        for col in target_cols:
            data[col] = data[col].progress_apply(lambda html: BeautifulSoup(html, "lxml").get_text("\n"))
        msg.good(f"dp | Successfully converted HTML rows into plain text in columns: {target_cols}")
        return data

    @staticmethod
    def convert_whitespaces_to_spaces(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Remove whitespaces from rows in target columns (title)
        (to prevent .tsv confusion in TensorBoard)
        """
        fn_dict = {
            "10kgnad-p2p": ["text"],
            "10kgnad-s2s": ["title"],
        }
        target_cols = fn_dict.get(dataset_name, dict())

        for col in target_cols:
            data[col] = data[col].progress_apply(
                lambda text: " ".join(text.split()))
        msg.good(f"dp | Successfully removed whitespaces: {target_cols}")
        return data

    @staticmethod
    def remove_non_target_lang(data: pd.DataFrame, dataset_name: str, lang: str) -> pd.DataFrame:
        """
        Remove rows with non-target language based on target columns (lead) -- cuz not too short, not too long
        """

        def detect_lang(text: str) -> str | None:
            try:
                return detect(text)
            except LangDetectException:
                msg.warn(f"Unable to detect language of text: {text}, returning None")
                return None

        fn_dict = {
            "10kgnad-p2p": ["text"],
            "10kgnad-s2s": ["title"],
        }
        target_cols = fn_dict.get(dataset_name, list())
        for col in target_cols:
            data = data[data[col].progress_apply(lambda text: detect_lang(text)) == lang]
        data.reset_index(drop=True, inplace=True)
        msg.good(f"dp | Successfully removed non-target language rows: {target_cols}, {lang}")
        return data

    @staticmethod
    def add_index(data: pd.DataFrame, dataset_name: str, name_prefix: str, col_position: int = 0) -> pd.DataFrame:
        """
        Add index column of specified name prefix at specified position
        """
        data.reset_index(drop=True, inplace=True)
        # insert col
        col = f"{name_prefix}_idx"
        data.insert(col_position, col, pd.Series(range(data.shape[0])))
        num_digit = len(str(data.shape[0]))
        # create id-00 ~ id-99 instead of id-0 ~ id-99
        data[col] = data[col].astype(str).str.zfill(num_digit)
        data[col] = data[col].apply(lambda idx: f"id-{dataset_name}-{idx}")
        msg.good(f"dp | Successfully added index col: {col}")
        return data


if __name__ == "__main__":
    fire.Fire(DataPreprocessor)
