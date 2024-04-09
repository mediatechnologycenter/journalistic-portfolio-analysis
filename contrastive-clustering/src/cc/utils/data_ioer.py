# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Data IOer - load from and save to file path
"""

import torch
from bidict import bidict
from datasets import load_dataset
import fire
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
import json
from json.decoder import JSONDecodeError
import pandas as pd
from pathlib import Path
import skops.io as sio
from typing import Any, Hashable
from wasabi import msg

from ..utils.data_sampler import DataSampler

class DataIOer:
    def __init__(self):
        self.ds = DataSampler()

    def load(self, file_path: str | Path, **kwargs) -> Any:
        # Create directory
        pl_file_path = Path(file_path)

        # Load data_preprocessing
        try:
            if pl_file_path.name.endswith(".csv"):
                data = self.load_csv(file_path, **kwargs)
            if pl_file_path.name.endswith(".tsv"):
                data = self.load_csv(file_path, sep="\t", **kwargs)
            if pl_file_path.name.endswith(".parquet"):
                data = self.load_parquet(file_path, **kwargs)
            if pl_file_path.name.endswith(".json"):
                data = self.load_json(file_path, **kwargs)
            if pl_file_path.name.endswith(".jsonl"):
                data = self.load_jsonl(file_path, **kwargs)
            if pl_file_path.name.endswith(".txt") or pl_file_path.name.endswith(".pbtxt"):
                data = self.load_text(file_path, **kwargs)
            if pl_file_path.name.endswith(".pt"):
                data = self.load_tensor(file_path, **kwargs)
            if pl_file_path.name.endswith(".skops"):
                data = self.load_skops(file_path, **kwargs)
            msg.good(f"dataIOer | Successfully loaded the file.  Path: {file_path}")
            self.ds.sample(data)
            return data
        except FileNotFoundError:
            msg.fail(f"dataIOer | Target file not found! Path: {file_path}")

    @staticmethod
    def load_csv(file_path: str | Path, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, **kwargs)
            return df
        except pd.errors.ParserError:
            msg.fail(f"dataIOer | Target file exists but is not valid CSV! Path: {file_path}")

    @staticmethod
    def load_json(file_path: str | Path, **kwargs) -> dict | bidict:
        try:
            with open(file_path, "r") as f:
                ddict = json.load(f)
            return ddict
        except JSONDecodeError:
            msg.fail(f"dataIOer | Target file exists but is not valid JSON! Path: {file_path}")

    @staticmethod
    def load_jsonl(file_path: str | Path, **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_json(file_path, lines=True, **kwargs)
            return df
        except pd.errors.ParserError | ValueError:
            msg.fail(f"dataIOer | Target file exists but is not valid JSONL! Path: {file_path}")

    @staticmethod
    def load_parquet(file_path: str | Path, engine: str = "fastparquet", **kwargs) -> pd.DataFrame:
        try:
            df = pd.read_parquet(file_path, engine=engine, **kwargs)
            return df
        except pd.errors.ParserError:
            msg.fail(f"dataIOer | Target file exists but is not valid Parquet! Path: {file_path}")

    @staticmethod
    def load_text(file_path: str | Path, **kwargs) -> str:
        try:
            with open(file_path, "r") as f:
                text = f.read()  # : handle large OOM file
            return text
        except:
            msg.fail(f"dataIOer | Can't load the file! Path: {file_path}")  # 

    @staticmethod
    def load_tensor(file_path: str | Path, **kwargs) -> str:
        try:
            with torch.no_grad():
                tensor = torch.load(file_path, map_location=torch.device("cpu"))
            return tensor
        except:
            msg.fail(f"dataIOer | Can't load the file! Path: {file_path}")  # 

    @staticmethod
    def load_skops(file_path: str | Path, **kwargs) -> Any:
        try:
            obj = sio.load(file_path)
            return obj
        except:
            msg.fail(f"dataIOer | Can't load the file! Path: {file_path}")  # 

    @staticmethod
    def load_hf_dataset(repo_id: str, **kwargs) -> pd.DataFrame:
        try:
            df = load_dataset(repo_id).to_pandas(**kwargs)
            return df
        except HFValidationError:
            msg.fail(f"eh | Failed to load data_preprocessing - invalid repo_id. Repo ID: {repo_id}")
        except RepositoryNotFoundError:
            msg.fail(f"eh | Failed to load data_preprocessing - repo not found. Repo ID: {repo_id}")

    def save(self, data: Any, file_path: str | Path, **kwargs) -> None:
        # Create parent directory
        pl_file_path = Path(file_path)
        Path(pl_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save data_preprocessing
        if pl_file_path.name.endswith(".csv"):
            return self.save_csv(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".tsv"):
            return self.save_csv(data, file_path, sep="\t", **kwargs)
        if pl_file_path.name.endswith(".parquet"):
            return self.save_parquet(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".json"):
            return self.save_json(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".jsonl"):
            return self.save_jsonl(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".txt") or pl_file_path.name.endswith(".pbtxt"):
            return self.save_text(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".pt"):
            return self.save_tensor(data, file_path, **kwargs)
        if pl_file_path.name.endswith(".skops"):
            return self.save_skops(data, file_path, **kwargs)

        # Print success message
        self.ds.sample(data)
        msg.good(f"dataIOer | Successfully saved the file.  Path: {file_path}")

    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: str | Path, index: bool = False, **kwargs) -> None:
        try:
            df.to_csv(file_path, index=index, **kwargs)
        except AttributeError:
            msg.fail(f"dataIOer | Failed to save file. Is object a DataFrame?")

    @staticmethod
    def save_json(ddict: dict[str, Hashable] | bidict, file_path: str | Path, ensure_ascii: bool = False,
                  indent: int = 4, **kwargs) -> None:
        try:
            with open(file_path, "w") as f:
                json.dump(ddict, f, ensure_ascii=ensure_ascii, indent=indent, **kwargs)
        except TypeError:
            try:
                ddict.to_json(file_path, force_ascii=not ensure_ascii, lines=False)
            except:
                msg.fail(f"dataIOer | Failed to save file. Is object a dict? Is the values JSON serializable?")

    @staticmethod
    def save_jsonl(df: pd.DataFrame, file_path: str | Path, force_ascii: bool = False, **kwargs) -> None:
        try:
            df = pd.DataFrame(df)
        except:
            msg.fail(f"dataIOer | Failed to coerce object to DataFrame. Is object a valid list of dict?")
        try:
            df.to_json(file_path, orient="records", force_ascii=force_ascii, lines=True,
                       **kwargs)  # don't put indent here, o.w. invalid JSONL
        except AttributeError:
            msg.fail(f"dataIOer | Failed to save file. Is object a DataFrame?")

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: str | Path, engine: str = "fastparquet", index: bool = False,
                     **kwargs) -> None:
        try:
            df.to_parquet(file_path, engine, index=index, **kwargs)
        except AttributeError:
            msg.fail(f"dataIOer | Failed to save file. Is object a DataFrame?")

    @staticmethod
    def save_text(text: str, file_path: str | Path, **kwargs) -> None:
        try:
            with open(file_path, "w") as f:
                f.write(text)
        except:
            msg.fail(f"dataIOer | Failed to save file. Is the object string?")  # 

    @staticmethod
    def save_tensor(tensor: torch.Tensor, file_path: str | Path, **kwargs) -> None:
        try:
            torch.save(tensor, file_path)
        except:
            msg.fail(f"dataIOer | Failed to save file. Is the object Tensor?")  # 

    @staticmethod
    def save_skops(obj: Any, file_path: str | Path, **kwargs) -> None:
        try:
            sio.dump(obj, file_path)
        except:
            msg.fail(f"dataIOer | Failed to save file. Is the object from sklearn?")  # 


if __name__ == "__main__":
    fire.Fire(DataIOer)
