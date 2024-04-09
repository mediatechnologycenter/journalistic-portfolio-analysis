# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Data IOer - load from + save to file path

Note:
    - don't use class as there is no data + fn encapsulation necessary
"""

import fire
import json
from json.decoder import JSONDecodeError
import pandas as pd
from pathlib import Path
from typing import Any, Hashable, Literal
from wasabi import msg

from . import data_sampler


def load(file_path: str, **kwargs) -> Any:
    # Create Pathlib file path
    path = Path(file_path)

    # placeholder TBD
    data = None

    # Load data
    try:
        if path.name.endswith(".csv"):
            data = load_csv(file_path, **kwargs)
        if path.name.endswith(".tsv"):
            data = load_csv(file_path, sep="\t", **kwargs)
        if path.name.endswith(".parquet"):
            data = load_parquet(file_path, **kwargs)
        if path.name.endswith(".json"):
            try:
                data = load_json_df(file_path, lines=False, **kwargs)
            except:
                data = load_json_dict(file_path)  # no kwargs
        if path.name.endswith(".jsonl"):
            data = load_json_df(file_path, lines=True, **kwargs)
        if path.name.endswith(".txt"):
            data = load_text(file_path)  # no kwargs
    except FileNotFoundError:
        msg.fail("data-loader | Target file not found!", f"Path: {file_path}", exits=True)
    except (pd.errors.ParserError, JSONDecodeError):
        msg.fail("data-loader | Target file exists but invalid!", f"Path: {file_path}", exits=True)

    # Print success message
    msg.good("data-loader | Successfully loaded the file.",  f"Path: {file_path}")
    data_sampler.sample(data)
    return data


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path, **kwargs)
    return df


def load_json_df(file_path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_json(file_path, **kwargs)
    return df


def load_json_dict(file_path: str) -> dict:
    with open(file_path, "r") as f:
        ddict = json.load(f)
    return ddict


def load_parquet(file_path: str, engine: str = "fastparquet", **kwargs) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine=engine, **kwargs)
    return df


def load_text(file_path: str) -> str:
    with open(file_path, "r") as f:
        text = f.read()  # TODO: handle OOM file
    return text


def save(data: Any, file_path: str, **kwargs) -> None:
    # Create parent directory
    path = Path(file_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save data
    try:
        if path.name.endswith(".csv"):
            save_csv(data, file_path, **kwargs)
        if path.name.endswith(".tsv"):
            save_csv(data, file_path, sep="\t", **kwargs)
        if path.name.endswith(".parquet"):
            save_parquet(data, file_path, **kwargs)
        if path.name.endswith(".json"):
            save_json_dict(data, file_path, **kwargs)
        if path.name.endswith(".jsonl"):
            save_json_df(data, file_path, lines=True, **kwargs)
        if path.name.endswith(".txt"):
            save_text(data, file_path)  # no kwargs
    except (AttributeError, TypeError):
        msg.warn("data-loader | File cannot be saved.", "Is object and/or value a valid type / serializable / ...?")
        data_sampler.sample(data)
        return None

    # Print success message
    msg.good("data-loader | Successfully saved the file.",  f"Path: {file_path}")


def save_csv(df: pd.DataFrame | list, file_path: str, index: bool = False, **kwargs) -> None:
    df = pd.DataFrame(df)
    df.to_csv(file_path, index=index, **kwargs)


def save_json_df(df: pd.DataFrame | list, file_path: str, force_ascii: bool = False, **kwargs) -> None:
    """
    Note: don't put indent here if lines=True; o.w. invalid JSONL
    """
    df = pd.DataFrame(df)
    df.to_json(file_path, orient="records", force_ascii=force_ascii, **kwargs)


def save_json_dict(ddict: dict[str, Hashable], file_path: str, ensure_ascii: bool = False, indent: int = 4, **kwargs) \
        -> None:
    with open(file_path, "w") as f:
        json.dump(ddict, f, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def save_parquet(df: pd.DataFrame | list, file_path: str, engine: Literal["pyarrow", "fastparquet"] = "fastparquet",
                 index: bool = False, **kwargs) -> None:
    df = pd.DataFrame(df)
    df.to_parquet(file_path, engine, index=index, **kwargs)


def save_text(text: str, file_path: str) -> None:
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    fire.Fire()
