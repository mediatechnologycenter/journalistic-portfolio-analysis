# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Data sampler - print sampled data_preprocessing to console
"""

from bidict import bidict
import fire
import pandas as pd
from typing import Any
from wasabi import msg, Printer

class DataSampler:
	def __init__(self):
		self.msg = Printer()

	def sample(self, data: Any, n: int = 2) -> None:
		if isinstance(data, pd.DataFrame):
			self.sample_df(data, n)
		if isinstance(data, dict | bidict):
			self.sample_dict(data, n)
		if isinstance(data, list):
			self.sample_list(data, n)

	def sample_df(self, df: pd.DataFrame, n: int) -> None:
		# get possible n_rows to print
		n = min(n, df.shape[0])

		self.msg.good(f"data_preprocessing-sampler | Loaded {df.shape[0]} rows")
		for i in range(n):
			self.msg.info(f"data_preprocessing-sampler | Display example: {i}-th row")
			print(df.iloc[i])

	def sample_dict(self, ddict: dict, n: int) -> None:
		# get possible n_rows to print
		n = min(n, len(ddict))

		self.msg.good(f"data_preprocessing-sampler | Loaded {len(ddict)} items")
		for i in range(n):
			self.msg.info(f"data_preprocessing-sampler | Display example: {i}-th item")
			k, v = list(ddict.items())[i]
			print(f"  Key:\t {k}\t (dtype: {type(k).__name__})")
			print(f"  Value:\t {v}\t (dtype: {type(v).__name__})")

	def sample_list(self, llist: list, n: int) -> None:
		# get possible n_rows to print
		n = min(n, len(llist))

		self.msg.good(f"data_preprocessing-sampler | Loaded {len(llist)} items")
		for i in range(n):
			self.msg.info(f"data_preprocessing-sampler | Display example: {i}-th item")
			item = llist[i]
			print(f"  Item:\t {item}\t (dtype: {type(item).__name__})")

	def test(self) -> None:
		self.sample({"a": [0, 1], "b": [1, 2], "c": [1, 2, 3]})
		self.sample([1, 2, 3, 4, 5])
		self.sample(pd.DataFrame(data={"x": [1, 2, 3], "y": ["I", "am", "a"]}))

if __name__== "__main__":
	fire.Fire(DataSampler)