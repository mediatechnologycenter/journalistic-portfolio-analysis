# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Unit tests for data sampler
"""

import pandas as pd
import unittest

from .. import data_sampler


class TestDataSampler(unittest.TestCase):
    def test_df(self) -> None:
        data = pd.DataFrame(data={"x": [1, 2, 3], "y": ["I", "am", "a"]})
        print(f"Data: \n{data}")
        data_sampler.sample(data)
        input("Press any button to continue once verified the correctness")

    def test_list(self) -> None:
        data = [1, 2, 3, 4, 5]
        print(f"Data: \n{data}")
        data_sampler.sample(data)
        input("Press any button to continue once verified the correctness")

    def test_dict(self) -> None:
        data = {"a": [0, 1], "b": [1, 2], "c": [1, 2, 3]}
        print(f"Data: \n{data}")
        data_sampler.sample(data)
        input("Press any button to continue once verified the correctness")


if __name__ == "__main__":
    unittest.main()
