# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import fire
import pandas as pd

from ..chatgpt import ChatGPT


class ArgumentLabeler(ChatGPT):
    def create_field_maps(self, data: pd.DataFrame, token_threshold: int) -> list[dict[str, str]]:
        data.reset_index(drop=True, inplace=True)
        data["index"] = data.apply(lambda row: str(row.name + 1), axis="columns")

        samples = data.apply(lambda row:
                             f"Batch[{row['index']}]:\nQuestion: \"{row['question']}\"\nSentence: \"{row['sentence']}\""
                             , axis="columns")

        batches = self.batch_data(samples, token_threshold)

        return [{"batch": batch} for batch in batches]


if __name__ == "__main__":
    fire.Fire(ArgumentLabeler)
