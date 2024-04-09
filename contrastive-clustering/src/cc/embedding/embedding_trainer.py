# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Embedding model trainer
"""

from datasets import load_dataset
import fire
import numpy as np
import pandas as pd
from pathlib import Path
import random
from sentence_transformers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
import torch
from torch.utils.data import DataLoader
from wasabi import msg

from ..utils.data_ioer import DataIOer
from ..utils import path_utils
from ..embedding.embedding_helper import EmbeddingHelper

SEED = 10
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.empty_cache()
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)


class EmbeddingTrainer:
    def __init__(self, checkpoint_name: str):
        # Load our modules
        self.dl = DataIOer()
        self.eh = EmbeddingHelper(checkpoint_name)

        # Set attributes
        self.checkpoint_path = None

        self.loss_short_name_dict = {
            "mnrl": MultipleNegativesRankingLoss
        }

    @staticmethod
    def prepare_input_examples(data: pd.DataFrame) -> list[InputExample]:
        samples = list()
        # for loop to make sure no OOM
        try:
            if "similarity_score" not in data.columns:
                data["similarity_score"] = 0
            else:
                data["similarity_score"] /= 5.0
            for row in zip(data["sentence1"], data["sentence2"], data["similarity_score"]):  # col consistent to STS
                samples.append(InputExample(texts=[row[0], row[1]], label=row[2]))

            msg.good("eh | Successfully prepared input examples")
            return samples
        except KeyError:
            msg.fail("eh | Failed to prepare data_preprocessing - key error. Does the data_preprocessing have columns "
                     "sentence1 and 2? Is it df?")
        except:
            msg.fail("eh | Failed to prepare data_preprocessing. IMPROVEMENT")

    def prepare_dataloader(self, data: pd.DataFrame, batch_size: int, **kwargs) -> DataLoader:
        samples = self.prepare_input_examples(data)
        dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return dataloader

    def prepare_evaluator(self, data: pd.DataFrame, batch_size: int, **kwargs) -> SentenceEvaluator:
        samples = self.prepare_input_examples(data)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=batch_size,
                                                                     show_progress_bar=True, write_csv=True, **kwargs)
        return evaluator

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, loss: str = "mnrl", batch_size: int = 8,
              num_epoch: int = 2, warmup_frac: float = 0.1, **kwargs):
        train_dataloader = self.prepare_dataloader(train_data, batch_size)
        # Set up model parameters
        msg.info("eh | Setting up model parameters...")
        train_loss = self.loss_short_name_dict[loss](self.eh.checkpoint)
        num_step_per_epoch = len(train_dataloader) * num_epoch
        num_warmup_step = int(num_step_per_epoch * warmup_frac)

        if val_data is not None:
            val_evaluator = self.prepare_evaluator(val_data, batch_size)
            num_val_step = num_step_per_epoch
        else:
            val_evaluator = None
            num_val_step = 0

        self.eh.checkpoint_name = f"{self.eh.checkpoint_name}-e{num_epoch}b{batch_size}"
        self.checkpoint_path = Path("checkpoints") / "st_checkpoints" / self.eh.checkpoint_name
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        # Train model
        msg.info("eh | Training model...")
        torch.cuda.empty_cache()
        self.eh.checkpoint.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=val_evaluator,
            epochs=num_epoch,
            evaluation_steps=num_val_step,
            warmup_steps=num_warmup_step,
            output_path=str(self.checkpoint_path),
            save_best_model=True,
            show_progress_bar=True,
            checkpoint_path=str(self.checkpoint_path / "checkpoints"),
            checkpoint_save_steps=num_step_per_epoch
        )
        msg.good(f"et | Successfully saved trained model to {self.checkpoint_path}")

    def train_from_file(self, augmented_path: str | Path, sts_eval: bool = False, **kwargs):
        train_data = self.dl.load(augmented_path)
        if sts_eval:
            val_data = load_dataset("stsb_multi_mt", name="de", split="dev").to_pandas()
        else:
            val_data = None
        self.train(train_data, val_data, **kwargs)
        self.eh.create_embedding_from_file(augmented_path, 256)
        checkpoint_path = (str(Path(self.checkpoint_path).parent / self.eh.checkpoint_name) + "_" +
                           str(Path(augmented_path).stem).replace("augmented_", "").replace(".jsonl", ""))
        self.eh.checkpoint.save(checkpoint_path)
        self.checkpoint_path = None


if __name__ == "__main__":
    fire.Fire(EmbeddingTrainer)
