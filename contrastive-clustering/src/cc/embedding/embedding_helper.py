# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Embedding helper, with functionalities like
- get embedding file path
- load embedding
- load single-vector embedding by document ID
- create embedding

Improvements:
- load_embedding_by_id: don't need to load everything (using LMDB?)
- automatic batch size calculation
"""

import fire
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from wasabi import msg

from ..utils.data_ioer import DataIOer
from ..utils import path_utils

SEED = 10
torch.cuda.empty_cache()
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)


class EmbeddingHelper:
    def __init__(self, checkpoint_dir: str = None):
        # Load our modules
        self.dl = DataIOer()

        # Set attributes
        self.checkpoint_name = None
        self.checkpoint = None
        if checkpoint_dir is not None:
            self.load_model(checkpoint_dir)

    def load_model(self, checkpoint_dir: str):
        """
        Load model
        """
        if Path(checkpoint_dir).exists():
            self.load_model_from_local(checkpoint_dir)
        else:
            self.load_model_from_hf(checkpoint_dir)
        if self.checkpoint is not None:
            self.checkpoint_name = Path(checkpoint_dir).stem # assign name only if successfully loaded

    def load_model_from_hf(self, repo_id: str):
        """
        Load HuggingFace Hub's model checkpoint
        """
        namespace, repo_name = repo_id.split(repo_id)
        msg.info(f"eh | Loading from HuggingFace Hub...")
        try:
            self.checkpoint = SentenceTransformer(repo_id)
            msg.good(f"eh | Successfully loaded model. Repo ID: {repo_id}")
        except HFValidationError:
            msg.fail(f"eh | Failed to load model - invalid repo_id. Repo ID: {repo_id}")
        except RepositoryNotFoundError:
            msg.fail(f"eh | Failed to load model - repo not found. Repo ID: {repo_id}")
        except ValueError:
            msg.fail(f"eh | Failed to load model - local directory not found. Repo ID: {repo_id}")

    def load_model_from_local(self, checkpoint_dir: str):
        """
        Load local model checkpoint
        """
        msg.info(f"eh | Loading from local directory...")
        try:
            self.checkpoint = SentenceTransformer(checkpoint_dir)
            msg.good(f"eh | Successfully loaded model. Local directory: {checkpoint_dir}")
        except OSError:
            msg.fail(f"eh | Failed to load model. Local directory: {checkpoint_dir}")

    @staticmethod
    def find_span_between_two_strings(original_string: str, string_1: str, string_2: str) -> str:
        a = original_string.find(string_1)
        b = original_string[a + len(string_1):].find(string_2)
        if (a != -1) and (b != -1):
            return original_string[a+len(string_1):b]
        else:
            return original_string.split(string_2)[0]

    def load_embedding(self, emb_path: str | Path) -> torch.Tensor:
        """
        Load embedding from file, given dataset path
        """
        # Catch FileNotFoundError
        if not Path(emb_path).is_file():
            msg.warn(f"eh | Embedding file not found at {emb_path}, creating new embedding")
            dataset_path = path_utils.get_original_path_from_embedding(emb_path)
            self.create_embedding_from_file(dataset_path)
        # Load embedding
        emb = self.dl.load(emb_path)
        msg.good(f"eh | Successfully loaded {emb.shape[0]} entries of {emb.shape[1]}-dim embeddings from {emb_path}")
        return emb

    def load_embedding_by_id(self, dataset_path: str, idx: str) -> torch.Tensor:
        """
        Load embedding by document ID from file
        """
        # Load embedding
        emb = self.load_embedding(dataset_path)
        return emb[idx, :]

    def create_embedding(self, data: list[str], batch_size: int = 256) -> torch.Tensor | None:
        """
        Create embedding
        """
        msg.info(f"eh | Creating embedding on batch size {batch_size}...")
        try:
            torch.cuda.empty_cache()
            emb = self.checkpoint.encode(data, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            msg.good(f"eh | Successfully created {emb.shape[0]} entries of {emb.shape[1]}-dim embeddings")
            return emb
        except torch.cuda.OutOfMemoryError:
            msg.fail("eh | Failed due to CUDA OOM. Try reducing batch size.")
        except TypeError:
            msg.fail("eh | Failed due to Type. Is the data_preprocessing texts?")

    def create_embedding_from_file(self, augmented_path: str | Path, batch_size: int = 256):
        """
        Create embedding and save to file, given dataset
        """
        emb_path = path_utils.create_emb_path_from_augmented(augmented_path, self.checkpoint_name)
        data = self.dl.load(augmented_path)["text"]
        emb = self.create_embedding(data, batch_size)
        if emb is not None:
            self.dl.save(emb, emb_path)


if __name__ == "__main__":
    fire.Fire(EmbeddingHelper)
