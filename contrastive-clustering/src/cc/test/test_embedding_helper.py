# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import unittest
from pathlib import Path
import shutil

import torch
from sentence_transformers import SentenceTransformer

from ..embedding.embedding_helper import EmbeddingHelper


class MyTestCase(unittest.TestCase):
    def test_load_model(self):
        # valid repo_id
        eh = EmbeddingHelper("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(eh.checkpoint, SentenceTransformer)
        self.assertEqual(eh.checkpoint_name, "all-MiniLM-L6-v2", "wrong checkpoint name")

        # valid repo_name
        eh = EmbeddingHelper("all-MiniLM-L6-v2")
        self.assertIsInstance(eh.checkpoint, SentenceTransformer)
        self.assertEqual(eh.checkpoint_name, "all-MiniLM-L6-v2", "wrong checkpoint name")

        cache_dir = ".cache/valid-ckpt/"
        eh.checkpoint.save(cache_dir)

        # valid cache_dir
        eh = EmbeddingHelper(".cache/valid-ckpt/")
        self.assertIsInstance(eh.checkpoint, SentenceTransformer)
        self.assertEqual(eh.checkpoint_name, "valid-ckpt", "wrong checkpoint name")

        shutil.rmtree(cache_dir)

        # invalid repo_id
        eh = EmbeddingHelper("sentence-transformers/repo_name")
        self.assertIsNone(eh.checkpoint, "checkpoint should be None")
        self.assertIsNone(eh.checkpoint_name, "checkpoint name should be None")

        # invalid repo_name
        eh = EmbeddingHelper("repo_name")
        self.assertIsNone(eh.checkpoint, "checkpoint should be None")
        self.assertIsNone(eh.checkpoint_name, "checkpoint name should be None")

        # invalid cache_dir - not existed
        eh = EmbeddingHelper(".cache/invalid-ckpt/")
        self.assertIsNone(eh.checkpoint, "checkpoint should be None")
        self.assertIsNone(eh.checkpoint_name, "checkpoint name should be None")

        # invalid cache_dir - existed but invalid
        eh = EmbeddingHelper("/home/")
        self.assertIsNone(eh.checkpoint, "checkpoint should be None")
        self.assertIsNone(eh.checkpoint_name, "checkpoint name should be None")

    def test_get_embedding_path(self):
        # valid repo_id
        eh = EmbeddingHelper("sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(eh.get_embedding_path("data_preprocessing.jsonl"), f"embeddings/emb_data_all-MiniLM-L6-v2.pt")
        self.assertEqual(eh.get_embedding_path("dir/data_preprocessing.jsonl"), f"dir/embeddings/emb_data_all-MiniLM-L6-v2.pt")
        self.assertEqual(eh.get_dataset_path("embeddings/emb_data_ckpt.pt"), f"data_preprocessing.jsonl")
        self.assertEqual(eh.get_dataset_path("dir/embeddings/emb_data_ckpt.pt"), f"dir/data_preprocessing.jsonl")

        # valid repo_name
        eh = EmbeddingHelper("all-MiniLM-L6-v2")
        self.assertEqual(eh.get_embedding_path("data_preprocessing.jsonl"), f"embeddings/emb_data_all-MiniLM-L6-v2.pt")
        self.assertEqual(eh.get_embedding_path("dir/data_preprocessing.jsonl"), f"dir/embeddings/emb_data_all-MiniLM-L6-v2.pt")
        self.assertEqual(eh.get_dataset_path("embeddings/emb_data_ckpt.pt"), f"data_preprocessing.jsonl")
        self.assertEqual(eh.get_dataset_path("dir/embeddings/emb_data_ckpt.pt"), f"dir/data_preprocessing.jsonl")

        cache_dir = ".cache/valid-ckpt/"
        eh.checkpoint.save(cache_dir)

        # valid cache_dir
        eh = EmbeddingHelper(".cache/valid-ckpt/")
        self.assertEqual(eh.get_embedding_path("data_preprocessing.jsonl"), f"embeddings/emb_data_valid-ckpt.pt")
        self.assertEqual(eh.get_embedding_path("dir/data_preprocessing.jsonl"), f"dir/embeddings/emb_data_valid-ckpt.pt")
        self.assertEqual(eh.get_dataset_path("embeddings/emb_data_valid-ckpt.pt"), f"data_preprocessing.jsonl")
        self.assertEqual(eh.get_dataset_path("dir/embeddings/emb_data_valid-ckpt.pt"), f"dir/data_preprocessing.jsonl")

        shutil.rmtree(cache_dir)

    def test_create_embedding(self):
        # valid repo_id
        eh = EmbeddingHelper("sentence-transformers/all-MiniLM-L6-v2")
        emb = eh.create_embedding(["test", "test " * 2 ** 9, "test " * 2 ** 16])
        self.assertIsInstance(emb, torch.Tensor)
        self.assertEqual(emb.shape[0], 3)
        self.assertFalse(torch.equal(emb[0, :], emb[1, :]))
        self.assertTrue(torch.equal(emb[1, :], emb[2, :]))  # same embedding due to truncation

        # valid repo_name
        eh = EmbeddingHelper("all-MiniLM-L6-v2")
        emb = eh.create_embedding(["test", "test " * 2 ** 9, "test " * 2 ** 16])
        self.assertIsInstance(emb, torch.Tensor)
        self.assertEqual(emb.shape[0], 3)
        self.assertFalse(torch.equal(emb[0, :], emb[1, :]))
        self.assertTrue(torch.equal(emb[1, :], emb[2, :]))  # same embedding due to truncation

        cache_dir = ".cache/valid-ckpt/"
        eh.checkpoint.save(cache_dir)

        # valid cache_dir
        eh = EmbeddingHelper(".cache/valid-ckpt/")
        emb = eh.create_embedding(["test", "test " * 2 ** 9, "test " * 2 ** 16])
        self.assertIsInstance(emb, torch.Tensor)
        self.assertEqual(emb.shape[0], 3)
        self.assertFalse(torch.equal(emb[0, :], emb[1, :]))
        self.assertTrue(torch.equal(emb[1, :], emb[2, :]))  # same embedding due to truncation


if __name__ == '__main__':
    unittest.main()
