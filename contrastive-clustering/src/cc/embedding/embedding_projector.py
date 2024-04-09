# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Embedding projector using TensorBoard
"""

import fire
import os
from pathlib import Path
from tensorboard import program
import torch
from torch.utils.tensorboard import SummaryWriter
from wasabi import msg
import webbrowser

from ..utils.data_ioer import DataIOer
from ..utils import path_utils
from ..embedding.embedding_helper import EmbeddingHelper

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class EmbeddingProjector:
    def __init__(self):
        self.dl = DataIOer()
        self.eh = EmbeddingHelper()

    def add_run(self, emb_path: str | Path, tb_dir: str | Path = "tensorboard_runs"):
        """
        Add Tensorboard run
        """
        Path(tb_dir).mkdir(parents=True, exist_ok=True)

        # set run directory
        dataset_name = path_utils.get_original_name_from_embedding_path(emb_path)
        run_dir = Path(tb_dir) / dataset_name

        # set tensor name
        emb_checkpoint_name = path_utils.get_emb_checkpoint_name_from_path(emb_path)
        aug_method = path_utils.get_aug_method_from_emb_path(emb_path)
        tensor_name = f"{aug_method}_{emb_checkpoint_name}"
        tensor_dir = run_dir / tensor_name

        if not tensor_dir.is_dir():
            # load embeddings
            emb = self.eh.load_embedding(emb_path)
            # load cluster data
            cluster_path = path_utils.convert_to_clustered_path_from_embedding(emb_path)
            data = self.dl.load(cluster_path)
            titles = data["title"].tolist() if "title" in data.columns else data["text"].tolist()

            clusters = [data["cluster_idx"].tolist()]
            names = ["cluster_idx"]
            if "gold_cluster_name" in data.columns:
                clusters += [data["gold_cluster_name"].tolist()]
                names += ["gold_cluster_name"]

            # write tensorboard data
            self.write_data(run_dir, emb, titles)
            self.update_color(run_dir, clusters, names)
            self.update_tensor_name(run_dir, tensor_name)
            msg.good(f"Successfully added run to Tensorboard at {run_dir}")
        else:
            msg.info(f"Run directory at {run_dir} already existed, using cache")

    def write_data(self, run_dir: str | Path, embeddings: torch.Tensor, titles: list[str]):
        """
        Write tensorboard event to run_dir/
        """
        config_path = Path(run_dir) / "projector_config.pbtxt"
        if config_path.is_file():
            config = self.dl.load(config_path)
        else:
            config = ""

        writer = SummaryWriter(run_dir)
        writer.add_embedding(embeddings, metadata=titles)
        writer.flush()  # make sure everything is written to disk
        writer.close()

        config += "\n" + self.dl.load(config_path)
        self.dl.save(config, config_path)

    def update_tensor_name(self, run_dir: str | Path, tensor_name: str):
        # update directory name
        metadata_path = Path(run_dir) / "00000"
        metadata_path.rename(Path(run_dir) / tensor_name)
        # update name in config
        config_path = Path(run_dir) / "projector_config.pbtxt"
        config = self.dl.load(config_path)
        config = config.replace("default:00000", tensor_name)
        config = config.replace("00000", tensor_name)
        self.dl.save(config, config_path)

    def update_color(self, run_dir: str | Path, colors: list[list], names: list[str]):
        """
        Add color to Tensorboard event metadata in run_dir/
        """
        metadata_path = Path(run_dir) / "00000" / "default" / "metadata.tsv"
        metadata = self.dl.load(metadata_path, header=None)
        metadata.columns = ["title"]

        for name, color in zip(names, colors):
            metadata[name] = color
        # Save metadata
        self.dl.save(metadata, metadata_path)

    @staticmethod
    def launch(tb_dir: str | Path = "tensorboard_runs"):
        """
        Start tensorboard
        """
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", str(tb_dir), "--load_fast=false"])
        # Start server
        url = tb.launch()
        # Open web browser
        webbrowser.open(f"{url}?darkMode=true#projector", new=0, autoraise=True)
        msg.good(f"Tensorboard served at {url}")
        input() # o.w., just close after launching


if __name__ == "__main__":
    fire.Fire(EmbeddingProjector)
