# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import fire
import numpy as np
import pandas as pd
from pathlib import Path
import random
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import torch
from wasabi import msg

from src.cc.utils.data_ioer import DataIOer
from ..embedding.embedding_helper import EmbeddingHelper
from ..embedding.embedding_projector import EmbeddingProjector
from ..utils import path_utils

SEED = 10
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.empty_cache()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

models = {
    "agglo": AgglomerativeClustering
}


class ClusterTrainer:
    def __init__(self):
        # Load our modules
        self.dl = DataIOer()
        self.ep = EmbeddingProjector()
        self.eh = EmbeddingHelper()

    @staticmethod
    def train(model_name: str, embeddings: torch.Tensor, **kwargs) -> AgglomerativeClustering:
        model = models[model_name](compute_full_tree=True, compute_distances=True, **kwargs)  # TODO
        model.fit(embeddings)
        msg.good("ct | Successfully performed clustering")
        return model

    def train_from_file(self, model_name: str, emb_path: str | Path, **kwargs):
        # load embedding
        embeddings = self.dl.load(emb_path)
        # load dataset
        processed_path = path_utils.get_processed_path_from_embedding(emb_path)
        processed_data = self.dl.load(processed_path)
        # load gold label, if exists
        gold_cluster_labels = processed_data.get("gold_cluster_name", None)  # TODO name or label
        gold_cluster_labels = gold_cluster_labels.tolist() if gold_cluster_labels is not None else None

        # train model
        if gold_cluster_labels is not None:
            kwargs["n_clusters"] = len(set(gold_cluster_labels))
        model = self.train(model_name, embeddings, **kwargs)
        # save model
        clustered_path = path_utils.convert_to_clustered_path_from_embedding(emb_path)
        checkpoint_path = Path("checkpoints") / "sklearn_checkpoints" / f"{Path(clustered_path).stem}.skops"
        self.dl.save(model, checkpoint_path)

        # save data with cluster assignment
        cluster_labels = model.labels_.tolist()
        processed_data = self.add_cluster_data(processed_data, cluster_labels)
        self.dl.save(processed_data, clustered_path)
        # save cluster-to-text assignment
        try:
            cluster_to_text = processed_data.groupby("cluster_idx")["text"].apply(list)
        except KeyError:
            cluster_to_text = processed_data.groupby("cluster_idx")["title"].apply(list)
        cluster_to_text_path = Path(clustered_path).parent / f"{Path(clustered_path).stem}_cluster-to-text.json"
        self.dl.save(cluster_to_text, cluster_to_text_path)
        # save metric scores
        scores = self.compute_metrics(cluster_labels, embeddings, gold_cluster_labels)
        scores_path = Path(clustered_path).parent / f"{Path(clustered_path).stem}_scores.json"
        self.dl.save(scores, scores_path)
        # save metadata
        metadata_path = Path(clustered_path).parent / f"{Path(clustered_path).stem}_metadata.json"
        self.dl.save(kwargs, metadata_path)

        # add run to Tensorboard
        self.ep.add_run(emb_path)

    @staticmethod
    def add_cluster_data(data: pd.DataFrame, cluster_data: list[int]):
        # insert col
        col = "cluster_idx"
        data[col] = cluster_data
        num_digit = len(str(max(cluster_data)))
        # create id-00 ~ id-99 instead of id-0 ~ id-99
        data[col] = data[col].astype(str).str.zfill(num_digit)
        msg.good("ct | Successfully added cluster data_preprocessing")
        return data

    def compute_metrics(self, cluster_labels: list[int], embeddings: torch.Tensor,
                        gold_cluster_labels: list[int] = None) -> dict[str, float]:
        scores_without_gold = self.compute_metrics_without_gold_labels(cluster_labels, embeddings)
        scores_with_gold = self.compute_metrics_with_gold_labels(cluster_labels, embeddings, gold_cluster_labels)

        scores = scores_without_gold | scores_with_gold  # merge two dicts

        for k, v in scores.items():
            scores[k] = round(float(v), 3) if v is not None else None
        msg.good("ct | Successfully computed metrics")
        return scores

    @staticmethod
    def compute_metrics_without_gold_labels(cluster_labels: list[int], embeddings: torch.Tensor) -> dict[str, float]:
        scores = {
            "num_cluster": len(set(cluster_labels)),
            "sh_score": None,
            "ch_score": None,
            "db_score": None
        }

        # Silhouette score
        try:
            scores["sh_score"] = metrics.silhouette_score(embeddings, cluster_labels, metric="cosine",
                                                          random_state=SEED)  # TODO
        except:
            pass
        # Calinski-Harabasz score
        try:
            scores["ch_score"] = metrics.calinski_harabasz_score(embeddings, cluster_labels)
        except:
            pass
        # Davies-Bouldin score
        try:
            scores["db_score"] = metrics.davies_bouldin_score(embeddings, cluster_labels)
        except:
            pass

        for k, v in scores.items():
            scores[k] = round(float(v), 3) if v is not None else None
        msg.good("ct | Successfully computed metrics without gold labels")
        return scores

    @staticmethod
    def compute_metrics_with_gold_labels(cluster_labels: list[int], embeddings: torch.Tensor,
                                         gold_cluster_labels: list[int]) -> dict[str, float]:

        scores = {
            "v_score": None,
            "rand_score": None,
            "adj_rand_score": None,
            "mi_score": None,
            "adj_mi_score": None,
            "norm_mi_score": None,
            "hm_score": None,
            "completeness_score": None,
            "fm_score": None
        }

        if gold_cluster_labels is not None:

            try:
                scores["v_score"] = metrics.v_measure_score(gold_cluster_labels, cluster_labels, beta=1.0)
            except:
                pass

            try:
                scores["rand_score"] = metrics.rand_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["adj_rand_score"] = metrics.adjusted_rand_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["mi_score"] = metrics.mutual_info_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["adj_mi_score"] = metrics.adjusted_mutual_info_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["norm_mi_score"] = metrics.normalized_mutual_info_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["hm_score"] = metrics.homogeneity_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["completeness_score"] = metrics.completeness_score(gold_cluster_labels, cluster_labels)
            except:
                pass

            try:
                scores["fm_score"] = metrics.fowlkes_mallows_score(gold_cluster_labels, cluster_labels)
            except:
                pass

        for k, v in scores.items():
            scores[k] = round(float(v), 3) if v is not None else None
        msg.good("ct | Successfully computed metrics with gold labels")
        return scores

if __name__ == "__main__":
    fire.Fire(ClusterTrainer)