# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import fire
from pathlib import Path


# from/to original
def create_processed_path_from_original(dataset_path: str | Path) -> str:
    processed_path = Path(dataset_path).parent / "processed" / f"processed_{Path(dataset_path).stem}.jsonl"
    return str(processed_path)


def get_original_name_from_path(dataset_path: str | Path) -> str:
    dataset_name = Path(dataset_path).stem
    return dataset_name


def get_original_name_from_processed(processed_path: str | Path) -> str:
    dataset_path = get_original_path_from_processed(processed_path)
    dataset_name = get_original_name_from_path(dataset_path)
    return dataset_name


def get_original_path_from_processed(processed_path: str | Path) -> str:
    dataset_path = Path(processed_path).parent.parent / str(Path(processed_path).name).replace("processed_", "")
    return str(dataset_path)


# from/to augmented
def create_augmented_path_from_original(dataset_path: str | Path, aug_method: str) -> str:
    dataset_name = get_original_name_from_path(dataset_path)
    augmented_path = (Path(dataset_path).parent / "processed" / "augmented" /
                      f"augmented_{dataset_name}_{aug_method}.jsonl")
    return str(augmented_path)


def create_augmented_path_from_processed(processed_path: str | Path, aug_method: str) -> str:
    dataset_path = get_original_path_from_processed(processed_path)
    augmented_path = create_augmented_path_from_original(dataset_path, aug_method)
    return str(augmented_path)


def get_original_name_from_augmented_path(augmented_path: str | Path) -> str:
    dataset_path = get_original_path_from_augmented(augmented_path)
    dataset_name = get_original_name_from_path(dataset_path)
    return dataset_name


def get_aug_method_from_augmented_path(augmented_path: str | Path) -> str:
    aug_method = str(Path(augmented_path).stem).split("_")[-1]
    return str(aug_method)


def get_processed_path_from_augmented(augmented_path: str | Path) -> str:
    file_stem = str(Path(augmented_path).stem).replace("augmented_", "processed_")
    file_stem = "_".join(file_stem.split("_")[:-1])
    processed_path = Path(augmented_path).parent.parent / f"{file_stem}.jsonl"
    return str(processed_path)


def get_original_path_from_augmented(augmented_path: str | Path) -> str:
    processed_path = get_processed_path_from_augmented(augmented_path)
    dataset_path = get_original_path_from_processed(processed_path)
    return str(dataset_path)


# from/to embedding
def create_emb_path_from_original(dataset_path: str | Path, aug_method: str, emb_checkpoint_name: str) -> str:
    dataset_name = get_original_name_from_path(dataset_path)
    emb_path = (Path(dataset_path).parent / "processed" / "augmented" / "embeddings" /
                      f"emb_{dataset_name}_{aug_method}_{emb_checkpoint_name}.pt")
    return str(emb_path)


def create_emb_path_from_processed(processed_path: str | Path, aug_method: str, emb_checkpoint_name: str) -> str:
    dataset_path = get_original_path_from_processed(processed_path)
    emb_path = create_emb_path_from_original(dataset_path, aug_method, emb_checkpoint_name)
    return str(emb_path)


def create_emb_path_from_augmented(augmented_path: str | Path, emb_checkpoint_name: str) -> str:
    dataset_path = get_original_path_from_augmented(augmented_path)
    aug_method = get_aug_method_from_augmented_path(augmented_path)
    emb_path = create_emb_path_from_original(dataset_path, aug_method, emb_checkpoint_name)
    return str(emb_path)


def get_original_name_from_embedding_path(embedding_path: str | Path) -> str:
    dataset_path = get_original_path_from_embedding(embedding_path)
    dataset_name = get_original_name_from_path(dataset_path)
    return dataset_name


def get_emb_checkpoint_name_from_path(embedding_path: str | Path) -> str:
    emb_checkpoint_name = str(Path(embedding_path).stem).split("_")[-1]
    return str(emb_checkpoint_name)


def get_emb_checkpoint_name_from_clustered_path(clustered_path: str | Path) -> str:
    emb_checkpoint_name = str(Path(clustered_path).stem).split("_")[-1]
    return str(emb_checkpoint_name)


def get_aug_method_from_emb_path(embedding_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_embedding(embedding_path)
    aug_method = get_aug_method_from_augmented_path(augmented_path)
    return aug_method


def get_aug_method_from_clustered_path(clustered_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_clustered(clustered_path)
    aug_method = get_aug_method_from_augmented_path(augmented_path)
    return aug_method


def get_augmented_path_from_embedding(embedding_path: str | Path) -> str:
    file_stem = str(Path(embedding_path).stem).replace("emb_", "augmented_")
    file_stem = "_".join(file_stem.split("_")[:-1])
    augmented_path = Path(embedding_path).parent.parent / f"{file_stem}.jsonl"
    return str(augmented_path)


def get_processed_path_from_embedding(embedding_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_embedding(embedding_path)
    processed_path = get_processed_path_from_augmented(augmented_path)
    return str(processed_path)


def get_original_path_from_embedding(embedding_path: str | Path) -> str:
    processed_path = get_processed_path_from_embedding(embedding_path)
    dataset_path = get_original_path_from_processed(processed_path)
    return str(dataset_path)


# from/to clustered

def create_clustered_path_from_original(dataset_path: str | Path, aug_method: str, emb_checkpoint_name: str) -> str:
    dataset_name = get_original_name_from_path(dataset_path)
    clustered_path = (Path(dataset_path).parent / "processed" / "augmented" / "clustered" /
                      f"clustered_{dataset_name}_{aug_method}_{emb_checkpoint_name}.jsonl")
    return str(clustered_path)


def create_clustered_path_from_processed(processed_path: str | Path, aug_method: str, emb_checkpoint_name: str) -> str:
    dataset_path = get_original_path_from_processed(processed_path)
    clustered_path = create_clustered_path_from_original(dataset_path, aug_method, emb_checkpoint_name)
    return str(clustered_path)


def create_clustered_path_from_augmented(augmented_path: str | Path, emb_checkpoint_name: str) -> str:
    dataset_path = get_original_path_from_augmented(augmented_path)
    aug_method = get_aug_method_from_augmented_path(augmented_path)
    clustered_path = create_clustered_path_from_original(dataset_path, aug_method, emb_checkpoint_name)
    return str(clustered_path)


def get_original_name_from_clustered_path(clustered_path: str | Path) -> str:
    dataset_path = get_original_path_from_clustered(clustered_path)
    dataset_name = get_original_name_from_path(dataset_path)
    return dataset_name


def get_augmented_path_from_clustered(clustered_path: str | Path) -> str:
    file_stem = str(Path(clustered_path).stem).replace("clustered_", "augmented_")
    file_stem = "_".join(file_stem.split("_")[:-1])
    augmented_path = Path(clustered_path).parent.parent / f"{file_stem}.jsonl"
    return str(augmented_path)


def get_processed_path_from_clustered(clustered_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_clustered(clustered_path)
    processed_path = get_processed_path_from_augmented(augmented_path)
    return str(processed_path)


def get_original_path_from_clustered(clustered_path: str | Path) -> str:
    processed_path = get_augmented_path_from_clustered(clustered_path)
    dataset_path = get_original_path_from_processed(processed_path)
    return str(dataset_path)


def convert_to_clustered_path_from_embedding(embedding_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_embedding(embedding_path)
    emb_checkpoint_name = get_emb_checkpoint_name_from_path(embedding_path)
    clustered_path = create_clustered_path_from_augmented(augmented_path, emb_checkpoint_name)
    return str(clustered_path)


def convert_to_embedding_path_from_clustered(clustered_path: str | Path) -> str:
    augmented_path = get_augmented_path_from_clustered(clustered_path)
    emb_checkpoint_name = get_emb_checkpoint_name_from_clustered_path(clustered_path)
    embedding_path = create_emb_path_from_augmented(augmented_path, emb_checkpoint_name)
    return str(embedding_path)


if __name__ == "__main__":
    fire.Fire()
