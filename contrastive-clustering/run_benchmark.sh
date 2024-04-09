#!/bin/bash
# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details


python3 -m src.cc.data_preprocessing.download_10kgnad # download benchmark dataset

for data_split in {0..9}
do
  echo $data_split # pre-process data
  python3 -m src.cc.data_preprocessing.data_preprocessor process_data_from_file "$(pwd)/data/10kgnad-p2p-${data_split}.jsonl"
  for aug_fn in repeat data_split sample shuffle-and-data_split
  do
    echo $data_split $aug_fn # create contrastive data pairs
    python3 -m src.cc.data_augmentation.augmentation_processor process_data_from_file "$(pwd)/data/processed/processed_10kgnad-p2p-${data_split}.jsonl" $aug_fn
    for emb_ckpt in sentence-t5-base use-cmlm-multilingual paraphrase-multilingual-MiniLM-L12-v2 paraphrase-multilingual-mpnet-base-v2
    do
      echo $data_split $aug_fn $emb_ckpt # fine-tune embedding model checkpoint + cluster
      time python3 -m src.cc.embedding.embedding_trainer --checkpoint_name "${emb_ckpt}" train_from_file "$(pwd)/data/processed/augmented/augmented_10kgnad-p2p-${data_split}_${aug_fn}.jsonl" --batch_size=64 --num_epoch=4
      time python3 -m src.cc.clustering.cluster_trainer train_from_file agglo "$(pwd)/data/processed/augmented/embeddings/emb_10kgnad-p2p-${data_split}_${aug_fn}_${emb_ckpt}-e4b64.pt"
    done
  done
done