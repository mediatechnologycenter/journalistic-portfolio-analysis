# contrastive-clustering

This tool allows you to perform a clustering on a text dataset using representations fine-tuned by contrastive learning. It is a part of [Journalistic Portfolio Analysis](https://mtc.ethz.ch/research/natural-language-processing/journalistic-portfolio-analysis.html) project at Media Technology Center, ETH Zurich.

## Installation Manual

This guide assumes that you use Unix operating systems or the likes (such as WSL - Windows Subsystem for Linux).

### Python Virtual Environment
You can set up this repository using Python Virtual Environment (venv) using `requirements.txt` file.

```bash
cd contrastive-clustering
python3 -m venv cluster # set up venv
source cluster/bin/activate # activate venv
pip3 install -r requirements.txt # install packages
```

## Quick start

Run the script to see the results on benchmark dataset: [TenKGNADClusteringP2P](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p). This dataset consists of news articles, each corresponding to one of the 9 labels.

```bash
source cluster/bin/activate # activate venv
python3 -m src.cc.data_preprocessing.download_10kgnad # download benchmark dataset
bash run_benchmark.sh # (1) pre-process data, (2) create contrastive data pairs, (3) fine-tune embedding checkpoint, and (4) cluster embeddings
```

For the details of each step, you can check the bash script itself.

The results are saved to `data/processed/augmented/*`.

## Adapt to your own dataset

To run the clustering, you have to do the following:

1. prepare your dataset in `.jsonl` format and has column `text` at least (see `data/10kgnad-p2p-test.jsonl` for example). 
2. update the `data_processor.py` script to include your dataset's metadata for pre-processing. The script is located in folder `src/cc/data_preprocessing`.
3. check `run_benchmark.sh` and update the dataset references to your dataset, as well as the desired number of epochs and batch size.

## Background

This project improves text representation using contrastive learning. This is achieved by fine-tuning the pre-trained sentence-embedding models using positive pairs (i.e. similar texts) via augmentation methods.

The project involves 4 steps:
1. Data preprocessing
2. Data augmentation
3. Fine-tuning embedding models
4. Clustering

Each step has the corresponding scripts:
- Data preprocessing - `data_preprocessing/*`
- Data augmentation - `data_augmentation/*`
- Fine-tuning embedding models - `embedding/*`
- Clustering - `clustering/*`
