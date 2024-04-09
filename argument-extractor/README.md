# argument-extractor

This tool allows you to extract pro/against arguments from a German news dataset using OpenAI's API (ChatGPT, GPT-4). It is a part of [Journalistic Portfolio Analysis](https://mtc.ethz.ch/research/natural-language-processing/journalistic-portfolio-analysis.html) project at Media Technology Center, ETH Zurich.

**This project requires OpenAI API credentials. Please obtain it for your organization.**

## Installation Manual

This guide assumes that you use Unix operating systems or the likes (such as WSL - Windows Subsystem for Linux).

### Python Virtual Environment
You can set up this repository using Python Virtual Environment (venv) using `requirements.txt` file.

```bash
cd argument-extractor
python3 -m venv arg-extractor # set up venv
source arg-extractor/bin/activate # activate venv
pip3 install -r requirements.txt # install packages
```

## Test run

To see how to run the code, use our provided test data `data/test.jsonl`. Run the following command:

1. complete `.env` file with your OpenAI API credential.
2. run the script to extract the arguments on your dataset

```bash
source arg-extractor/bin/activate # activate venv
python3 -m scripts.sentence_extractor extract_boolean_questions_from_file data/test.jsonl # extract sentences and questions
python3 -m scripts.sentence_extractor extract_relevant_sentences_from_file data/internal/test-questions.jsonl data/internal/test-sentences.jsonl # extract relevant (question, sentence) pairs
python3 -m scripts.sentence_extractor online_extract_arguments_from_file data/internal/test-pairs-relevant.jsonl # extract valid arguments
```

The results are saved to the parent folder of your data `data/internal/test-pairs-relevant-Argument.jsonl` file.

### Adapt to your dataset

To utilize the codebase with your dataset, prepare your dataset in `.jsonl` format. At minimum, it must contain the following columns:
	- `article_idx` - article ID
	- `topic_idx` - topic ID of the article
	- `text` - article text

## Adapt to other languages

To adapt to other languages, please update the pre-trained models used in functions `set_*`. Strictly speaking, POS tagger in function `set_pos_tagger` has to be updated; other pre-trained models already support multiple languages by default.

## Background

This project extracts arguments presented in news articles using language models.

The project involves 4 steps:
1. Sentence extraction
2. Debatable question extraction
3. Relevant (Question, Sentence) pairs ranking
4. Pairs labeling

Each step involves offline processing using pre-trained language models and online processing using OpenAI's ChatGPT and GPT-4.

- Folder `prompts/` contains the template for ChatGPT prompts, to be used in `scripts/chatgpt.py`.
- Folder `scripts/` contains the extractor codebase.
