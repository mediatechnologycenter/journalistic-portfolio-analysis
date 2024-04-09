# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import pysbd
import random

SEED = 10
random.seed(SEED)


def split_text(text: str, lang: str, cut_frac: float = 0.5, **kwargs) -> tuple[str, str]:
    """
    Split text into half, sentence-wise
    """
    # split into sentences
    seg = pysbd.Segmenter(language=lang, clean=False)
    sents = seg.segment(text)

    # merge splits
    cut_idx = int(len(sents) * cut_frac)
    split_1 = [sent for sent in sents[:cut_idx]]
    split_2 = [sent for sent in sents[cut_idx:]]

    return "".join(split_1), "".join(split_2)


def shuffle_and_split_text(text: str, lang: str, cut_frac: float = 0.5, **kwargs) -> tuple[str, str]:
    """
    Shuffle and split text into half, sentence-wise
    """
    # split into sentences
    seg = pysbd.Segmenter(language=lang, clean=False)
    sents = seg.segment(text)

    # shuffle
    sents = random.sample(sents, len(sents))

    # merge splits
    cut_idx = int(len(sents) * cut_frac)
    split_1 = [sent for sent in sents[:cut_idx]]
    split_2 = [sent for sent in sents[cut_idx:]]

    return "".join(split_1), "".join(split_2)


def sample_text(text: str, lang: str, sample_frac: float = 0.7, **kwargs) -> tuple[str, str]:
    """
    Sample text, keeping same order
    """
    # split into sentences
    seg = pysbd.Segmenter(language=lang, clean=False)
    sents = seg.segment(text)

    # sample
    split_1 = sents
    num_sample = int(len(sents) * sample_frac)
    idx_samples = random.sample(range(len(sents)), num_sample)
    idx_samples.sort()
    split_2 = [sents[idx] for idx in idx_samples]

    return "".join(split_1), "".join(split_2)


def repeat_text(text: str, **kwargs) -> tuple[str, str]:
    """
    Repeat text, like in SimCSE
    """
    return text, text


fns = {
    "split": split_text,
    "shuffle-and-split": shuffle_and_split_text,
    "sample": sample_text,
    "repeat": repeat_text
}