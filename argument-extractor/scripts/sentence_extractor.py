# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

import fire
from flair.nn import Classifier
from flair.data import Sentence  # spacy not good for German
import pandas as pd
from pathlib import Path
import re
from sentence_transformers import CrossEncoder
from string import digits, punctuation, whitespace
from tqdm import tqdm
from transformers import pipeline
from wasabi import msg
from wtpsplit import WtP  # better than PySBD

from . import data_ioer
from . import data_utils
from .labelers.question_labeler import QuestionLabeler
from .labelers.argument_labeler import ArgumentLabeler

tqdm.pandas()


class ArgumentExtractor:
    def __init__(self):
        # Placeholder TBD
        self.sentence_segmenter = None
        self.pos_tagger = None
        self.question_clf = None
        self.answer_retriever = None

        # Set attributes
        self.set_sentence_segmenter()
        self.set_pos_tagger()
        self.set_question_clf()
        self.set_answer_retriever()

    def set_sentence_segmenter(self):
        self.sentence_segmenter = WtP("wtp-canine-s-12l")
        self.sentence_segmenter.half().to("cuda")

    def set_pos_tagger(self, checkpoint_name: str = "de-pos"):
        self.pos_tagger = Classifier.load(checkpoint_name)
        msg.good("Successfully loaded NER tagger.", f"Checkpoint name: {checkpoint_name}")

    def set_question_clf(self, checkpoint_name: str = "PrimeQA/tydiqa-boolean-question-classifier"):
        self.question_clf = pipeline("text-classification", model=checkpoint_name, device=0)  # TODO
        msg.good("Successfully loaded question classifier.", f"Checkpoint name: {checkpoint_name}")

    def set_answer_retriever(self, checkpoint_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.answer_retriever = CrossEncoder(checkpoint_name)
        msg.good("Successfully loaded QA retriever.", f"Checkpoint name: {checkpoint_name}")

    def extract_sentences(self, text: str, lang: str = "de") -> list[str]:
        try:
            return self.sentence_segmenter.split(text, lang_code=lang)
        except AssertionError:
            return list()

    def extract_sentences_from_file(self, file_path: str, lang: str = "de") -> pd.DataFrame:
        df = data_ioer.load(file_path)

        df_sentence = df[["article_idx", "topic_idx"]].copy()
        # Extract sentences
        df_sentence["sentences"] = df["text"].progress_apply(lambda text: self.extract_sentences(text, lang))
        df_sentence = df_sentence[df_sentence["sentences"].apply(lambda s: type(s) is list)]
        df_sentence = df_sentence.explode("sentences", ignore_index=True)
        df_sentence.reset_index(drop=True, inplace=True)
        # Rename column
        df_sentence.rename(columns={"sentences": "sentence"}, inplace=True)
        # Clean: stripping whitespaces
        df_sentence["sentence"] = df_sentence["sentence"].str.strip()

        target_path = str(Path(file_path).parent / "internal" / f"{Path(file_path).stem}-sentences.jsonl")
        data_ioer.save(df_sentence, target_path)
        return df_sentence

    @staticmethod
    def manual_check_question(sentence: str) -> bool:
        # Check that the ending is question mark
        try:
            if not sentence.endswith("?"):
                return False
            return True
        except:
            return False

    def remove_leading_names(self, sent: str) -> str:
        """
        Remove leading names that are NN or NE nouns
        """
        sentence_copy_pos = Sentence(sent)
        self.pos_tagger.predict(sentence_copy_pos)
        sentence_copy_pos_tags = sentence_copy_pos.get_labels()

        leading_nouns = list()
        leading_punctuations = list()
        if sentence_copy_pos_tags != list():
            # Remove [>=0 NOUN Block] [>=1 PUNCTUATION block] like 'Mr Tom: ...'
            # First: find consecutive NN or NE nouns
            idx = 0
            for idx, token in enumerate(sentence_copy_pos_tags):
                if token.value in ["NN", "NE"]:
                    leading_nouns += [token.data_point.text]
                else:
                    break  # break once not noun

            # Second: once not noun, find consecutive punctuations
            for idx_2 in range(idx, len(sentence_copy_pos_tags)):
                token = sentence_copy_pos_tags[idx_2]
                if token.value.startswith("$"):
                    leading_punctuations += [token.data_point.text]
                else:
                    break  # break once not punctuation

        # Remove tokens when >=1 PUNCTUATION after NOUN block
        if len(leading_punctuations) > 0:
            for token in leading_nouns:
                sent = sent.replace(token, "", 1)
            for token in leading_punctuations:
                sent = sent.replace(token, "", 1)

        return sent

    @staticmethod
    def remove_leading_quote(sentence: str) -> str:
        """
        Only keep what comes after the quote « », e.g. '«Das ist falsch», sagte Anna' --> ', sagte Anna'

        Note: https://german.stackexchange.com/questions/117/what-is-the-correct-way-to-denote-a-quotation-in-german
        """
        sentence = re.sub("«[^»]+»", "", sentence)
        return sentence

    @staticmethod
    def remove_invalid_leading_characters(sentence: str) -> str:
        """
        Remove invalid leading characters (punctuations, digits, whitespaces)

        TODO: might lose, e.g. '2.5% of ...' --> 'of ...' ?
        """
        invalid_lead_chars = punctuation + digits + whitespace

        sentence = sentence.lstrip(invalid_lead_chars)
        return sentence

    def check_non_wh_question(self, question: str) -> bool:
        """
        Check that sentence is not Wh-Type question based on POS tags
        - STTS (Stuttgart-Tubingen Tagset)
        - p7 of https://www.phonetik.uni-muenchen.de/studium/skripten/P6_Sprachsynthese/3_synthese_pos.pdf
        """
        question_copy = Sentence(question)
        self.pos_tagger.predict(question_copy)
        pos_tags = question_copy.get_labels()

        question_tags = ["PWS", "PWAT", "PWAV"]

        try:
            # PWS/PWAT/PWAV like Wann
            if pos_tags[0].value in question_tags:
                return False
            # KON + PW* like Und wann
            if (pos_tags[0].value == "KON") and pos_tags[1].value in question_tags:
                return False
            # APPR + PW* like Vor welchem
            if (pos_tags[0].value == "APPR") and pos_tags[1].value in question_tags:
                return False
        except IndexError:
            pass

        return True

    def clf_check_boolean_question(self, question: str) -> bool:
        label = self.question_clf([question])[0]["label"]
        return label == "LABEL_1"

    def offline_extract_boolean_questions(self, df_sentence: pd.DataFrame) -> pd.DataFrame:
        # Keep sentence with question mark
        df_filtered = df_sentence[df_sentence["sentence"].progress_apply(self.manual_check_question)]

        # Rename column
        df_filtered.rename(columns={"sentence": "question"}, inplace=True)
        # Remove leading names, invalid characters (like "1."), and quotes
        df_filtered["question"] = df_filtered["question"].progress_apply(self.remove_invalid_leading_characters)
        df_filtered["question"] = df_filtered["question"].progress_apply(self.remove_leading_names)
        df_filtered["question"] = df_filtered["question"].progress_apply(self.remove_invalid_leading_characters)
        df_filtered["question"] = df_filtered["question"].progress_apply(self.remove_leading_quote)
        df_filtered["question"] = df_filtered["question"].progress_apply(self.remove_invalid_leading_characters)
        # Check that it is not Wh-type question
        df_filtered = df_filtered[df_filtered["question"].progress_apply(self.check_non_wh_question)]
        # Check that it is not boolean question based on model checkpoint
        df_filtered = df_filtered[df_filtered["question"].progress_apply(self.clf_check_boolean_question)]
        # Drop empty rows
        df_filtered = df_filtered[df_filtered["question"] != ""]
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered

    def offline_extract_boolean_questions_from_file(self, file_path: str) -> None:
        df_sentence = data_ioer.load(file_path)

        df_filtered = self.offline_extract_boolean_questions(df_sentence)

        target_file_path = str(Path(file_path).parent /
                               f"{Path(file_path).stem.replace('-sentences', '')}-questions-offline.jsonl")
        data_ioer.save(df_filtered, target_file_path)

    @staticmethod
    def online_extract_boolean_questions(df_question: pd.DataFrame) -> pd.DataFrame:
        labeler = QuestionLabeler()
        df_label = df_question
        for template_path in [
            "prompts/labelers/yes-no-question-0.txt",
            "prompts/labelers/impersonal-question-0.txt",
            "prompts/labelers/self-contained-question-0.txt",
        ]:
            list_of_completions = labeler.process(df_label, template_path)
            completions = sum(list_of_completions, [])
            df_label = pd.DataFrame(completions)
        return df_label

    @staticmethod
    def online_extract_boolean_questions_from_file(file_path: str) -> None:
        original_file_path = file_path

        labeler = QuestionLabeler()

        # Process
        label_to_keep = "YesNo"  # consistent to prompt >>
        labeler.process_file(file_path, "prompts/labelers/yes-no-question-0.txt", label_to_keep)
        file_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{label_to_keep}.jsonl")

        label_to_keep = "Impersonal"  # consistent to prompt >>
        labeler.process_file(file_path, "prompts/labelers/impersonal-question-0.txt", label_to_keep)
        file_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{label_to_keep}.jsonl")

        label_to_keep = "SelfContained"  # consistent to prompt >>
        labeler.process_file(file_path, "prompts/labelers/self-contained-question-0.txt", label_to_keep)
        file_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{label_to_keep}.jsonl")

        # Add index (article)
        file_suffix = "original"
        data_utils.filter_file(original_file_path, file_path, column_name="question", file_suffix=file_suffix)
        file_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{file_suffix}.jsonl")
        # Add index (question)
        data_utils.add_index_to_file(file_path, idx_name="question_idx", prefix="question-", file_suffix="idx")

    def extract_boolean_questions_from_file(self, file_path: str) -> None:
        original_file_path = file_path

        file_suffix = "sentences"  # TODO link to another fn naming
        self.extract_sentences_from_file(file_path)
        file_path = str(Path(original_file_path).parent / "internal" / f"{Path(original_file_path).stem}-{file_suffix}.jsonl")

        file_suffix = "questions-offline"  # TODO link to another fn naming
        self.offline_extract_boolean_questions_from_file(file_path)
        file_path = str(Path(original_file_path).parent / "internal" / f"{Path(original_file_path).stem}-{file_suffix}.jsonl")

        file_suffix = "questions-offline-YesNo-Impersonal-SelfContained-original-idx"  # TODO link to another fn naming
        self.online_extract_boolean_questions_from_file(file_path)
        file_path = str(Path(original_file_path).parent / "internal" / f"{Path(original_file_path).stem}-{file_suffix}.jsonl")
        print(file_path)

        data = data_ioer.load(file_path)
        file_suffix = "questions"
        file_path = str(Path(original_file_path).parent / "internal" / f"{Path(original_file_path).stem}-{file_suffix}.jsonl")
        data_ioer.save(data, file_path)

    def _get_relevance_scores(self, anchor_sentence: str, candidate_sentences: list[str]) -> list[float]:
        pairs = [(anchor_sentence, s) for s in candidate_sentences]
        relevance_scores = self.answer_retriever.predict(pairs, show_progress_bar=True)
        return relevance_scores.tolist()  # convert from np.ndarray

    def get_sentence_relevance(self, anchor_sentence: str, candidate_sentences: list[str], threshold: float = 0.0) -> (
            list[bool]):
        """
        Threshold is output logit score between -10 and +10
        Ref: https://github.com/UKPLab/sentence-transformers/issues/1128#issuecomment-902422486
        """
        scores = self._get_relevance_scores(anchor_sentence, candidate_sentences)
        return (pd.Series(scores) >= threshold).tolist()

    def get_relevance_scores(self, anchor_sentence: str, df_sentence: pd.DataFrame) -> pd.DataFrame:
        # Find cluster which anchor sentence belongs to
        anchor_topic_idx = df_sentence[df_sentence["sentence"] == anchor_sentence]["topic_idx"].values[0]

        df_filtered = df_sentence[["article_idx", "topic_idx", "sentence"]]
        # Find sentences from the same cluster
        df_filtered = df_filtered[df_filtered["topic_idx"] == anchor_topic_idx]
        # Keep sentence withOUT question mark
        df_filtered = df_filtered[~df_filtered["sentence"].progress_apply(self.manual_check_question)]
        # Add anchor sentence column (should come before score)
        df_filtered["anchor_sentence"] = anchor_sentence
        # Keep relevant sentences
        df_filtered["relevance_score"] = self._get_relevance_scores(anchor_sentence, df_filtered["sentence"].tolist())

        df_filtered.sort_values("relevance_score", ascending=False, inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered

    def extract_relevant_sentences_from_file(self, question_file_path: str, sentence_file_path: str,
                                             threshold: float = 0.0) -> None:
        # Load question and sentences
        questions = data_ioer.load(question_file_path)[["question_idx", "question"]].to_dict(orient="records")
        df_sentence = data_ioer.load(sentence_file_path)[["article_idx", "topic_idx", "sentence"]]
        # Remove duplicated sentences from the same cluster (sometimes from different articles)
        df_sentence.drop_duplicates(["topic_idx", "sentence"], inplace=True, ignore_index=True)

        list_of_df_filtered = list()

        for idx, item in enumerate(tqdm(questions)):
            question, question_idx = item["question"], item["question_idx"]
            msg.info(f"sentence-extractor | Extracting relevance sentences for question {idx} / {len(questions)}",
                     question)
            try:
                # Get relevant sentences
                df_filtered = self.get_relevance_scores(question, df_sentence)
                # Rename column
                df_filtered.rename(columns={"anchor_sentence": "question"}, inplace=True)
                # Add question ID
                df_filtered["question_idx"] = question_idx
                # Save
                target_path = str(Path(sentence_file_path).parent / "internal-st" /
                                  f"{Path(sentence_file_path).stem}-relevant-{idx}-of-{len(questions)}.jsonl")
                data_ioer.save(df_filtered, target_path)
                # Note: using + result in ValueError: Unable to coerce to Series, length must be 3: given 0
                list_of_df_filtered.append(df_filtered)
            except IndexError:  # TODO no matching cluster id
                pass

        df_relevant = pd.concat(list_of_df_filtered, ignore_index=True)
        df_relevant = df_relevant[df_relevant["relevance_score"] >= threshold]
        target_path = str(Path(sentence_file_path).parent /
                          f"{Path(sentence_file_path).stem.replace('-sentences', '-pairs')}-relevant.jsonl")
        data_ioer.save(df_relevant, target_path)

    @staticmethod
    def online_extract_arguments_from_file(pair_file_path: str) -> None:
        original_file_path = pair_file_path

        labeler = ArgumentLabeler()

        # Process
        prompt_name = "argument-0"  # consistent to prompt >>
        # label_to_keep = None  # consistent to prompt >>
        labeler.process_file(pair_file_path, f"prompts/labelers/{prompt_name}.txt")
        file_path = str(Path(original_file_path).parent / "internal-openai" /
                        f"{Path(original_file_path).stem}-{prompt_name}.jsonl")

        # Keep only some labels  # TODO
        df = data_ioer.load(file_path)
        df = df[df["label"] != "NoArgument"]
        df.rename(columns={"statement": "sentence"}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        file_path = str(Path(original_file_path).parent / f"{Path(original_file_path).stem}-Arguments.jsonl")
        data_ioer.save(df, file_path)

        # Add index (question, sentence)
        original_df = data_ioer.load(original_file_path)
        df = data_utils.filter_by_column_pairs(original_df, df, "question", "sentence")

        file_path = str(Path(original_file_path).parent /
                        f"{Path(original_file_path).stem.replace('-pairs-relevant', '-arguments')}.jsonl")
        data_ioer.save(df, file_path)


if __name__ == "__main__":
    fire.Fire(ArgumentExtractor)
