# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
ChatGPT base class - implement shared processing for any ChatGPT prompt
"""

import ast  # don't use json because sometimes single-quote
from dotenv import load_dotenv, find_dotenv
import fire
from openai import OpenAI, AuthenticationError
import pandas as pd
from pathlib import Path
from requests.exceptions import ConnectionError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from typing import Any, Optional
import tiktoken
from wasabi import msg

from .openai_config import ChatModelConfig
from . import data_ioer


class ChatGPT:
    def __init__(self):
        # Load our modules
        self.config = ChatModelConfig()

        # Placeholder TBD
        self.client = None
        self.tokenizer = None

        # Set attributes
        self.set_up_openai_api()

    def set_up_openai_api(self):
        if load_dotenv(find_dotenv()):
            self.client = OpenAI()
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.config.model)
            except ConnectionError:
                msg.fail("Connection error - are you connected to the Internet?")
        else:
            msg.fail("chatgpt | .env file not found.")

        try:
            self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            )
        except AuthenticationError:
            msg.fail("chatgpt | Authentication failed. Check API keys in .env file.")

    def _process_field_map(self, field_map: dict[str, str], template_path: str, **kwargs) -> list[dict]:
        """
        Process field map

        {"field_name": ...} when there is {field_name} field in the template
        """
        template = self.load_template(template_path)
        prompt = self.convert_template_to_prompt(template, field_map)
        response = self.completion_with_backoff(prompt, **kwargs)
        completion = self.parse_completion_as_list(response)
        return completion
    
    def process(self, data: pd.DataFrame, template_path: str) -> list[list[dict]]:
        """
        Note: has to be consistent with process_file fn. Implemented separately due to increment saving
        """
        # Create field maps
        field_maps = self.create_field_maps(data, int(self.config.TPM / self.config.RPM))  # TODO
        num_batch = len(field_maps)
        list_of_completions = list()

        # Iterate through each field map
        for idx, field_map in enumerate(field_maps):
            msg.info(f"chatgpt | Batch {idx + 1} / {num_batch}", f"Batch size: {len(field_map['batch'])}")
            # Retrieve parsed completion
            completion = self._process_field_map(field_map, template_path)
            list_of_completions.append(completion)

        return list_of_completions

    @staticmethod
    def load_template(template_path: str):
        return data_ioer.load(template_path)

    @staticmethod
    def convert_template_to_prompt(template: str, field_map: dict[str, str]) -> str:
        prompt = template
        for k, v in field_map.items():
            prompt = prompt.replace(f"{{{k}}}", v)  # {k} --> v
        msg.good("chatgpt | Successfully created prompt")
        print(prompt)
        return prompt

    @retry(wait=wait_random_exponential(min=5), stop=stop_after_attempt(12), after=(
            lambda retry_state: msg.warn(f"Retrying - {retry_state.attempt_number}-th time")))  # TODO can't use self
    def completion_with_backoff(self, prompt: str, **kwargs):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.config.model,
            response_format={"type": "json_object"},  # json_object require prompt to contain word "JSON"
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            seed=self.config.seed,
            **kwargs
        )
        return response

    @staticmethod
    def parse_completion_as_list(response) -> list[dict]:
        # Retrieve first completion
        str_completion = response.choices[0].message.content
        try:
            # Convert to Python object
            dict_completion = ast.literal_eval(str_completion)  # handle both single and double quotes
            msg.good("chatgpt | Successfully parse completion", dict_completion)
            return list(dict_completion.values())  # return [{"question": ..., "label": ...}, ...]
        except SyntaxError:
            msg.warn("chatgpt | Completion cannot be parsed as Python object.", f"Completion: {str_completion}")
            return list()

    def process_file(self, file_path: str, template_path: str, label_to_keep: Optional[str] = None) -> None:
        """
        Note: has to be consistent with process fn. Implemented separately due to increment saving
        """
        # Load data
        data = data_ioer.load(file_path)
        # Create field maps
        field_maps = self.create_field_maps(data, int(self.config.TPM / self.config.RPM))  # TODO
        num_batch = len(field_maps)

        list_of_completions = list()

        # Iterate through each field map
        for idx, field_map in enumerate(tqdm(field_maps)):
            idx = str(idx + 1).zfill(len(str(num_batch)))
            completion = self._process_field_map(field_map, template_path)  # do it one by one, s.t. completion is saved
            # If failing, save field map to JSON
            if completion == list():  # linked to syntax error line
                target_path = str(Path(file_path).parent / "internal-openai" / "fail_batches" /
                                  f"{Path(file_path).stem}-{Path(template_path).stem}-batch-{idx}-of-{num_batch}.json")
                data_ioer.save(field_map, target_path)
            # If successful, save completion to JSONL
            else:
                target_path = str(Path(file_path).parent / "internal-openai" / "good_batches" /
                                  f"{Path(file_path).stem}-{Path(template_path).stem}-batch-{idx}-of-{num_batch}.jsonl")
                data_ioer.save(completion, target_path)
            list_of_completions.append(completion)
        # Save all completions from all field maps
        completions = sum(list_of_completions, [])
        target_path = str(Path(file_path).parent / "internal-openai" / 
                          f"{Path(file_path).stem}-{Path(template_path).stem}.jsonl")
        data_ioer.save(completions, target_path)

        # TODO: fix this
        if "labelers" in template_path:
            if label_to_keep is None:
                label_to_keep = input("Type  the label you want to keep, e.g. YesNo.\n")
            data_to_keep = [c for c in completions if c["label"] == label_to_keep]
            target_path = str(Path(file_path).parent / f"{Path(file_path).stem}-{label_to_keep}.jsonl")
            data_ioer.save(data_to_keep, target_path)

    def create_field_maps(self, data: Any, token_threshold: int) -> dict[str, str]:
        pass

    def batch_data(self, data: pd.DataFrame, token_threshold: int) -> list:
        counts = data.apply(self.check_num_token).tolist()
        return self._batch_data_by_threshold(data.tolist(), counts, token_threshold)

    def _batch_data_by_threshold(self, data: list, counts: list[int], token_threshold: int) -> list:
        assert len(data) == len(counts)
        import sys
        sys.setrecursionlimit(10000)
        running_count = 0
        for idx in range(1, len(counts)):
            running_count += counts[idx]
            if running_count > token_threshold:
                batch = "\n\n".join(data[:idx])
                return [batch] + self._batch_data_by_threshold(data[idx:], counts[idx:], token_threshold)
        return ["\n\n".join(data)]

    def check_num_token(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


if __name__ == "__main__":
    fire.Fire(ChatGPT)
