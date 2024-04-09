# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

from dataclasses import dataclass


@dataclass
class ChatModelConfig:
    model: str = "gpt-4-1106-preview"
    TPM: int = 80000
    RPM: int = 500
    RPD: int = 10000
    temperature: int = 0
    top_p: int = 1
    frequency_penalty: int = 0
    presence_penalty: int = 0
    seed: int = 42
