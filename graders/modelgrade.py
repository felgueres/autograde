# Contains the base model graded spec class 

from typing import Optional, Union
from dataclasses import dataclass
from prompt import OpenAICreateChatPrompt

@dataclass
class ModelGradedSpec:
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[list[str], str]
    input_outputs: dict[str, str]

    eval_type: Optional[str] = None
    choice_scores: Optional[Union[dict[str,float], str]] = None
    output_template: Optional[str] = None
