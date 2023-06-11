'''Contains types and classes to create prompts for chat models
Original source: https://github.com/openai/evals/blob/main/evals/prompt/base.py#L19
'''
from typing import List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# This is an approximation to the type accepted as the `prompt` field to `openai.Completion.create` calls
OpenAICreatePrompt = Union[str, list[str], list[int], list[list[int]]]

OpenAIChatMessage = Dict[str, str]  # A message is a dictionary with "role" and "content" keys
OpenAICreateChatPrompt = List[OpenAIChatMessage]  # A chat log is a list of messages

def chat_prompt_to_text_prompt(prompt: OpenAICreateChatPrompt, for_completion: bool = True) -> str:
    """
    Render a chat prompt as a text prompt. User and assistant messages are separated by newlines
    and prefixed with "User: " and "Assistant: ", respectively, unless there is only one message.
    System messages have no prefix.
    """
    assert is_chat_prompt(prompt), f"Expected a chat prompt, got {prompt}"
    chat_to_prefixes = {
        # roles
        "system": "",
        # names
        "example_user": "User: ",
        "example_assistant": "Assistant: ",
    }

    # For a single message, be it system, user, or assistant, just return the message
    if len(prompt) == 1:
        return prompt[0]["content"]

    text = ""
    for msg in prompt:
        role = msg["name"] if "name" in msg else msg["role"]
        prefix = chat_to_prefixes.get(role, role.capitalize() + ": ")
        content = msg["content"]
        text += f"{prefix}{content}\n"
    if for_completion:
        text += "Assistant: "
    return text.lstrip()


def text_prompt_to_chat_prompt(prompt: str, role: str = "system") -> OpenAICreateChatPrompt:
    assert isinstance(prompt, str), f"Expected a text prompt, got {prompt}"
    return [
        {"role": role, "content": prompt},
    ]


@dataclass
class Prompt(ABC):
    """
    A `Prompt` encapsulates everything required to present the `raw_prompt` in different formats,
    e.g., a normal unadorned format vs. a chat format.
    """

    @abstractmethod
    def to_formatted_prompt(self):
        """
        Return the actual data to be passed as the `prompt` field to your model.
        See the above types to see what each API call is able to handle.
        """

def is_chat_prompt(prompt: Prompt) -> bool:
    return isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt)

@dataclass
class ChatCompletionPrompt(Prompt):
    """
    A `Prompt` object that wraps prompts to be compatible with chat models, which use `openai.ChatCompletion.create`.

    The format expected by chat models is a list of messages, where each message is a dict with "role" and "content" keys.
    """

    raw_prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt]

    def _render_text_as_chat_prompt(self, prompt: str) -> OpenAICreateChatPrompt:
        """
        Render a text string as a chat prompt. The default option we adopt here is to simply take the full prompt
        and treat it as a system message.
        """
        return text_prompt_to_chat_prompt(prompt)

    def to_formatted_prompt(self) -> OpenAICreateChatPrompt:
        if is_chat_prompt(self.raw_prompt):
            return self.raw_prompt
        return self._render_text_as_chat_prompt(self.raw_prompt)

