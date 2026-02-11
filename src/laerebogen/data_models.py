"""Data models used in the project."""

import typing as t
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .constants import NUM_PROMPT_INSTRUCTIONS


class Message(BaseModel):
    """A message in a conversation between a user and an LLM."""

    role: t.Literal["user", "assistant"]
    content: t.Annotated[str, Field(min_length=1)]


class Conversation(BaseModel):
    """A conversation between a user and an LLM, used for instruction tuning.

    Attributes:
        messages:
            A list of messages in the conversation, where each message is a dictionary
            with keys "role" and "content".
    """

    messages: list[Message]

    def add_message(self, role: t.Literal["user", "assistant"], content: str) -> None:
        """Add a message to the conversation.

        Args:
            role:
                The role of the message sender (e.g., "user", "assistant").
            content:
                The content of the message.
        """
        self.messages.append(Message(role=role, content=content))


@dataclass
class Response:
    """A response from an LLM.

    Attributes:
        completion:
            The LLM's completion of the prompt.
        done_reason:
            The reason why the LLM stopped generating.
    """

    completion: str
    done_reason: str | None


class InstructionSample(BaseModel):
    """An instruction sample for the model.

    Attributes:
        instruction:
            The instruction to be followed by the model.
        output:
            The expected output of the instruction.
    """

    instruction: t.Annotated[str, Field(min_length=10, max_length=5000)]
    output: t.Annotated[str, Field(min_length=1)]


class InstructionInput(BaseModel):
    """An input to the instruction generation model.

    Attributes:
        instruction:
            The instruction to be followed by the model.
    """

    instruction: t.Annotated[str, Field(min_length=10, max_length=5000)]


class InstructionOutput(BaseModel):
    """An output from the instruction generation model.

    Attributes:
        output:
            The expected output of the instruction.
    """

    output: t.Annotated[str, Field(min_length=1)]


class InstructionInputSamples(BaseModel):
    """A list of generated instruction inputs."""

    instructions: t.Annotated[
        list[InstructionInput],
        Field(min_items=NUM_PROMPT_INSTRUCTIONS, max_items=NUM_PROMPT_INSTRUCTIONS),
    ]
