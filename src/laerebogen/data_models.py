"""Data models used in the project."""

import hashlib
import typing as t
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .constants import NUM_PROMPT_INSTRUCTIONS


class Message(BaseModel):
    """A message in a conversation between a user and an LLM."""

    role: t.Literal["user", "assistant"]
    content: t.Annotated[str, Field(min_length=1)]
    hash: str | None = None

    def model_post_init(self, __context: None) -> None:
        """Compute the hash of the message after it is initialised."""
        self.hash = hashlib.md5(string=(self.role + self.content).encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        """Check if two messages are equal.

        Args:
            other:
                The other message to compare to.

        Returns:
            True if the messages are equal, False otherwise.
        """
        if not isinstance(other, Message):
            return False
        return self.hash == other.hash


class Conversation(BaseModel):
    """A conversation between a user and an LLM, used for instruction tuning.

    Attributes:
        messages:
            A list of messages in the conversation, where each message is a dictionary
            with keys "role" and "content".
    """

    messages: list[Message]
    hash: str | None = None

    def model_post_init(self, __context: None) -> None:
        """Compute the hash of the conversation after it is initialised."""
        self.hash = hashlib.md5(
            string="".join(message.hash or "" for message in self.messages).encode()
        ).hexdigest()

    def add_message(self, role: t.Literal["user", "assistant"], content: str) -> None:
        """Add a message to the conversation.

        Args:
            role:
                The role of the message sender (e.g., "user", "assistant").
            content:
                The content of the message.
        """
        self.messages.append(Message(role=role, content=content))
        self.hash = hashlib.md5(
            string="".join(message.hash or "" for message in self.messages).encode()
        ).hexdigest()

    def __eq__(self, other: object) -> bool:
        """Check if two conversations are equal.

        Args:
            other:
                The other conversation to compare to.

        Returns:
            True if the conversations are equal, False otherwise.
        """
        if not isinstance(other, Conversation):
            return False
        return self.hash == other.hash


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
    hash: str | None = None

    def model_post_init(self, __context: None) -> None:
        """Compute the hash of the instruction after it is initialised."""
        self.hash = hashlib.md5(
            string=(self.instruction + self.output).encode()
        ).hexdigest()

    def __eq__(self, other: object) -> bool:
        """Check if two instruction samples are equal.

        Args:
            other:
                The other instruction sample to compare to.

        Returns:
            True if the instruction samples are equal, False otherwise.
        """
        if not isinstance(other, InstructionSample):
            return False
        return self.hash == other.hash


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
