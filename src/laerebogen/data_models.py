"""Data models used in the project."""

import json
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class Conversation:
    """A conversation between a user and an LLM, used for instruction tuning.

    Attributes:
        messages:
            A list of messages in the conversation, where each message is a dictionary
            with keys "role" and "content".
    """

    messages: list[dict[t.Literal["role", "content"], str]] = field(
        default_factory=list
    )

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.

        Args:
            role:
                The role of the message sender (e.g., "user", "assistant").
            content:
                The content of the message.
        """
        self.messages.append({"role": role, "content": content})

    def json(self) -> str:
        """Convert the conversation to a JSON string.

        Returns:
            A JSON string representation of the conversation.
        """
        return json.dumps(dict(messages=self.messages), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Conversation":
        """Create a conversation from a JSON string.

        Args:
            json_str:
                A JSON string representation of the conversation.

        Returns:
            An instance of Conversation.

        Raises:
            ValueError:
                If the JSON string is invalid or does not contain the expected
                structure.
        """
        try:
            conversation = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {json_str!r}") from e
        if not isinstance(conversation, dict):
            raise ValueError(
                f"Expected a dictionary, got {type(conversation).__name__}"
            )
        if "messages" in conversation:
            return cls(messages=conversation["messages"])
        elif "instruction" in conversation:
            instruction_sample = InstructionSample.model_validate_json(json_str)
            return cls.from_instruction_sample(instruction_sample=instruction_sample)
        else:
            raise ValueError(
                "JSON string does not contain 'messages' or 'instruction' key."
            )

    @classmethod
    def from_instruction_sample(
        cls, instruction_sample: "InstructionSample"
    ) -> "Conversation":
        """Create a conversation from an instruction sample.

        Args:
            instruction_sample:
                An instance of InstructionSample.

        Returns:
            An instance of Conversation with the instruction and output as messages.
        """
        conversation = cls()
        conversation.add_message(role="user", content=instruction_sample.instruction)
        conversation.add_message(role="assistant", content=instruction_sample.output)
        return conversation

    def __str__(self) -> str:
        """Return a string representation of the conversation.

        Returns:
            A string representation of the conversation.
        """
        return "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in self.messages
        )


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

    instruction: str
    output: str


class InstructionSamples(BaseModel):
    """A list of 20 generated instructions."""

    instructions: t.Annotated[
        list[InstructionSample], Field(min_items=20, max_items=20)
    ]


class GrammarCorrectionResponse(BaseModel):
    """A response from the grammar correction model."""

    corrected_text: t.Annotated[str, Field(min_length=1)]


class EvolvedInstruction(BaseModel):
    """An evolved instruction."""

    new_prompt: t.Annotated[str, Field(min_length=1)]


class EvolvedOutput(BaseModel):
    """An evolved output."""

    new_output: t.Annotated[str, Field(min_length=1)]
