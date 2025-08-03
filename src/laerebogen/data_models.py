"""Data models used in the project."""

import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Conversation:
    """A conversation between a user and an LLM, used for instruction tuning.

    Attributes:
        messages:
            A list of messages in the conversation, where each message is a dictionary
            with keys "role" and "content".
    """

    messages: list[dict[Literal["role", "content"], str]] = field(default_factory=list)

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
        return json.dumps(self, ensure_ascii=False)

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
            instruction_sample = InstructionSample.from_json(json_str)
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
        conversation.add_message(
            role="user",
            content=f"{instruction_sample.instruction}\n\n{instruction_sample.input}"
            if instruction_sample.input
            else instruction_sample.instruction,
        )
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
class InstructionSample:
    """An instruction sample for the model.

    Attributes:
        instruction:
            The instruction to be followed by the model.
        input:
            The input to the instruction - can be empty.
        output:
            The expected output of the instruction.
        most_similar_instructions:
            A dictionary of the most similar instructions to the current instruction,
            with their similarity scores.
        avg_similarity_score:
            The average similarity score of the most similar instructions.
        instruction_tokens:
            A list of token IDs representing the instruction.
    """

    instruction: str
    input: str
    output: str
    most_similar_instructions: dict[str, float] = field(default_factory=dict)
    avg_similarity_score: float = float("nan")
    instruction_tokens: list[str] = field(default_factory=list)

    def json(self) -> str:
        """Convert the instruction sample to a JSON string.

        Returns:
            A JSON string representation of the instruction sample.
        """
        return json.dumps(
            dict(
                instruction=self.instruction,
                input=self.input,
                output=self.output,
                most_similar_instructions=self.most_similar_instructions,
                avg_similarity_score=self.avg_similarity_score,
            ),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionSample":
        """Create an instruction sample from a JSON string.

        Args:
            json_str:
                A JSON string representation of the instruction sample.

        Returns:
            An instance of InstructionSample.

        Raises:
            ValueError:
                If the JSON string is invalid.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {json_str!r}") from e
        return cls(
            instruction=data["instruction"],
            input=data["input"],
            output=data["output"],
            most_similar_instructions=data.get("most_similar_instructions", {}),
            avg_similarity_score=data.get("avg_similarity_score", float("nan")),
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
