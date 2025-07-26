"""Data models used in the project."""

from dataclasses import dataclass, field


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
    """

    instruction: str
    input: str
    output: str
    most_similar_instructions: dict[str, float] = field(default_factory=dict)
    avg_similarity_score: float = float("nan")
