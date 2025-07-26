"""Data models used in the project."""

from dataclasses import dataclass, field


@dataclass
class InstructionSample:
    """An instruction sample for the model."""

    instruction: str
    input: str
    output: str
    most_similar_instructions: dict[str, float] = field(default_factory=dict)
    avg_similarity_score: float = float("nan")
