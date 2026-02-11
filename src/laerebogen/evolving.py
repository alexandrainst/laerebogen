"""Evolving an existing instruction-following dataset.

This is the method described in [1], using an existing instruction-tuned language model
to generate new instructions based on an existing dataset, to make the dataset more
complicated and diverse.

[1] https://doi.org/10.48550/arXiv.2304.12244
"""

import logging
import random
import typing as t
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from .data_models import InstructionSample
from .vllm_utils import generate_text_with_vllm

if t.TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


METHODS = dict(
    add_constraints="Tilføj endnu en begrænsning eller krav til instruktionen.",
    deepen="Hvis instruktionen indeholder forespørgsler om bestemte emner, skal du øge "
    "dybden og bredden af forespørgslen.",
    concretise="Erstat generelle begreber med mere specifikke begreber.",
    increase_reasoning_steps="Hvis den originale prompt kan løses med blot et par "
    "enkle tankeprocesser, kan du omskrive den, så den eksplicit beder om at "
    "ræsonnere i flere trin.",
    complicate_input="Du skal tilføje {format} data til den nye instruktion. Dette "
    "kan både være som input til den nye instruktion, eller bede om at få svaret "
    "i {format} format.",
)


FORMATS = ["JSON", "YAML", "Markdown", "Python"]


def evolve_instructions(
    instructions: list[InstructionSample],
    model: "LLM",
    rewriter_prompt_path: str,
    creator_prompt_path: str,
) -> list[InstructionSample]:
    """Evolve an instruction using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to evolve.
        model:
            The instruction-tuned large language model to use for evolution.
        rewriter_prompt_path:
            Path to the prompt file containing the rewriter prompt.
        creator_prompt_path:
            Path to the prompt file containing the creator prompt.

    Returns:
        The evolved instructions as well as the original instructions, shuffled.
    """
    # Deep copy the instructions to avoid side effects
    instructions = deepcopy(instructions)

    # Load the prompts
    with Path(rewriter_prompt_path).open() as f:
        rewriter_prompt = f.read() + "\n"
    with Path(creator_prompt_path).open() as f:
        creator_prompt = f.read() + "\n"

    # Prepare the prompt templates
    templates = [
        rewriter_prompt.replace("{method}", method) for method in METHODS.values()
    ] + [creator_prompt]

    # Evolve the instructions
    logger.info("Evolving instructions...")
    prompts = [
        random.choice(templates)
        .replace("{format}", random.choice(FORMATS))
        .format(instruction=instruction.model_dump_json())
        for instruction in instructions
    ]
    evolved_instructions: list[InstructionSample] = list()
    for response in generate_text_with_vllm(
        prompts=prompts,
        model=model,
        apply_chat_template=True,
        response_format=InstructionSample,
    ):
        if response.done_reason != "stop":
            continue
        try:
            evolved_instruction = InstructionSample.model_validate_json(
                json_data=response.completion
            )
            evolved_instructions.append(evolved_instruction)
        except ValidationError:
            continue

    return evolved_instructions
