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
from tqdm.auto import tqdm

from .data_models import EvolvedInstruction, EvolvedOutput, InstructionSample
from .filtering import keep_instruction
from .vllm_utils import generate_text_with_vllm

if t.TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


METHODS = dict(
    add_constraints="Tilføj venligst endnu en begrænsning eller krav til "
    "instruktionen.",
    deepen="Hvis instruktionen indeholder forespørgsler om bestemte emner, kan "
    "dybden og bredden af forespørgslen øges.",
    concretise="Erstat venligst generelle begreber med mere specifikke begreber.",
    increase_reasoning_steps="Hvis #Original Prompt# kan løses med blot et par enkle "
    "tankeprocesser, kan du omskrive den, så den eksplicit beder om at ræsonnere i "
    "flere trin.",
    complicate_input="Du skal tilføje {format} data som inputdata til "
    "#Omskrevet Prompt#.",
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
        .format(instruction=instruction.instruction)
        for instruction in instructions
    ]
    evolved_instructions: list[InstructionSample] = list()
    for response in generate_text_with_vllm(
        prompts=prompts,
        model=model,
        apply_chat_template=True,
        response_format=EvolvedInstruction,
    ):
        if response.done_reason != "stop":
            continue
        try:
            new_prompt = EvolvedInstruction.model_validate_json(
                json_data=response.completion
            ).new_prompt
        except ValidationError:
            continue

        evolved_instructions.append(
            InstructionSample(instruction=new_prompt, output="")
        )

    # Get the corresponding outputs
    logger.info("Generating outputs for evolved instructions...")
    responses = generate_text_with_vllm(
        prompts=prompts,
        model=model,
        apply_chat_template=True,
        response_format=EvolvedOutput,
    )
    for instruction, output in zip(evolved_instructions, responses):
        if output.done_reason == "stop":
            try:
                instruction.output = EvolvedOutput.model_validate_json(
                    output.completion.strip()
                ).new_output
            except ValidationError:
                continue

    # Filter the evolved instructions
    logger.info("Filtering evolved instructions...")
    evolved_instructions = [
        instruction
        for instruction in tqdm(
            iterable=evolved_instructions,
            desc="Filtering evolved instructions",
            unit="instruction",
        )
        if keep_instruction(instruction_sample=instruction)
    ]

    return evolved_instructions
