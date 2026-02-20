"""Evolving an existing instruction-following dataset.

This is the method described in [1], using an existing instruction-tuned language model
to generate new instructions based on an existing dataset, to make the dataset more
complicated and diverse.

[1] https://doi.org/10.48550/arXiv.2304.12244
"""

import collections.abc as c
import logging
import random
import typing as t
from pathlib import Path

import more_itertools as mit
from pydantic import ValidationError
from tqdm.auto import tqdm

from .data_models import InstructionSample
from .vllm_utils import generate_text_with_vllm, load_vllm_model

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
    model_id: str,
    rewriter_prompt_path: Path,
    creator_prompt_path: Path,
    num_evolutions: int,
    batch_size: int,
) -> c.Generator[tuple[InstructionSample, int], None, None]:
    """Evolve an instruction using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to evolve.
        model_id:
            The ID of the instruction-tuned large language model to use for evolution.
        rewriter_prompt_path:
            Path to the prompt file containing the rewriter prompt.
        creator_prompt_path:
            Path to the prompt file containing the creator prompt.
        num_evolutions:
            The number of evolutions to perform.
        batch_size:
            The batch size to use for generating new instructions.

    Yields:
        Pairs (instruction, evolution) where instruction is the evolved instruction and
        evolution is the current evolution iteration.
    """
    for instruction in instructions:
        yield instruction, 0

    with rewriter_prompt_path.open() as f:
        rewriter_prompt = f.read() + "\n"
    with creator_prompt_path.open() as f:
        creator_prompt = f.read() + "\n"

    model = load_vllm_model(model_id=model_id)

    prompt_templates = [
        rewriter_prompt.replace("{method}", method) for method in METHODS.values()
    ] + [creator_prompt]

    pbar = (
        tqdm(
            iterable=range(1, 1 + num_evolutions),
            desc="Evolving dataset",
            unit="evolution",
        )
        if num_evolutions > 1
        else range(1, 1 + num_evolutions)
    )
    for evolution in pbar:
        yield from evolve_single_iteration(
            instructions=instructions,
            prompt_templates=prompt_templates,
            model=model,
            batch_size=batch_size,
            evolution=evolution,
        )


def evolve_single_iteration(
    instructions: list[InstructionSample],
    prompt_templates: list[str],
    model: "LLM",
    batch_size: int,
    evolution: int,
) -> c.Generator[tuple[InstructionSample, int], None, None]:
    """Evolve a single iteration of the dataset.

    Args:
        instructions:
            The instructions to evolve.
        prompt_templates:
            The prompt templates to use for evolution.
        model:
            The instruction-tuned large language model to use for evolution.
        batch_size:
            The batch size to use for evolution.
        evolution:
            The current evolution iteration.

    Yields:
        Pairs (instruction, evolution) where instruction is the evolved instruction
        and evolution is the current evolution iteration.
    """
    prompts = [
        random.choice(prompt_templates)
        .replace("{format}", random.choice(FORMATS))
        .format(instruction=instruction.model_dump_json())
        for instruction in instructions
    ]

    num_batches = len(prompts) // batch_size
    if len(prompts) % batch_size:
        num_batches += 1

    for batch in tqdm(
        iterable=mit.chunked(iterable=instructions, n=batch_size),
        desc="Correcting grammar",
        total=num_batches,
        unit="batch",
    ):
        responses = generate_text_with_vllm(
            prompts=batch,
            model=model,
            apply_chat_template=True,
            response_format=InstructionSample,
        )
        for response in responses:
            if response.done_reason != "stop":
                continue
            try:
                evolved_instruction = InstructionSample.model_validate_json(
                    json_data=response.completion
                )
                yield evolved_instruction, evolution
            except ValidationError:
                continue
