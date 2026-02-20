"""Correcting generated instructions."""

import collections.abc as c
import logging
from copy import deepcopy
from pathlib import Path

import more_itertools as mit
from pydantic import ValidationError

from .data_models import InstructionSample
from .vllm_utils import generate_text_with_vllm, load_vllm_model

logger = logging.getLogger(__name__)


def correct_grammar_in_instructions(
    instructions: list[InstructionSample],
    prompt_path: str,
    model_id: str,
    batch_size: int,
) -> c.Generator[list[InstructionSample], None, None]:
    """Correct spelling mistakes using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to correct.
        prompt_path:
            Path to the prompt file containing the correction prompt.
        model_id:
            The model ID of the instruction-tuned large language model to use for
            correction.
        batch_size:
            The number of samples to process with the LLM at the same time.

    Yields:
        Batches of corrected instructions.
    """
    # Load the model and tokenizer
    logger.info(f"Loading model {model_id!r} for correcting grammar in instructions...")
    model = load_vllm_model(model_id=model_id)

    # Load the prompt
    with Path(prompt_path).open() as f:
        correction_prompt = f.read() + "\n"

    logger.info("Correcting grammar in instructions...")

    corrected_instructions: list[InstructionSample] = list()
    for batch in mit.chunked(iterable=instructions, n=batch_size):
        prompts = [
            correction_prompt.format(instruction=instruction.instruction)
            for instruction in batch
        ]
        responses = generate_text_with_vllm(
            prompts=prompts,
            model=model,
            apply_chat_template=True,
            response_format=InstructionSample,
        )
        for response in responses:
            if response.done_reason == "stop":
                continue
            try:
                corrected_instruction = InstructionSample.model_validate_json(
                    response.completion
                )
                corrected_instructions.append(corrected_instruction)
            except ValidationError:
                continue

        yield corrected_instructions


def correct_bad_quality_instructions(
    instructions: list[InstructionSample], prompt_path: str, model_id: str
) -> list[InstructionSample]:
    """Correct bad quality instructions using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to correct.
        prompt_path:
            Path to the prompt file containing the correction prompt.
        model_id:
            The model ID of the instruction-tuned large language model to use for
            correction.

    Returns:
        The corrected instructions.
    """
    # Load the model and tokenizer
    logger.info(
        f"Loading model {model_id!r} for correcting bad quality instructions..."
    )
    model = load_vllm_model(model_id=model_id)

    # Load the prompt
    with Path(prompt_path).open() as f:
        correction_prompt = f.read() + "\n"

    # Copy the instructions to avoid modifying the original ones
    corrected_instructions = deepcopy(instructions)

    # Correct the instructions
    logger.info("Correcting bad quality instructions...")
    prompts = [
        correction_prompt.format(instruction=instruction.model_dump_json())
        for instruction in instructions
    ]
    responses = generate_text_with_vllm(
        prompts=prompts,
        model=model,
        apply_chat_template=True,
        response_format=InstructionSample,
    )
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            try:
                new_instruction = InstructionSample.model_validate_json(
                    json_data=response.completion
                )
            except ValidationError:
                continue
            instruction.instruction = new_instruction.instruction.strip()
            instruction.output = new_instruction.output.strip()

    return corrected_instructions
