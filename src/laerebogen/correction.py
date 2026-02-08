"""Correcting generated instructions."""

import logging
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError
from tqdm.auto import tqdm

from .data_models import InstructionSample
from .filtering import keep_instruction
from .vllm_utils import generate_text_with_vllm, load_vllm_model

logger = logging.getLogger(__name__)


def correct_grammar_in_instructions(
    instructions: list[InstructionSample], prompt_path: str, model_id: str
) -> list[InstructionSample]:
    """Correct spelling mistakes using an instruction-tuned large language model.

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
    logger.info(f"Loading model {model_id!r} for correcting grammar in instructions...")
    model = load_vllm_model(model_id=model_id)

    # Load the prompt
    with Path(prompt_path).open() as f:
        correction_prompt = f.read() + "\n"

    # Correct the instructions
    logger.info("Correcting grammar in instructions...")
    prompts = [
        correction_prompt.format(instruction=instruction.instruction)
        for instruction in instructions
    ]
    responses = generate_text_with_vllm(
        prompts=prompts,
        model=model,
        apply_chat_template=True,
        response_format=InstructionSample,
    )

    # Convert the responses to InstructionSample objects
    corrected_instructions: list[InstructionSample] = list()
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

    # Filter the corrected instructions
    logger.info(f"Filtering {len(corrected_instructions):,} corrected instructions...")
    corrected_instructions = [
        instruction
        for instruction in tqdm(
            iterable=corrected_instructions,
            desc="Filtering corrected instructions",
            unit="instruction",
        )
        if keep_instruction(instruction_sample=instruction)
    ]

    return corrected_instructions


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

    # Filter the corrected instructions
    logger.info(f"Filtering {len(corrected_instructions):,} corrected instructions...")
    corrected_instructions = [
        instruction
        for instruction in tqdm(
            iterable=corrected_instructions,
            desc="Filtering corrected instructions",
            unit="instruction",
        )
        if keep_instruction(instruction_sample=instruction)
    ]

    return corrected_instructions
