"""Correcting generated instructions."""

import json
import logging
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .data_models import GrammarCorrectionResponse, InstructionSample
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
    tokenizer = model.get_tokenizer()

    # Load the prompt
    with Path(prompt_path).open() as f:
        correction_prompt = f.read() + "\n"

    # Copy the instructions to avoid modifying the original ones
    corrected_instructions = deepcopy(instructions)

    # Correct the instructions
    logger.info("Correcting grammar in instructions...")
    prompts = [
        correction_prompt.format(
            text=instruction.instruction if instruction.instruction else "<empty>"
        )
        for instruction in instructions
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in tqdm(
            iterable=prompts,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]
    responses = generate_text_with_vllm(
        prompts=prompts, model=model, response_format=GrammarCorrectionResponse
    )
    for instruction, response in zip(corrected_instructions, responses):
        response.completion = GrammarCorrectionResponse.model_validate_json(
            response.completion
        ).corrected_instruction
        if response.done_reason == "stop":
            instruction.instruction = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

    # Correct the inputs
    logger.info("Correcting grammar in inputs...")
    prompts = [
        correction_prompt.format(
            text=instruction.input if instruction.input else "<empty>"
        )
        for instruction in instructions
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in tqdm(
            iterable=prompts,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            instruction.input = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

    # Correct the outputs
    logger.info("Correcting grammar in outputs...")
    prompts = [
        correction_prompt.format(
            text=instruction.output if instruction.output else "<empty>"
        )
        for instruction in instructions
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in tqdm(
            iterable=prompts,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            instruction.output = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

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
    tokenizer = model.get_tokenizer()

    # Load the prompt
    with Path(prompt_path).open() as f:
        correction_prompt = f.read() + "\n"

    # Copy the instructions to avoid modifying the original ones
    corrected_instructions = deepcopy(instructions)

    # Correct the instructions
    logger.info("Correcting bad quality instructions...")
    prompts = [
        correction_prompt.format(prompt=repr(instruction))
        for instruction in instructions
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in tqdm(
            iterable=prompts,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]

    class ResponseFormat(BaseModel):
        """Response format for the vLLM model."""

        instruction: str
        input: str
        output: str

    responses = generate_text_with_vllm(
        prompts=prompts, model=model, response_format=ResponseFormat
    )
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            try:
                new_instruction = ResponseFormat.model_validate_json(
                    json_data=response.completion
                )
            except (json.JSONDecodeError, ValidationError):
                continue
            instruction.instruction = new_instruction.instruction.strip()
            instruction.input = new_instruction.input.strip()
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
