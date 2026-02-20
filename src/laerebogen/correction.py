"""Correcting generated instructions."""

import collections.abc as c
import logging
from pathlib import Path

import more_itertools as mit
from pydantic import ValidationError
from tqdm.auto import tqdm

from .data_models import InstructionSample
from .vllm_utils import generate_text_with_vllm, load_vllm_model

logger = logging.getLogger(__name__)


def correct_instructions(
    instructions: list[InstructionSample],
    prompt_path: Path,
    model_id: str,
    batch_size: int,
) -> c.Generator[InstructionSample, None, None]:
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
            The batch size to use for correction.

    Yields:
        The corrected instructions.
    """
    with prompt_path.open() as f:
        correction_prompt = f.read() + "\n"

    model = load_vllm_model(model_id=model_id)

    num_batches = len(instructions) // batch_size
    if len(instructions) % batch_size:
        num_batches += 1

    logger.info("Correcting grammar in instructions...")

    for batch in tqdm(
        iterable=mit.chunked(iterable=instructions, n=batch_size),
        desc="Correcting grammar",
        total=num_batches,
        unit="batch",
    ):
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
            if response.done_reason != "stop":
                continue
            try:
                corrected_instruction = InstructionSample.model_validate_json(
                    response.completion
                )
                yield corrected_instruction
            except ValidationError as e:
                logger.warning(f"Error while correcting instruction: {e}")
                continue
