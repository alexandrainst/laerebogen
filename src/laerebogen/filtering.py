"""Filtering of generated instructions."""

import logging
import re

from .data_models import InstructionSample

logger = logging.getLogger(__name__)


def keep_instruction(instruction_sample: InstructionSample) -> bool:
    """Check if an instruction sample should be kept based on all filters.

    Args:
        instruction_sample:
            The instruction sample to check.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    all_filters = [not_too_short, not_too_long, does_not_contain_prompt_words]
    for filter_fn in all_filters:
        if not filter_fn(instruction_sample):
            logger.debug(
                "The following instruction was filtered out by the "
                f"{filter_fn.__name__!r} filter:\n{instruction_sample.instruction}"
            )
            return False
    return True


def not_too_short(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that are too short.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    num_instruction_words = len(instruction_sample.instruction.split())
    return num_instruction_words > 3


def not_too_long(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that are too long.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    num_instruction_words = len(instruction_sample.instruction.split())
    return num_instruction_words < 300


def does_not_contain_prompt_words(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that contain prompt words.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    prompt_words_regex = re.compile(pattern=r"Instruktion|Input|Output")
    instruction_contains_prompt_words = (
        re.search(pattern=prompt_words_regex, string=instruction_sample.instruction)
        is not None
    )
    output_contains_prompt_words = (
        re.search(pattern=prompt_words_regex, string=instruction_sample.output)
        is not None
    )
    return not instruction_contains_prompt_words and not output_contains_prompt_words
