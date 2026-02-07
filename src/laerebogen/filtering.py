"""Filtering of generated instructions."""

import logging
import re
import string

from lingua import Language, LanguageDetectorBuilder

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
    all_filters = [
        not_too_short,
        not_too_long,
        does_not_contain_prompt_words,
        does_not_contain_banned_word,
        does_not_start_with_write_a_program,
        does_not_start_with_punctuation,
        starts_with_danish_character,
        is_danish,
        is_not_similar_to_existing_instructions,
    ]
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
    num_input_words = len(instruction_sample.input.split())
    return num_instruction_words < 150 and num_input_words < 150


def does_not_contain_prompt_words(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that contain prompt words.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    prompt_words_regex = re.compile(pattern=r"[0-9]+\. (Instruktion|Input|Output)")
    instruction_contains_prompt_words = (
        re.search(pattern=prompt_words_regex, string=instruction_sample.instruction)
        is not None
    )
    input_contains_prompt_words = (
        re.search(pattern=prompt_words_regex, string=instruction_sample.input)
        is not None
    )
    output_contains_prompt_words = (
        re.search(pattern=prompt_words_regex, string=instruction_sample.output)
        is not None
    )
    return (
        not instruction_contains_prompt_words
        and not input_contains_prompt_words
        and not output_contains_prompt_words
    )


def does_not_contain_banned_word(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that contain banned words.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    banned_words = [
        "flowchart",
        "diagram",
        "billede",
        "billeder",
        "graf",
        "grafer",
        "foto",
        "fotos",
        "fil",
        "filer",
        "illustrer",
        "illustration",
        "tegning",
        "gå til",
        "musik",
    ]
    return (
        re.search(
            pattern=r"\b|\b".join([rf"\b{word}\b" for word in banned_words]),
            string=instruction_sample.instruction,
            flags=re.IGNORECASE,
        )
        is None
    )


def does_not_start_with_write_a_program(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that starts with 'Skriv et program'.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    return not instruction_sample.instruction.lower().startswith("skriv et program")


def does_not_start_with_punctuation(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that start with punctuation.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    return not instruction_sample.instruction.startswith(tuple(string.punctuation))


def starts_with_danish_character(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that do not start with a Danish character.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    return instruction_sample.instruction[
        0
    ].isascii() or instruction_sample.instruction.lower().startswith(tuple("æøå"))


def is_danish(instruction_sample: InstructionSample) -> bool:
    """Filter out instructions that are not in Danish.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    texts_that_need_detection = [
        instruction_sample.instruction,
        instruction_sample.output,
    ]
    if instruction_sample.input and instruction_sample.input != "<empty>":
        texts_that_need_detection.append(instruction_sample.input)

    detector = LanguageDetectorBuilder.from_all_languages().build()
    language_confidences = [
        detector.compute_language_confidence(text=text, language=Language.DANISH)
        for text in texts_that_need_detection
    ]
    return all(confidence > 0.7 for confidence in language_confidences)


def is_not_similar_to_existing_instructions(
    instruction_sample: InstructionSample,
) -> bool:
    """Filter out instructions that are too similar to existing instructions.

    Args:
        instruction_sample:
            The instruction sample to filter.

    Returns:
        True if the instruction should be kept, False if it should be filtered out.
    """
    if not instruction_sample.most_similar_instructions:
        return True
    max_similarity = max(instruction_sample.most_similar_instructions.values())
    return max_similarity < 0.7
