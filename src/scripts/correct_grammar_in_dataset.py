"""Correct grammatical mistakes in a generated instruction-following dataset.

Usage:
    python correct_grammar_in_dataset.py \
        [--dataset-path <dataset_path>] \
        [--prompt-path <prompt_path>] \
        [--model <model>] \
        [--verbose]
"""

import logging
import os
import re
import warnings
from pathlib import Path

import click

from laerebogen.correction import correct_grammar_in_instructions
from laerebogen.data_models import InstructionSample


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.jsonl",
    show_default=True,
    help="Path to the dataset file.",
)
@click.option(
    "--prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/grammar_correction_prompt.txt",
    show_default=True,
    help="Path to the grammar correction prompt file.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    show_default=True,
    help="Model ID of the instruction-tuned large language model to use for grammar "
    "correction.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32_768,
    show_default=True,
    help="Number of samples to process with the LLM at the same time.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose logging.",
)
def main(
    dataset_path: str | Path,
    prompt_path: str,
    model: str,
    batch_size: int,
    verbose: bool,
) -> None:
    """Correct grammatical mistakes in a generated instruction-following dataset.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("correct_grammar_in_dataset")

    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path!r}...")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path!r}")
    with dataset_path.open("r", encoding="utf-8") as f:
        instructions = [
            InstructionSample.model_validate_json(line.strip())
            for line in f
            if line.strip()
        ]

    # Set up the output path
    corrected_path = dataset_path.with_name(
        re.sub(r"\..+", "", dataset_path.stem) + ".grammar_corrected.jsonl"
    )
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_path.touch(exist_ok=True)

    # Correct the grammar in the instructions and save the corrected instructions
    for corrected_instruction_batch in correct_grammar_in_instructions(
        instructions=instructions,
        prompt_path=prompt_path,
        model_id=model,
        batch_size=batch_size,
    ):
        with corrected_path.open("a", encoding="utf-8") as f:
            for instruction in instructions:
                f.write(instruction.model_dump_json() + "\n")

    logger.info(
        f"Saved {len(instructions):,} corrected instructions to "
        f"{corrected_path.resolve()!r}"
    )


if __name__ == "__main__":
    main()
