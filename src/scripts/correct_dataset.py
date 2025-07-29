"""Evolve the generated instruction-following dataset.

Usage:
    python correct_dataset.py \
        [--dataset-path <dataset_path>] \
        [--model <model>] \
        [--verbose]
"""

import logging
import os
import warnings
from pathlib import Path

import click

from laerebogen.correcting import correct_instructions
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
    "--model",
    type=str,
    default="google/gemma-3-27b-it",
    show_default=True,
    help="Model ID of the instruction-tuned large language model to use for evolution.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose logging.",
)
def evolve(dataset_path: str | Path, model: str, verbose: bool) -> None:
    """Evolve the instruction-following dataset.

    Args:
        dataset_path:
            Path to the dataset file.
        model:
            Model ID of the instruction-tuned large language model to use for evolution.
        verbose:
            Enable verbose logging.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Load the dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path!r}")
    with dataset_path.open("r", encoding="utf-8") as f:
        instructions = [
            InstructionSample.from_json(line.strip()) for line in f if line.strip()
        ]

    # Correct the dataset
    instructions = correct_instructions(instructions=instructions, model_id=model)
    with dataset_path.with_suffix(".corrected.jsonl").open("w", encoding="utf-8") as f:
        for instruction in instructions:
            f.write(instruction.json() + "\n")


if __name__ == "__main__":
    evolve()
