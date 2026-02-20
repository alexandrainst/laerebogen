"""Evolve the generated instruction-following dataset.

Usage:
    python evolve_dataset.py \
        [--dataset-path <dataset_path>] \
        [--correction-prompt-path <correction_prompt_path>] \
        [--creator-prompt-path <creator_prompt_path>] \
        [--model <model>] \
        [--num-evolutions <num_evolutions>] \
        [--verbose]
"""

import json
import logging
import os
import re
import warnings
from pathlib import Path

import click

from laerebogen.data_models import InstructionSample
from laerebogen.evolving import evolve_instructions


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    default="data/dataset.quality_corrected.jsonl",
    show_default=True,
    help="Path to the dataset file.",
)
@click.option(
    "--rewriter-prompt-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    default="data/rewriter_prompt.txt",
    show_default=True,
    help="Path to the prompt file for rewriting instructions.",
)
@click.option(
    "--creator-prompt-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    default="data/creator_prompt.txt",
    show_default=True,
    help="Path to the prompt file for creating new instructions.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    show_default=True,
    help="Model ID of the instruction-tuned large language model to use for evolution.",
)
@click.option(
    "--num-evolutions",
    type=int,
    default=4,
    show_default=True,
    help="Number of times to evolve the dataset. The resulting dataset will be "
    "1 + num_evolutions times larger than the original dataset.",
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
    dataset_path: Path,
    rewriter_prompt_path: Path,
    creator_prompt_path: Path,
    model: str,
    num_evolutions: int,
    batch_size: int,
    verbose: bool,
) -> None:
    """Evolve the instruction-following dataset.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("evolve_dataset")

    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path.as_posix()!r}...")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path.as_posix()!r}")
    with dataset_path.open("r", encoding="utf-8") as f:
        instructions = [
            InstructionSample.model_validate_json(line.strip())
            for line in f
            if line.strip()
        ]

    # Set up the output path
    evolution_path = dataset_path.with_name(
        re.sub(r"\..+", "", dataset_path.stem) + ".evolved.jsonl"
    )
    evolution_path.parent.mkdir(parents=True, exist_ok=True)
    evolution_path.touch(exist_ok=True)

    # Remove the samples that have already been corrected
    evolved_instructions: list[InstructionSample] = []
    if evolution_path.exists():
        with evolution_path.open() as f:
            evolved_instructions = [
                InstructionSample.model_validate_json(line.strip())
                for line in f
                if line.strip()
            ]
            logger.info(
                f"Found {len(evolved_instructions):,} instructions that have already "
                f"been evolved in {evolution_path.as_posix()!r}"
            )

    if len(instructions) == len(evolved_instructions):
        return

    for evolved_instruction, evolution in evolve_instructions(
        instructions=instructions,
        already_evolved=evolved_instructions,
        model_id=model,
        rewriter_prompt_path=rewriter_prompt_path,
        creator_prompt_path=creator_prompt_path,
        num_evolutions=num_evolutions,
        batch_size=batch_size,
    ):
        with evolution_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(evolved_instruction.model_dump() | dict(evolution=evolution))
                + "\n"
            )


if __name__ == "__main__":
    main()
