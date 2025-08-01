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

import logging
import os
import warnings
from pathlib import Path

import click
from tqdm.auto import tqdm

from laerebogen.data_models import InstructionSample
from laerebogen.evolving import evolve_instructions
from laerebogen.vllm_utils import load_vllm_model


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.quality_corrected.jsonl",
    show_default=True,
    help="Path to the dataset file.",
)
@click.option(
    "--rewriter-prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/rewriter_prompt.txt",
    show_default=True,
    help="Path to the prompt file for rewriting instructions.",
)
@click.option(
    "--creator-prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/creator_prompt.txt",
    show_default=True,
    help="Path to the prompt file for creating new instructions.",
)
@click.option(
    "--model",
    type=str,
    default="google/gemma-3-27b-it",
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
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose logging.",
)
def evolve(
    dataset_path: str | Path,
    rewriter_prompt_path: str,
    creator_prompt_path: str,
    model: str,
    num_evolutions: int,
    verbose: bool,
) -> None:
    """Evolve the instruction-following dataset.

    Args:
        dataset_path:
            Path to the dataset file.
        rewriter_prompt_path:
            Path to the prompt file for rewriting instructions.
        creator_prompt_path:
            Path to the prompt file for creating new instructions.
        model:
            Model ID of the instruction-tuned large language model to use for evolution.
        num_evolutions:
            Number of times to evolve the dataset.
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
    logger = logging.getLogger("evolve_dataset")

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
            InstructionSample.from_json(line.strip()) for line in f if line.strip()
        ]

    # Load the model
    logger.info(f"Loading model {model!r} for evolving instructions...")
    vllm_model = load_vllm_model(model_id=model)

    # Evolve the dataset
    pbar = (
        tqdm(iterable=range(num_evolutions), desc="Evolving dataset", unit="evolution")
        if num_evolutions > 1
        else range(num_evolutions)
    )
    all_evolutions = [instructions]
    for iteration in pbar:
        evolved_instructions = evolve_instructions(
            instructions=instructions,
            model=vllm_model,
            rewriter_prompt_path=rewriter_prompt_path,
            creator_prompt_path=creator_prompt_path,
        )
        all_evolutions.append(evolved_instructions)
        evolution_path = dataset_path.with_name(
            dataset_path.stem + f".evolved_{iteration + 1}.jsonl"
        )
        with evolution_path.open("w", encoding="utf-8") as f:
            entire_dataset = [
                instruction for evolution in all_evolutions for instruction in evolution
            ]
            for instruction in entire_dataset:
                f.write(instruction.json() + "\n")
        logger.info(
            f"Saved {len(entire_dataset):,} evolved instructions for iteration "
            f"{iteration + 1} to {evolution_path!r}."
        )


if __name__ == "__main__":
    evolve()
