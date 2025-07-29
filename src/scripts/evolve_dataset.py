"""Evolve the generated instruction-following dataset.

Usage:
    python evolve_dataset.py \
        [--dataset-path <dataset_path>] \
        [--model <model>] \
        [--num-evolutions <num_evolutions>] \
        [--num-cpus <num_cpus>] \
        [--verbose]
"""

import logging
import multiprocessing as mp
from pathlib import Path

import click
from tqdm.auto import tqdm

from laerebogen.data_models import InstructionSample
from laerebogen.evolving import evolve_instructions


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.corrected.jsonl",
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
    "--num-evolutions",
    type=int,
    default=4,
    show_default=True,
    help="Number of times to evolve the dataset. The resulting dataset will be "
    "1 + num_evolutions times larger than the original dataset.",
)
@click.option(
    "--num-cpus",
    type=int,
    default=2,
    show_default=True,
    help="Number of CPU cores to use for parallel processing. Set to -1 to use all "
    "available cores.",
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
    model: str,
    num_evolutions: int,
    num_cpus: int,
    verbose: bool,
) -> None:
    """Evolve the instruction-following dataset.

    Args:
        dataset_path:
            Path to the dataset file.
        model:
            Model ID of the instruction-tuned large language model to use for evolution.
        num_evolutions:
            Number of times to evolve the dataset.
        num_cpus:
            Number of CPU cores to use for parallel processing.
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

    # Load the dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path!r}")
    instructions = [
        InstructionSample.from_json(line.strip())
        for line in Path(dataset_path).read_text().splitlines()
        if line.strip()
    ]

    # Evolve the dataset
    pbar = (
        tqdm(iterable=range(num_evolutions), desc="Evolving dataset", unit="evolution")
        if num_evolutions > 1
        else range(num_evolutions)
    )
    for iteration in pbar:
        instructions = evolve_instructions(
            instructions=instructions,
            model_id=model,
            num_cpus=mp.cpu_count() if num_cpus == -1 else num_cpus,
        )
        with dataset_path.with_suffix(f".evolved_{iteration + 1}.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for instruction in instructions:
                f.write(instruction.json() + "\n")


if __name__ == "__main__":
    evolve()
