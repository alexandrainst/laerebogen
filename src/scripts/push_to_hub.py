"""Push the dataset to the Hugging Face Hub.

Usage:
    python push_to_hub.py [--data-path <data_path>] [--repo_id <repo_id>] [--public]
"""

import logging

import click
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("push_to_hub")


@click.command()
@click.option(
    "--data_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.with_follow_ups.jsonl",
    show_default=True,
    help="Path to the dataset stored as a JSONL file.",
)
@click.option(
    "--repo_id",
    type=str,
    default="danish-foundation-models/laerebogen",
    show_default=True,
    help="The repository ID on the Hugging Face Hub where the dataset will be pushed.",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Whether to create a public repository on the Hugging Face Hub.",
)
def main(data_path: str, repo_id: str, public: bool) -> None:
    """Push the dataset to the Hugging Face Hub."""
    logger.info(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    assert isinstance(dataset, Dataset)
    logger.info(f"Dataset loaded with {len(dataset):,} examples.")

    if "most_similar_instructions" in dataset.column_names:
        dataset = dataset.remove_columns(column_names="most_similar_instructions")
    assert isinstance(dataset, Dataset), (
        f"Expected dataset to be of type 'Dataset', but got {type(dataset)}."
    )

    logger.info(f"Pushing dataset to Hugging Face Hub at {repo_id!r}...")
    dataset.push_to_hub(repo_id, private=not public)
    logger.info(f"Dataset pushed to {repo_id!r} successfully.")


if __name__ == "__main__":
    main()
