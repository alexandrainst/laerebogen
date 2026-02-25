"""Push the dataset to the Hugging Face Hub.

Usage:
    python push_to_hub.py JSONL_DATA_PATH REPO_ID[::SUBSET] [--public]
"""

import logging
from pathlib import Path

import click
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("push_to_hub")


@click.command()
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, readable=True),
    help="The path to the JSONL dataset.",
)
@click.argument(
    "repo_id",
    type=str,
    help="The repository ID on the Hugging Face Hub. Include the subset by separating "
    "with double colons ('::'), e.g. 'username/repo::subset'.",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Push the dataset to the Hugging Face Hub as a public dataset.",
)
def main(data_path: Path, repo_id: str, public: bool) -> None:
    """Push the dataset to the Hugging Face Hub."""
    logger.info(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path.as_posix(), split="train")
    assert isinstance(dataset, Dataset)
    logger.info(f"Dataset loaded with {len(dataset):,} examples.")

    if "most_similar_instructions" in dataset.column_names:
        dataset = dataset.remove_columns(column_names="most_similar_instructions")
    assert isinstance(dataset, Dataset), (
        f"Expected dataset to be of type 'Dataset', but got {type(dataset)}."
    )

    logger.info(f"Pushing dataset to Hugging Face Hub at {repo_id!r}...")
    if "::" in repo_id:
        repo_id, subset = repo_id.split("::")
    else:
        pass
    dataset.push_to_hub(repo_id, config_name="default", private=not public)
    logger.info(f"Dataset pushed to {repo_id!r} successfully.")


if __name__ == "__main__":
    main()
