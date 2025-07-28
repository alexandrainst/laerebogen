"""Push the dataset to the Hugging Face Hub.

Usage:
    python push_to_hub.py \
        [--data-path <data_path>] \
        [--repo_id <repo_id>] \
        [--private]
"""

import logging

import click
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("build_seed_task_jsonl")


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.jsonl",
    show_default=True,
    help="Path to the dataset stored as a JSONL file.",
)
@click.option(
    "--repo_id",
    type=str,
    default="alexandrainst/laerebogen",
    show_default=True,
    help="The repository ID on the Hugging Face Hub where the dataset will be pushed.",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Whether to create a private repository on the Hugging Face Hub.",
)
def main(
    data_path: str, repo_id: str = "alexandrainst/laerebogen", private: bool = False
) -> None:
    """Push the dataset to the Hugging Face Hub.

    Args:
        data_path:
            Path to the dataset stored as a JSONL file.
        repo_id:
            The repository ID on the Hugging Face Hub where the dataset will be pushed.
        private:
            Whether to create a private repository on the Hugging Face Hub.
    """
    logger.info(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Dataset loaded with {len(dataset):,} examples.")

    logger.info(f"Pushing dataset to Hugging Face Hub at {repo_id!r}...")
    dataset.push_to_hub(repo_id, private=private)
    logger.info(f"Dataset pushed to {repo_id!r} successfully.")


if __name__ == "__main__":
    main()
