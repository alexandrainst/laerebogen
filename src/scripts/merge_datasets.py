"""Merge multiple finetuning datasets into one and push to the Hugging Face Hub.

Usage:
    python merge_datasets.py \
        --dataset <dataset> --dataset <dataset> ... --dataset <dataset> \
        --new-dataset <new_dataset> \
        [--public]
"""

import logging

import click
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("push_to_hub")


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    multiple=True,
    required=True,
    help="Hugging Face dataset ID to merge. Can be specified multiple times. Can "
    "specify a specific split as '<dataset>:<split>'. If no split is specified, "
    "we will use the 'train' split by default.",
)
@click.option(
    "--new-dataset",
    type=str,
    required=True,
    help="The repository ID on the Hugging Face Hub where the merged dataset will "
    "be pushed.",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Whether to create a public repository on the Hugging Face Hub.",
)
def main(dataset: list[str], new_dataset: str, public: bool) -> None:
    """Merge multiple finetuning datasets into one and push to the Hugging Face Hub."""
    # Ensure that there are at least two datasets
    if len(dataset) < 2:
        raise ValueError("You need to specify at least two datasets to merge.")

    # Check that the new dataset ID does not already exist
    api = HfApi()
    if api.repo_exists(repo_id=new_dataset):
        raise FileExistsError(
            f"The dataset {new_dataset!r} already exists on the Hugging Face Hub. "
            "Please select a different one."
        )

    # Rename `dataset` to the more informative `dataset_names`
    dataset_names = dataset
    del dataset

    # Load datasets
    logger.info("Loading datasets...")
    datasets: list[Dataset] = [
        load_dataset(
            dataset_name,
            split="train" if ":" not in dataset_name else dataset_name.split(":")[1],
        )
        for dataset_name in dataset_names
    ]

    # Check if all datasets have a `messages` column
    for name, ds in zip(dataset_names, datasets):
        if "messages" not in ds.column_names:
            raise ValueError(
                f"The {name!r} dataset does not have a 'messages' feature and is "
                "thus not eligible to be merged with the others. Please fix this."
            )

    # Only keep the `messages` column
    logger.info("Keeping only the 'messages' column in each dataset...")
    datasets = [
        ds.remove_columns(
            column_names=[col for col in ds.column_names if col != "messages"]
        )
        for ds in datasets
    ]

    # Merge datasets
    logger.info("Merging datasets...")
    merged_dataset = concatenate_datasets(dsets=datasets).shuffle(seed=4242)

    # Push merged dataset to Hugging Face Hub
    logger.info(f"Pushing merged dataset to Hugging Face Hub at {new_dataset!r}...")
    merged_dataset.push_to_hub(repo_id=new_dataset, private=not public)

    logger.info(f"All done! Access the new dataset at hf.co/datasets/{new_dataset}.")


if __name__ == "__main__":
    main()
