"""Add follow-up questions to a dataset of conversations.

Usage:
    python add_follow_ups.py \
        [--dataset-path <dataset_path>] \
        [--prompt-path <prompt_path>] \
        [--model <model>] \
        [--num-follow-ups <num_follow_ups>] \
        [--verbose]
"""

import logging
import os
import re
import warnings
from pathlib import Path

import click
from tqdm.auto import tqdm

from laerebogen.data_models import Conversation
from laerebogen.following_up import add_follow_up_to_conversations
from laerebogen.vllm_utils import load_vllm_model


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/dataset.evolved.jsonl",
    show_default=True,
    help="Path to the dataset file.",
)
@click.option(
    "--prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/follow_up_prompt.txt",
    show_default=True,
    help="Path to the grammar correction prompt file.",
)
@click.option(
    "--model",
    type=str,
    default="google/gemma-3-27b-it",
    show_default=True,
    help="Model ID of the instruction-tuned large language model used to create the "
    "follow-up queries and answers.",
)
@click.option(
    "--num-follow-ups",
    type=int,
    default=3,
    show_default=True,
    help="Number of follow-up questions to generate for each conversation.",
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
    num_follow_ups: int,
    verbose: bool,
) -> None:
    """Add follow-up questions to a dataset of conversations."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("add_follow_ups")

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
        conversations = [
            Conversation.from_json(line.strip()) for line in f if line.strip()
        ]

    # Load the model
    logger.info(f"Loading model {model!r} for adding follow-ups to instructions...")
    vllm_model = load_vllm_model(model_id=model)

    # Add follow-ups to each conversation
    pbar = (
        tqdm(
            iterable=range(num_follow_ups),
            desc="Adding follow-ups to conversations",
            unit="dataset pass",
        )
        if num_follow_ups > 1
        else range(num_follow_ups)
    )
    for _ in pbar:
        conversations = add_follow_up_to_conversations(
            conversations=conversations, prompt_path=prompt_path, model=vllm_model
        )

    # Store the extended conversations
    conversation_path = dataset_path.with_name(
        re.sub(r"\..+", "", dataset_path.stem) + ".with_follow_ups.jsonl"
    )
    with conversation_path.open("w", encoding="utf-8") as f:
        for conversation in conversations:
            f.write(conversation.json() + "\n")
    logger.info(
        f"Saved {len(conversations):,} conversations with follow-ups to "
        f"{conversation_path.resolve()!r}"
    )


if __name__ == "__main__":
    main()
