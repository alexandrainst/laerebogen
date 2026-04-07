"""Add follow-up questions to a dataset of conversations.

Usage:
    python add_follow_ups.py \
        [--dataset-path <dataset_path>] \
        [--prompt-path <prompt_path>] \
        [--model <model>] \
        [--num-follow-ups <num_follow_ups>] \
        [--batch-size <batch_size>] \
        [--verbose]
"""

import json
import logging
import os
import re
import warnings
from pathlib import Path

import click
from pydantic import ValidationError

from laerebogen.data_models import Conversation, InstructionSample
from laerebogen.following_up import add_follow_up_to_conversations


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    default="data/dataset.evolved.jsonl",
    show_default=True,
    help="Path to the dataset file.",
)
@click.option(
    "--prompt-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    default="data/follow_up_prompt.txt",
    show_default=True,
    help="Path to the grammar correction prompt file.",
)
@click.option(
    "--model",
    type=str,
    default="google/gemma-4-31B-it",
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
    prompt_path: Path,
    model: str,
    num_follow_ups: int,
    batch_size: int,
    verbose: bool,
) -> None:
    """Add follow-up questions to a dataset of conversations.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.
    """
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
    logger.info(f"Loading dataset from {dataset_path.as_posix()!r}...")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path.as_posix()!r}")
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_instructions = [json.loads(line.strip()) for line in f if line.strip()]
        for raw_instruction in raw_instructions:
            raw_instruction.pop("evolution", None)

        instructions: list[InstructionSample] = list()
        for raw_instruction in raw_instructions:
            try:
                instruction = InstructionSample.model_validate(raw_instruction)
                instructions.append(instruction)
            except ValidationError:
                pass

        conversations = [
            Conversation(
                messages=[
                    dict(role="user", content=instruction.instruction),
                    dict(role="assistant", content=instruction.output),
                ]
            )
            for instruction in instructions
        ]
        logger.info(f"Loaded {len(conversations):,} conversations.")

    # Set up the output path
    follow_up_path = dataset_path.with_name(
        re.sub(r"\..+", "", dataset_path.stem) + ".with_follow_ups.jsonl"
    )
    follow_up_path.parent.mkdir(parents=True, exist_ok=True)
    follow_up_path.touch(exist_ok=True)

    # Remove the samples that have already been corrected
    conversations_with_follow_ups: list[Conversation] = []
    if follow_up_path.exists():
        with follow_up_path.open() as f:
            conversations_with_follow_ups: list[Conversation] = list()
            for line in f:
                if not line.strip():
                    continue
                try:
                    conversation = Conversation.model_validate_json(line.strip())
                    conversations_with_follow_ups.append(conversation)
                except ValidationError:
                    pass
            logger.info(
                f"Found {len(conversations_with_follow_ups):,} conversations that "
                f"already have follow-ups in {follow_up_path.as_posix()!r}"
            )

    # If we're >99% done, we're done
    if len(conversations_with_follow_ups) > 0.99 * len(conversations):
        return

    for conversation_with_follow_ups in add_follow_up_to_conversations(
        conversations=conversations,
        already_followed_up=conversations_with_follow_ups,
        prompt_path=prompt_path,
        model_id=model,
        num_follow_ups=num_follow_ups,
        batch_size=batch_size,
    ):
        with follow_up_path.open("a", encoding="utf-8") as f:
            f.write(conversation_with_follow_ups.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
