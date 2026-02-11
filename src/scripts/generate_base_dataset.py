"""Generate the initial dataset for instruction following.genera.

Usage:
    python generate_base_dataset.py \
        [--output-dir <output_dir>] \
        [--instruction-generation-prompt-path <prompt_path>] \
        [--output-generation-prompt-path <prompt_path>] \
        [--seed-tasks-path <seed_tasks_path>] \
        [--num-instructions-to-generate <num_instructions>] \
        [--model <model>] \
        [--num-prompt-instructions <num_prompt_instructions>] \
        [--batch-size <batch_size>] \
        [--verbose]
"""

import logging
import os
import warnings

import click

from laerebogen.generation import generate_instruction_following_data


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default="data",
    show_default=True,
    help="Directory to save the generated dataset.",
)
@click.option(
    "--instruction-generation-prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/instruction_generation_prompt.txt",
    show_default=True,
    help="Path to the instruction generation prompt file.",
)
@click.option(
    "--output-generation-prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/output_generation_prompt.txt",
    show_default=True,
    help="Path to the output generation prompt file.",
)
@click.option(
    "--seed-tasks-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/seed_tasks.jsonl",
    show_default=True,
    help="Path to the seed tasks file.",
)
@click.option(
    "--num-instructions-to-generate",
    type=int,
    default=100_000,
    show_default=True,
    help="Number of instructions to generate.",
)
@click.option(
    "--model",
    type=str,
    default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    show_default=True,
    help="The model ID of the model to use for generation.",
)
@click.option(
    "--num-prompt-instructions",
    type=int,
    default=3,
    show_default=True,
    help="Number of instructions to use as prompts for each generated instruction.",
)
@click.option(
    "--batch-size",
    type=int,
    default=512,
    show_default=True,
    help="Number of requests to send to the model at once.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging.")
def main(
    output_dir: str,
    instruction_generation_prompt_path: str,
    output_generation_prompt_path: str,
    seed_tasks_path: str,
    num_instructions_to_generate: int,
    model: str,
    num_prompt_instructions: int,
    batch_size: int,
    verbose: bool,
) -> None:
    """Generate the dataset."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    generate_instruction_following_data(
        output_dir=output_dir,
        instruction_generation_prompt_path=instruction_generation_prompt_path,
        output_generation_prompt_path=output_generation_prompt_path,
        seed_tasks_path=seed_tasks_path,
        num_instructions_to_generate=num_instructions_to_generate,
        model_id=model,
        num_prompt_instructions=num_prompt_instructions,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
