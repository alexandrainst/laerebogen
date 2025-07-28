"""Generate the dataset.

Usage:
    python generate_dataset.py \
        [--output-dir <output_dir>] \
        [--prompt-path <prompt_path>] \
        [--seed-tasks-path <seed_tasks_path>] \
        [--num-instructions-to-generate <num_instructions>] \
        [--model <model>] \
        [--num-prompt-instructions <num_prompt_instructions>] \
        [--batch-size <batch_size>] \
        [--num-cpus <num_cpus>] \
        [--backend <backend>] \
        [--verbose]
"""

import logging
import multiprocessing as mp
import os
import warnings
from typing import Literal

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
    "--prompt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/prompt.txt",
    show_default=True,
    help="Path to the prompt file.",
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
    default=52_000,
    show_default=True,
    help="Number of instructions to generate.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    show_default=True,
    help="The model ID of the model to use for generation. Must be a base model, not "
    "a finetuned one. Defaults to 'llama3.1:70b-text-fp16' when using Ollama, and "
    "'meta-llama/Llama-3.1-70B' when using vLLM.",
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
    default=128,
    show_default=True,
    help="Number of requests to send to the model at once.",
)
@click.option(
    "--num-cpus",
    type=int,
    default=2,
    show_default=True,
    help="Number of CPUs to use for parallel processing. Set to -1 to use all "
    "available CPUs.",
)
@click.option(
    "--backend",
    type=click.Choice(choices=["ollama", "vllm"], case_sensitive=False),
    default="ollama",
    show_default=True,
    help="The generation backend to use. Can be either 'ollama' or 'vllm'. Note that "
    "vLLM requires a GPU.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging.")
def generate(
    output_dir: str,
    prompt_path: str,
    seed_tasks_path: str,
    num_instructions_to_generate: int,
    model: str | None,
    num_prompt_instructions: int,
    batch_size: int,
    num_cpus: int,
    backend: Literal["ollama", "vllm"],
    verbose: bool,
) -> None:
    """Generate the dataset.

    Args:
        output_dir:
            Directory to save the generated dataset.
        prompt_path:
            Path to the prompt file.
        seed_tasks_path:
            Path to the seed tasks file.
        num_instructions_to_generate:
            Number of instructions to generate.
        model:
            The model ID of the model to use for generation. Must be a base
            model, not a finetuned one.
        num_prompt_instructions:
            Number of instructions to use as prompts for each generated instruction.
        batch_size:
            Number of requests to send to the model at once.
        num_cpus:
            Number of CPUs to use for parallel processing.
        backend:
            The generation backend to use. Can be either 'ollama' or 'vllm'.
        verbose:
            Enable verbose logging.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    if model is None:
        match backend:
            case "ollama":
                model = "llama3:70b-text-fp16"
            case "vllm":
                model = "meta-llama/Meta-Llama-3-70B"

    generate_instruction_following_data(
        output_dir=output_dir,
        prompt_path=prompt_path,
        seed_tasks_path=seed_tasks_path,
        num_instructions_to_generate=num_instructions_to_generate,
        model_id=model,
        num_prompt_instructions=num_prompt_instructions,
        batch_size=batch_size,
        num_cpus=mp.cpu_count() if num_cpus == -1 else num_cpus,
        backend=backend,
    )


if __name__ == "__main__":
    generate()
