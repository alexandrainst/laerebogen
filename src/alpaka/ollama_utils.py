"""Functions related to generating text using Ollama."""

import dataclasses
import logging
import math
import time
from collections.abc import Sequence

import ollama
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# TODO: Consider deleting this
@dataclasses.dataclass
class DecodingArguments(object):
    """Decoding arguments for text generation."""

    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Sequence[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: str | None = None
    logprobs: int | None = None
    echo: bool = False


def generate_text_with_ollama(
    prompts: list[str], model_name: str, batch_size: int
) -> list[ollama.GenerateResponse]:
    """Decode with an Ollama LLM.

    Args:
        prompts:
            A list of strings to complete.
        decoding_args:
            Decoding arguments.
        model_name:
            Model name.
        batch_size:
            Number of prompts to send in a single request. Only for non chat model.

    Returns:
        A list of ollama.GenerateResponse objects, each containing the generated text
        for the corresponding prompt.
    """
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions: list[ollama.GenerateResponse] = []
    for prompt_batch in tqdm(
        prompt_batches, desc="prompt_batches", total=len(prompt_batches)
    ):
        while True:
            try:
                batch_completions: list[ollama.GenerateResponse] = []
                for prompt_batch_i in prompt_batch:
                    completion_batch = ollama.generate(
                        model=model_name, prompt=prompt_batch_i
                    )
                    batch_completions.append(completion_batch)
                completions.extend(batch_completions)
                break
            except ollama.ResponseError as e:
                logger.warning(f"Ollama ResponseError: {e}. Retrying...")
            except Exception as e:
                logger.info(f"An unexpected error occurred: {e}. Retrying...")
            finally:
                time.sleep(2.0)

    return completions
