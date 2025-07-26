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


def try_download_ollama_model(model_id: str) -> bool:
    """Try to download an Ollama model.

    Args:
        model_id:
            The Ollama model ID.

    Returns:
        Whether the model was downloaded successfully.

    Raises:
        ConnectionError:
            If Ollama is not running or the model cannot be downloaded.
        RuntimeError:
            If the model cannot be downloaded due to an unexpected error.
    """
    try:
        downloaded_ollama_models: list[str] = [
            model_obj.model
            for model_obj in ollama.list().models
            if model_obj.model is not None
        ]
    except ConnectionError:
        raise ConnectionError(
            "Ollama does not seem to be running, so we cannot evaluate the model "
            f"{model_id!r}. Please make sure that Ollama is running and try again."
        )

    if model_id not in downloaded_ollama_models:
        # Try fetching the model info
        try:
            response = ollama.pull(model=model_id, stream=True)
        except ollama.ResponseError as e:
            if "file does not exist" in str(e).lower():
                # Check if the model exists if we prepend "hf.co/"
                try:
                    ollama_model_id_with_prefix = f"hf.co/{model_id}"
                    model_id_with_prefix = (
                        f"{model_id.split('/')[0]}/{ollama_model_id_with_prefix}"
                    )
                    ollama.pull(model=ollama_model_id_with_prefix, stream=True)
                    logger.info(
                        f"The model {model_id!r} cannot be found on Ollama, but the "
                        f"model {model_id_with_prefix} *was* found, so try again with "
                        "that model ID."
                    )
                    return False
                except ollama.ResponseError as inner_e:
                    if "file does not exist" in str(inner_e).lower():
                        logger.error(
                            f"The model {model_id!r} cannot be found on Ollama."
                        )
                        return False
                    else:
                        raise RuntimeError(
                            f"Failed to download Ollama model {model_id}. "
                            f"The error message was: {inner_e}"
                        )
            else:
                raise RuntimeError(
                    f"Failed to download Ollama model {model_id}. "
                    f"The error message was: {e}"
                )

        # Download the model
        with tqdm(
            desc=f"Downloading {model_id}", unit_scale=True, unit="B", leave=False
        ) as pbar:
            for status in response:
                if status.total is not None:
                    pbar.total = status.total
                if status.completed is not None:
                    pbar.update(status.completed - pbar.n)
    return True
