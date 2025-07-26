"""Functions related to generating text using Ollama."""

import logging
import sys
import time

import ollama
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def generate_text_with_ollama(
    prompts: list[str], model_id: str
) -> list[ollama.GenerateResponse]:
    """Decode with an Ollama LLM.

    Args:
        prompts:
            A list of strings to complete.
        model_id:
            The Ollama model ID to use for generation.

    Returns:
        A list of ollama.GenerateResponse objects, each containing the generated text
        for the corresponding prompt.
    """
    completions: list[ollama.GenerateResponse] = []
    while True:
        try:
            # TODO: Do this asyncronously, to actually use the batching for something
            batch_completions: list[ollama.GenerateResponse] = []
            for prompt in prompts:
                completion_batch = ollama.generate(
                    model=model_id,
                    prompt=prompt,
                    options=ollama.Options(
                        num_batch=1,
                        num_ctx=3072,
                        temperature=0.2,
                        top_p=1.0,
                        stop=["\n20", "20."],
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                    ),
                )
                batch_completions.append(completion_batch)
            completions.extend(batch_completions)
            break
        except KeyboardInterrupt:
            logger.info("Stopping generation due to keyboard interrupt.")
            sys.exit(0)
        except ollama.ResponseError as e:
            logger.warning(f"Ollama ResponseError: {e}. Retrying...")
            time.sleep(2.0)
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}. Retrying...")
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
