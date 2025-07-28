"""Functions related to generating text using vLLM."""

import importlib.util
import logging
import os
import typing as t

from tqdm.auto import tqdm

from .constants import MAX_CONTEXT_LENGTH, STOP_TOKENS, TEMPERATURE
from .data_models import Response

if importlib.util.find_spec("torch") is not None or t.TYPE_CHECKING:
    import torch

if importlib.util.find_spec("vllm") is not None or t.TYPE_CHECKING:
    from vllm import LLM, SamplingParams

if importlib.util.find_spec("ray") is not None or t.TYPE_CHECKING:
    import ray


logger = logging.getLogger(__name__)


def generate_text_with_vllm(prompts: list[str], model: "LLM") -> list[Response]:
    """Decode with vLLM.

    Args:
        prompts:
            A list of strings to complete.
        model:
            The vLLM model.

    Returns:
        A list of responses.
    """
    sampling_params = SamplingParams(
        stop=STOP_TOKENS, temperature=TEMPERATURE, max_tokens=MAX_CONTEXT_LENGTH
    )
    request_outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=get_pbar_without_leave,
    )
    completions = [
        Response(
            completion=request_output.outputs[0].text,
            done_reason=request_output.outputs[0].finish_reason,
        )
        for request_output in request_outputs
    ]
    return completions


def load_vllm_model(model_id: str) -> "LLM":
    """Initialise a vLLM model.

    Args:
        model_id:
            The Hugging Face model ID.

    Returns:
        The loaded vLLM model.
    """
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.engine.llm_engine").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.transformers_utils.tokenizer").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.core.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.model_executor.weight_utils").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.platforms.__init__").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.worker").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.services").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.runtime_env.packaging").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.utils").setLevel(logging.CRITICAL)
    os.environ["LOG_LEVEL"] = "CRITICAL"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    if importlib.util.find_spec("ray") is not None:
        ray._private.worker._worker_logs_enabled = False

    return LLM(
        model=model_id,
        tokenizer=model_id,
        gpu_memory_utilization=0.9,
        max_model_len=MAX_CONTEXT_LENGTH,
        distributed_executor_backend="ray" if torch.cuda.device_count() > 1 else "mp",
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        enable_prefix_caching=True,
    )


def get_pbar_without_leave(*tqdm_args, **tqdm_kwargs) -> tqdm:
    """Get a progress bar for vLLM which disappears after completion.

    Args:
        *tqdm_args:
            Positional arguments to pass to tqdm.
        **tqdm_kwargs:
            Additional keyword arguments to pass to tqdm.

    Returns:
        A tqdm progress bar.
    """
    tqdm_kwargs.pop("leave", None)  # Remove the 'leave' key if it exists
    return tqdm(*tqdm_args, leave=False, **tqdm_kwargs)
