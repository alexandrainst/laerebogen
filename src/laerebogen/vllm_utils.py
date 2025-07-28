"""Functions related to generating text using vLLM."""

import importlib.util
import logging
import typing as t

import torch

from .data_models import Response

if importlib.util.find_spec("vllm") is not None or t.TYPE_CHECKING:
    from vllm import LLM, SamplingParams


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
        stop=["\n20", "20."], temperature=0.2, max_tokens=16_384
    )
    request_outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
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
    return LLM(
        model=model_id,
        tokenizer=model_id,
        gpu_memory_utilization=0.9,
        max_model_len=16_384,
        seed=4242,
        distributed_executor_backend="ray" if torch.cuda.device_count() > 1 else "mp",
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        enable_prefix_caching=True,
    )
