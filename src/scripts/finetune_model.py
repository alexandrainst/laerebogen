"""Finetune a model on an instruction following dataset.

Usage:
    python finetune_model.py \
        --base-model <base_model> \
        --new-model <new_model> \
        [--val-samples <val_samples>] \
        [--load-in-4-bit] \
        [--max-seq-length <max_seq_length>] \
        [--lora-rank <lora_rank>] \
        [--learning-rate <learning_rate>] \
        [--weight-decay <weight_decay>] \
        [--neftune-noise-alpha <neftune_noise_alpha>] \
        [--per-device-batch-size <per_device_batch_size>] \
        [--total-batch-size <total_batch_size>] \
        [--num-epochs <num_epochs>] \
        [--warmup-ratio <warmup_ratio>] \
        [--logging-steps <logging_steps>] \
        [--eval-steps <eval_steps>] \
        [--dataloader-num-workers <dataloader_num_workers>] \
        [--use-wandb] \
        [--testing]
"""

import logging
import typing as t

import click

from laerebogen.finetuning import finetune_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finetune_model")


@click.command()
@click.option(
    "--base-model", type=str, required=True, help="The base model ID to finetune."
)
@click.option(
    "--new-model",
    type=str,
    required=True,
    help="The new model ID to save the finetuned model as.",
)
@click.option(
    "--val-samples",
    type=int,
    default=1024,
    show_default=True,
    help="Number of validation samples to use.",
)
@click.option(
    "--load-in-4-bit",
    is_flag=True,
    default=False,
    help="Load the model in 4-bit precision.",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=8192,
    show_default=True,
    help="Maximum sequence length for the model.",
)
@click.option(
    "--lora-rank",
    type=click.Choice([8, 16, 32, 64, 128]),
    default=32,
    show_default=True,
    help="Rank for LoRA finetuning.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=3e-4,
    show_default=True,
    help="Learning rate for the optimizer.",
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.01,
    show_default=True,
    help="Weight decay for the optimiser.",
)
@click.option(
    "--neftune-noise-alpha",
    type=int,
    default=5,
    show_default=True,
    help="Noise alpha for Neftune.",
)
@click.option(
    "--per-device-batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size per device.",
)
@click.option(
    "--total-batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Total batch size for training.",
)
@click.option(
    "--num-epochs",
    type=int,
    default=1,
    show_default=True,
    help="Number of epochs to train for.",
)
@click.option(
    "--warmup-ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Warmup ratio for the learning rate scheduler.",
)
@click.option(
    "--logging-steps",
    type=int,
    default=1,
    show_default=True,
    help="Number of steps between logging.",
)
@click.option(
    "--eval-steps",
    type=int,
    default=20,
    show_default=True,
    help="Number of steps between evaluations.",
)
@click.option(
    "--dataloader-num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of workers for the dataloader.",
)
@click.option(
    "--use-wandb", is_flag=True, default=False, help="Use Weights & Biases for logging."
)
@click.option(
    "--testing",
    is_flag=True,
    default=False,
    help="Run in testing mode with a small dataset.",
)
def main(
    base_model: str,
    new_model: str,
    val_samples: int,
    load_in_4bit: bool,
    max_seq_length: int,
    lora_rank: t.Literal[8, 16, 32, 64, 128],
    learning_rate: float,
    weight_decay: float,
    neftune_noise_alpha: int,
    per_device_batch_size: int,
    total_batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    dataloader_num_workers: int,
    use_wandb: bool,
    testing: bool,
) -> None:
    """Finetune a model on an instruction following dataset.

    Args:
        base_model:
            The base model ID to finetune.
        new_model:
            The new model ID to save the finetuned model as.
        val_samples:
            Number of validation samples to use.
        load_in_4bit:
            Load the model in 4-bit precision.
        max_seq_length:
            Maximum sequence length for the model.
        lora_rank:
            Rank for LoRA finetuning.
        learning_rate:
            Learning rate for the optimiser.
        weight_decay:
            Weight decay for the optimiser.
        neftune_noise_alpha:
            Noise alpha for Neftune.
        per_device_batch_size:
            Batch size per device.
        total_batch_size:
            Total batch size for training.
        num_epochs:
            Number of epochs to train for.
        warmup_ratio:
            Warmup ratio for the learning rate scheduler.
        logging_steps:
            Number of steps between logging.
        eval_steps:
            Number of steps between evaluations.
        dataloader_num_workers:
            Number of workers for the dataloader.
        use_wandb:
            Use Weights & Biases for logging.
        testing:
            Run in testing mode with a small dataset.
    """
    finetune_model(
        base_model_id=base_model,
        new_model_id=new_model,
        val_samples=val_samples,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        neftune_noise_alpha=neftune_noise_alpha,
        per_device_batch_size=per_device_batch_size,
        total_batch_size=total_batch_size,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        dataloader_num_workers=dataloader_num_workers,
        use_wandb=use_wandb,
        testing=testing,
    )


if __name__ == "__main__":
    main()
