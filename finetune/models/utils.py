from typing import Dict, Literal

from finetune.trainer import Trainer

import torch
from typing import List, Sequence, Tuple


SUPPORTED_MODELS: Dict[str, Dict[str, Trainer]] = {}


def sample_by_mask_multi(
    mask: torch.Tensor,
    tensors: Sequence[torch.Tensor],
    N: int,
    empty_index_fill: int = -1
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    assert len(tensors) > 0
    B, n = mask.shape
    for t in tensors:
        assert t.ndim == 3 and t.shape[0] == B and t.shape[1] == n, f't shape {t.shape} is not equal with mask {mask.shape}'

    mask_bool = (mask != 0)
    device_mask = mask.device

    # 预分配输出与索引
    outs: List[torch.Tensor] = [
        torch.zeros((B, N, t.size(-1)), dtype=t.dtype, device=t.device)
        for t in tensors
    ]
    idxs = torch.full((B, N), empty_index_fill, dtype=torch.long, device=device_mask)

    for b in range(B):
        valid = torch.nonzero(mask_bool[b], as_tuple=False).squeeze(1)  # [m] on mask.device
        m = valid.numel()

        if m == 0:
            # 保持 outs[b] 为 0，idxs[b] 为 empty_index_fill
            continue

        if m >= N:
            choice = valid[torch.randperm(m, device=device_mask)[:N]]  # 无放回
        else:
            choice = valid[torch.randint(0, m, (N,), device=device_mask)]  # 有放回

        idxs[b] = choice  # 记录在 mask 的设备上

        # 对每个 tensor，用各自设备的索引进行抽取
        for out_i, t in zip(outs, tensors):
            choice_dev = choice.to(t.device)  # 索引需与被索引张量同设备
            out_i[b] = t[b].index_select(dim=0, index=choice_dev)  # [N, c_i]

    return outs


def register(model_name: str, training_type: Literal["lora", "sft"], trainer_cls: Trainer):
    """Register a model and its associated functions for a specific training type.

    Args:
        model_name (str): Name of the model to register (e.g. "cogvideox-5b")
        training_type (Literal["lora", "sft"]): Type of training - either "lora" or "sft"
        trainer_cls (Trainer): Trainer class to register.
    """

    # Check if model_name and training_type exists in SUPPORTED_MODELS
    if model_name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[model_name] = {}
    else:
        if training_type in SUPPORTED_MODELS[model_name]:
            raise ValueError(f"Training type {training_type} already exists for model {model_name}")

    SUPPORTED_MODELS[model_name][training_type] = trainer_cls


def show_supported_models():
    """Print all currently supported models and their training types."""

    print("\nSupported Models:")
    print("================")

    for model_name, training_types in SUPPORTED_MODELS.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))
        for training_type in training_types:
            print(f"  • {training_type}")


def get_model_cls(model_type: str, training_type: Literal["lora", "sft"]) -> Trainer:
    """Get the trainer class for a specific model and training type."""
    if model_type not in SUPPORTED_MODELS:
        print(f"\nModel '{model_type}' is not supported.")
        print("\nSupported models are:")
        for supported_model in SUPPORTED_MODELS:
            print(f"  • {supported_model}")
        raise ValueError(f"Model '{model_type}' is not supported")

    if training_type not in SUPPORTED_MODELS[model_type]:
        print(f"\nTraining type '{training_type}' is not supported for model '{model_type}'.")
        print(f"\nSupported training types for '{model_type}' are:")
        for supported_type in SUPPORTED_MODELS[model_type]:
            print(f"  • {supported_type}")
        raise ValueError(f"Training type '{training_type}' is not supported for model '{model_type}'")

    return SUPPORTED_MODELS[model_type][training_type]
