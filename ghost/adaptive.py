"""Adaptive white-box attacks against GHOST shadow-token mappings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import model_output_name
from .metrics import compute_text_similarity
from .pipeline import _resolve_torch_dtype, resolve_device
from .search import load_shadow_map


AdaptiveStrategy = Literal["sample", "max_similarity", "median_similarity", "mean_embedding"]


@dataclass(frozen=True)
class AdaptiveAttackConfig:
    dataset: str
    model_name: str
    selected_data_path: str
    shadow_map_path: str
    output_path: str
    strategy: AdaptiveStrategy = "max_similarity"
    device: str = "auto"
    seed: int = 42
    hf_token: str | None = None
    load_in_4bit: bool = False
    torch_dtype: str = "auto"
    lowercase_metrics: bool = False


def _inverse_shadow_map(shadow_map: list[list[int]]) -> dict[int, list[int]]:
    inverse: dict[int, list[int]] = {}
    for original_id, candidates in enumerate(shadow_map):
        for shadow_id in candidates:
            inverse.setdefault(int(shadow_id), []).append(int(original_id))
    return inverse


def _load_model_and_tokenizer(config: AdaptiveAttackConfig, device: torch.device) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {}
    token = config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, **kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_torch_dtype(config.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if config.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit bitsandbytes loading requires a CUDA device.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = AutoModel.from_pretrained(config.model_name, **kwargs)
    if not config.load_in_4bit:
        model.to(device)
    model.eval()
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def _choose_candidate(
    shadow_id: int,
    candidates: list[int],
    strategy: AdaptiveStrategy,
    embeddings: torch.Tensor,
    rng: random.Random,
) -> int:
    if not candidates:
        return shadow_id
    shadow = embeddings[shadow_id].unsqueeze(0)
    candidate_embeddings = embeddings[candidates]
    similarities = F.cosine_similarity(candidate_embeddings, shadow, dim=1)

    if strategy == "max_similarity":
        return candidates[int(torch.argmax(similarities).item())]
    if strategy == "median_similarity":
        order = torch.argsort(similarities)
        return candidates[int(order[len(order) // 2].item())]
    if strategy == "sample":
        probs = torch.softmax(similarities, dim=0).detach().cpu().tolist()
        return rng.choices(candidates, weights=probs, k=1)[0]
    if strategy == "mean_embedding":
        mean = candidate_embeddings.mean(dim=0, keepdim=True)
        mean_sims = F.cosine_similarity(candidate_embeddings, mean, dim=1)
        return candidates[int(torch.argmax(mean_sims).item())]
    raise ValueError(f"Unsupported adaptive strategy {strategy!r}.")


def run_adaptive_attack(config: AdaptiveAttackConfig) -> Path:
    """Recover original-token guesses from fully recovered shadow text."""

    device = resolve_device(config.device)
    rng = random.Random(config.seed)
    model, tokenizer = _load_model_and_tokenizer(config, device)
    embeddings = F.normalize(model.get_input_embeddings().weight.detach().to(device=device, dtype=torch.float32), dim=1)
    shadow_map, metadata = load_shadow_map(config.shadow_map_path)
    inverse = _inverse_shadow_map(shadow_map)

    with Path(config.selected_data_path).open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    shadow_texts = selected["transformed_sentences"]
    references = selected["original_sentences"]
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    recovered: list[str] = []
    for text in shadow_texts:
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        recovered_ids: list[int] = []
        for token_id in token_ids:
            if token_id in special_ids:
                recovered_ids.append(token_id)
                continue
            candidates = inverse.get(int(token_id), [])
            recovered_ids.append(_choose_candidate(int(token_id), candidates, config.strategy, embeddings, rng))
        recovered.append(tokenizer.decode(recovered_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

    similarity = compute_text_similarity(recovered, references, lowercase=config.lowercase_metrics)
    output = {
        "config": asdict(config),
        "model_dir": model_output_name(config.model_name),
        "shadow_metadata": metadata,
        "recovered_sentences": recovered,
        "original_sentences": references,
        "similarity": similarity.as_dict(),
    }
    path = Path(config.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return path
