"""Attack/defense evaluation helpers for GIA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Literal, Sequence

from .data import model_output_name
from .metrics import compute_text_similarity, defense_efficacy


OutputFormat = Literal["prediction_blocks", "grab", "json", "jsonl", "plain"]


@dataclass(frozen=True)
class AttackSampleConfig:
    dataset: str
    model_name: str
    transformed_data_dir: str = "data"
    output_dir: str = "attack_data"
    top_k: int = 70
    beam_width: int = 1
    overlap_threshold: float = 0.1
    select_size: int = 64
    train_ratio: float = 0.8
    seed: int = 42


@dataclass(frozen=True)
class AttackEvaluationConfig:
    selected_data_path: str
    predictions_path: str
    output_path: str
    output_format: OutputFormat = "prediction_blocks"
    batch_size: int = 1
    lowercase: bool = False
    report_defense_efficacy: bool = False


def _transformed_data_path(config: AttackSampleConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    filename = (
        f"{config.dataset}_top_{config.top_k}_beam_{config.beam_width}"
        f"_overlap_{config.overlap_threshold}_discrete_transformed_data.json"
    )
    return Path(config.transformed_data_dir) / model_dir / filename


def selected_data_path(config: AttackSampleConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    return Path(config.output_dir) / model_dir / f"{config.dataset}_selected_data.json"


def select_attack_samples(config: AttackSampleConfig) -> Path:
    """Create the 64-sample attack subset used by GIA evaluations."""

    with _transformed_data_path(config).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    original = list(payload["original_sentences"])
    transformed = list(payload["transformed_sentences"])
    labels = list(payload.get("labels", [0] * len(original)))
    if len(transformed) < len(original):
        raise ValueError(
            "Transformed data is incomplete; run ghost-transform first "
            f"({len(transformed)} transformed rows for {len(original)} originals)."
        )

    rng = random.Random(config.seed)
    indices = list(range(len(original)))
    rng.shuffle(indices)
    train_count = int(len(indices) * config.train_ratio)
    candidate_indices = indices[:train_count]
    if config.select_size > len(candidate_indices):
        raise ValueError(
            f"select_size={config.select_size} exceeds available train candidates "
            f"({len(candidate_indices)})."
        )
    selected = rng.sample(candidate_indices, config.select_size)

    output = {
        "metadata": asdict(config),
        "indices": selected,
        "original_sentences": [original[idx] for idx in selected],
        "transformed_sentences": [transformed[idx] for idx in selected],
        "labels": [labels[idx] for idx in selected],
    }
    path = selected_data_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return path


def _parse_prediction_blocks(lines: Sequence[str], batch_size: int) -> list[str]:
    predictions: list[str] = []
    for idx, line in enumerate(lines):
        if line.strip() == "Prediction:":
            for offset in range(batch_size):
                pos = idx + offset + 1
                if pos < len(lines):
                    predictions.append(lines[pos].strip())
    return predictions


def _parse_grab(lines: Sequence[str], batch_size: int) -> list[str]:
    predictions: list[str] = []
    for idx, line in enumerate(lines):
        if "solution is better" not in line:
            continue
        first = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if "solution:" in first:
            first = first.split("solution:", 1)[1].strip()
        next_pos = idx + 2
        if not first and idx + 2 < len(lines):
            first = lines[idx + 2].strip()
            next_pos = idx + 3
        predictions.append(first)
        for offset in range(1, batch_size):
            pos = next_pos + offset - 1
            if pos < len(lines):
                predictions.append(lines[pos].strip())
    return predictions


def _extract_prediction(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("prediction", "recovered", "text", "output", "sentence"):
            if key in payload:
                return str(payload[key])
    return str(payload)


def _parse_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [_extract_prediction(item) for item in payload]
    for key in ("predictions", "recovered_sentences", "model_out", "outputs"):
        if key in payload:
            return [_extract_prediction(item) for item in payload[key]]
    raise ValueError(f"Could not find predictions in {path}.")


def _parse_jsonl(path: Path) -> list[str]:
    predictions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, str):
                predictions.append(payload)
            else:
                for key in ("prediction", "recovered", "text", "output", "sentence"):
                    if key in payload:
                        predictions.append(str(payload[key]))
                        break
    return predictions


def _parse_plain(lines: Sequence[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def parse_attack_predictions(path: str | Path, output_format: OutputFormat, *, batch_size: int = 1) -> list[str]:
    path = Path(path)
    if output_format == "json":
        return _parse_json(path)
    if output_format == "jsonl":
        return _parse_jsonl(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    if output_format == "prediction_blocks":
        return _parse_prediction_blocks(lines, batch_size)
    if output_format == "grab":
        return _parse_grab(lines, batch_size)
    if output_format == "plain":
        return _parse_plain(lines)
    raise ValueError(f"Unsupported output format {output_format!r}.")


def _clean_prediction(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_attack_outputs(config: AttackEvaluationConfig) -> Path:
    """Score recovered attack text against the original selected samples."""

    selected_path = Path(config.selected_data_path)
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    references = [_clean_prediction(text) for text in selected["original_sentences"]]
    predictions = [
        _clean_prediction(text)
        for text in parse_attack_predictions(
            config.predictions_path,
            config.output_format,
            batch_size=config.batch_size,
        )
    ]
    raw_prediction_count = len(predictions)
    pair_count = min(len(references), len(predictions))
    if pair_count == 0:
        raise ValueError("No prediction/reference pairs available for evaluation.")
    references = references[:pair_count]
    predictions = predictions[:pair_count]

    similarity = compute_text_similarity(predictions, references, lowercase=config.lowercase)
    result: dict[str, Any] = {
        "config": asdict(config),
        "num_pairs": pair_count,
        "num_references": len(selected["original_sentences"]),
        "num_predictions": raw_prediction_count,
        "similarity": similarity.as_dict(),
    }
    if config.report_defense_efficacy:
        result["defense_efficacy"] = defense_efficacy(similarity)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path
