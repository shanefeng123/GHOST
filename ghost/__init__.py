"""Reconstructed GHOST token obfuscation pipeline."""

from .data import TextRecord, load_text_classification_data, load_text_generation_data
from .adaptive import AdaptiveAttackConfig, run_adaptive_attack
from .attack_eval import (
    AttackEvaluationConfig,
    AttackSampleConfig,
    evaluate_attack_outputs,
    parse_attack_predictions,
    select_attack_samples,
    selected_data_path,
)
from .metrics import TextSimilarity, compute_text_similarity, defense_efficacy
from .pipeline import TransformConfig, run_transform
from .search import ShadowSearchConfig, ShadowTokenSearcher, load_shadow_map, save_shadow_map
from .select import HiddenStateSelector, SelectionConfig, SelectionResult
from .train import TrainConfig, run_training

__all__ = [
    "AdaptiveAttackConfig",
    "AttackEvaluationConfig",
    "AttackSampleConfig",
    "HiddenStateSelector",
    "SelectionConfig",
    "SelectionResult",
    "ShadowSearchConfig",
    "ShadowTokenSearcher",
    "TextSimilarity",
    "TextRecord",
    "TransformConfig",
    "TrainConfig",
    "compute_text_similarity",
    "defense_efficacy",
    "evaluate_attack_outputs",
    "load_shadow_map",
    "load_text_classification_data",
    "load_text_generation_data",
    "parse_attack_predictions",
    "run_adaptive_attack",
    "run_transform",
    "run_training",
    "save_shadow_map",
    "select_attack_samples",
    "selected_data_path",
]
