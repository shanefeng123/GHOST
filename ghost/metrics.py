"""Shared text similarity metrics for privacy and attack evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


@dataclass(frozen=True)
class TextSimilarity:
    rouge1: float
    rouge2: float
    rougeL: float
    meteor: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "meteor": self.meteor,
        }


def _normalise_texts(texts: Sequence[str], lowercase: bool) -> list[str]:
    output = [str(text).strip() for text in texts]
    if lowercase:
        output = [text.lower() for text in output]
    return output


def _tokenize_for_meteor(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError as exc:
        raise RuntimeError(
            "METEOR scoring requires NLTK tokenizer data. Run "
            "`uv run python -c \"import nltk; nltk.download('punkt'); "
            "nltk.download('wordnet'); nltk.download('omw-1.4')\"`."
        ) from exc


def compute_text_similarity(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    lowercase: bool = False,
) -> TextSimilarity:
    """Compute the ROUGE/METEOR metrics used throughout the paper."""

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        raise ValueError("At least one prediction/reference pair is required.")

    predictions = _normalise_texts(predictions, lowercase)
    references = _normalise_texts(references, lowercase)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge_scores = [scorer.score(reference, prediction) for prediction, reference in zip(predictions, references)]

    # NLTK's METEOR implementation expects tokenized input and may consult
    # WordNet for synonym matching, so the README asks users to download both
    # tokenizer and WordNet corpora once during setup.
    try:
        meteor_scores = [
            meteor_score([_tokenize_for_meteor(reference)], _tokenize_for_meteor(prediction))
            for prediction, reference in zip(predictions, references)
        ]
    except LookupError as exc:
        raise RuntimeError(
            "METEOR scoring requires NLTK WordNet data. Run "
            "`uv run python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('omw-1.4')\"`."
        ) from exc

    count = float(len(predictions))
    return TextSimilarity(
        rouge1=sum(score["rouge1"].fmeasure for score in rouge_scores) / count,
        rouge2=sum(score["rouge2"].fmeasure for score in rouge_scores) / count,
        rougeL=sum(score["rougeL"].fmeasure for score in rouge_scores) / count,
        meteor=sum(meteor_scores) / count,
    )


def defense_efficacy(similarity: TextSimilarity) -> dict[str, float]:
    """Convert attack recovery similarity into defense efficacy values."""

    return {metric: 1.0 - value for metric, value in similarity.as_dict().items()}
