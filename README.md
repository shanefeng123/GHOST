# GHOST 

Official repo for the paper "Mitigating Gradient Inversion Risks in Language Models via Token Obfuscation" in Asia CCS'26.

## Installation

This repo uses `uv` for dependency management. Install `uv` first if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment and install all dependencies from the lockfile:

```bash
uv sync
```

The project pins the main experimental dependencies in `pyproject.toml`, including PyTorch, Transformers, Datasets, Evaluate, Accelerate, spaCy, NLTK, PEFT, Flair, Rouge, and scikit-learn. The spaCy English model is also installed through `uv`.

Download the NLTK data used by preprocessing and METEOR evaluation:

```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

For Linux/CUDA runs with Llama or Gemma 4-bit loading, `bitsandbytes` is installed automatically. It is skipped on macOS because official `bitsandbytes==0.45.0` wheels are not available for macOS arm64.

If you use gated Hugging Face models such as Llama or Gemma, set an access token:

```bash
export HF_TOKEN=your_huggingface_token
```

## Usage

Run the BERT-style classification pipeline with:

```bash
uv run python -m ghost.cli --data sst2 --model_name bert-base-uncased --num_of_samples 1000
```

Run the decoder/generation pipeline with GPT-2:

```bash
uv run ghost-transform --task generation --data enron --model_name gpt2 --num_of_samples 1000
```

For gated Llama/Gemma models, provide an HF token via `HF_TOKEN` or `--hf_token`.
Large decoder models can be loaded with 4-bit quantization on Linux/CUDA:

```bash
HF_TOKEN=... uv run ghost-transform \
  --task generation \
  --data enron \
  --model_name meta-llama/Llama-2-7b-hf \
  --device cuda:0 \
  --load_in_4bit \
  --add_eos_token
```

The pipeline has two explicit stages:

1. `ghost.search.ShadowTokenSearcher` builds per-token shadow candidate sets from
   embedding neighbors, then filters indirect similarity, direct mutual-neighbor
   similarity, and common-lemma similarity.
2. `ghost.select.HiddenStateSelector` performs coordinate beam search over those
   candidates to minimize hidden-state MSE against the original tokenized sentence.

Outputs and shadow-token caches are written under `data/<model-name>/` by default.

## Training and Evaluation

Training reads the transformation JSON produced by `ghost-transform`. Use
`--train_source transformed` to train on obfuscated data and evaluate on the
original test split, or `--train_source original` to train and evaluate on the
original data using the same split.

Classification utility evaluation reports loss, accuracy, and F1:

```bash
uv run ghost-train \
  --task classification \
  --data sst2 \
  --model_name bert-base-uncased \
  --train_source transformed \
  --num_of_samples 1000
```

Original-data classification baseline:

```bash
uv run ghost-train \
  --task classification \
  --data sst2 \
  --model_name bert-base-uncased \
  --train_source original \
  --num_of_samples 1000
```

Generative utility evaluation reports loss and perplexity:

```bash
uv run ghost-train \
  --task generation \
  --data enron \
  --model_name gpt2 \
  --train_source transformed \
  --num_of_samples 1000
```

For Llama/Gemma, the default behavior uses LoRA when the model name contains
`llama` or `gemma`. Use 4-bit loading on Linux/CUDA for larger models:

```bash
HF_TOKEN=... uv run ghost-train \
  --task generation \
  --data enron \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_source transformed \
  --device cuda:0 \
  --load_in_4bit
```

Gradient-noise or gradient-pruning baselines can be applied to original-data
training:

```bash
uv run ghost-train --task classification --data sst2 --model_name bert-base-uncased \
  --train_source original --gradient_noise 0.05

uv run ghost-train --task classification --data sst2 --model_name bert-base-uncased \
  --train_source original --gradient_prune 0.99
```

Metrics are written to `results/<model-name>/..._metrics.json`, and best
checkpoints are saved under `models/<model-name>/` unless `--no_save_model` is
provided.

## Attack and Defense Evaluation

The reconstructed repo keeps the GIA implementations decoupled from the GHOST
pipeline. Use `ghost-attack` to prepare the fixed attack subset, run an external
attack implementation on either the original or transformed text, then score the
recovered text against the original selected samples.

Create the 64-example attack subset used in the paper:

```bash
uv run ghost-attack select \
  --data sst2 \
  --model_name bert-base-uncased \
  --select_size 64
```

By default this reads the transformation output from `data/<model-name>/` and
writes `attack_data/<model-name>/<dataset>_selected_data.json`. The selected
file contains `original_sentences`, `transformed_sentences`, and `labels`, so
attack code can choose whether it is evaluating the undefended baseline or the
GHOST-transformed defense.

Score outputs from LAMP, TAG, DLG, or DAGER-style logs that contain
`Prediction:` blocks:

```bash
uv run ghost-attack score \
  --selected_data_path attack_data/bert-base-uncased/sst2_selected_data.json \
  --predictions_path path/to/attack_output.txt \
  --output_path attack_results/bert-base-uncased/sst2_lamp_transformed.json \
  --output_format prediction_blocks \
  --batch_size 1 \
  --defense_efficacy
```

For GRAB logs, use the GRAB parser:

```bash
uv run ghost-attack score \
  --selected_data_path attack_data/bert-base-uncased/sst2_selected_data.json \
  --predictions_path path/to/grab_output.txt \
  --output_path attack_results/bert-base-uncased/sst2_grab_transformed.json \
  --output_format grab
```

The scorer also accepts `json`, `jsonl`, and one-prediction-per-line `plain`
formats. Output JSON contains ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and optional
`1 - metric` defense efficacy values.

Run the adaptive white-box reversal baseline, where the attacker knows the
shadow-token map and tries to map recovered shadow text back to original tokens:

```bash
uv run ghost-attack adaptive \
  --data sst2 \
  --model_name bert-base-uncased \
  --selected_data_path attack_data/bert-base-uncased/sst2_selected_data.json \
  --shadow_map_path data/bert-base-uncased/shadow_top_70_overlap_0.1.json \
  --output_path attack_results/bert-base-uncased/sst2_adaptive_max_similarity.json \
  --strategy max_similarity
```

Adaptive strategies are `max_similarity`, `median_similarity`, `mean_embedding`,
and `sample`. For gated decoder models, pass `--hf_token` or set `HF_TOKEN`.
For larger Llama/Gemma adaptive runs on Linux/CUDA, add `--device cuda:0` and
`--load_in_4bit`.
