"""Command line interface for attack/defense evaluation."""

from __future__ import annotations

from argparse import ArgumentParser

from .adaptive import AdaptiveAttackConfig, run_adaptive_attack
from .attack_eval import AttackEvaluationConfig, AttackSampleConfig, evaluate_attack_outputs, select_attack_samples


def build_parser() -> ArgumentParser:
    parser = ArgumentParser("Prepare and evaluate gradient-inversion attack outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="Create selected attack data from transformed data")
    select.add_argument("--data", required=True)
    select.add_argument("--model_name", required=True)
    select.add_argument("--transformed_data_dir", default="data")
    select.add_argument("--output_dir", default="attack_data")
    select.add_argument("--topk", type=int, default=70)
    select.add_argument("--beam_width", type=int, default=1)
    select.add_argument("--overlap", type=float, default=0.1)
    select.add_argument("--select_size", type=int, default=64)
    select.add_argument("--train_ratio", type=float, default=0.8)
    select.add_argument("--seed", type=int, default=42)

    score = subparsers.add_parser("score", help="Score recovered text from an external GIA implementation")
    score.add_argument("--selected_data_path", required=True)
    score.add_argument("--predictions_path", required=True)
    score.add_argument("--output_path", required=True)
    score.add_argument(
        "--output_format",
        default="prediction_blocks",
        choices=["prediction_blocks", "grab", "json", "jsonl", "plain"],
    )
    score.add_argument("--batch_size", type=int, default=1)
    score.add_argument("--lowercase", action="store_true")
    score.add_argument("--defense_efficacy", action="store_true")

    adaptive = subparsers.add_parser("adaptive", help="Run white-box adaptive reversal from shadow tokens")
    adaptive.add_argument("--data", required=True)
    adaptive.add_argument("--model_name", required=True)
    adaptive.add_argument("--selected_data_path", required=True)
    adaptive.add_argument("--shadow_map_path", required=True)
    adaptive.add_argument("--output_path", required=True)
    adaptive.add_argument(
        "--strategy",
        default="max_similarity",
        choices=["sample", "max_similarity", "median_similarity", "mean_embedding"],
    )
    adaptive.add_argument("--device", default="auto")
    adaptive.add_argument("--seed", type=int, default=42)
    adaptive.add_argument("--hf_token", default=None)
    adaptive.add_argument("--load_in_4bit", action="store_true")
    adaptive.add_argument("--torch_dtype", default="auto")
    adaptive.add_argument("--lowercase_metrics", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        path = select_attack_samples(
            AttackSampleConfig(
                dataset=args.data,
                model_name=args.model_name,
                transformed_data_dir=args.transformed_data_dir,
                output_dir=args.output_dir,
                top_k=args.topk,
                beam_width=args.beam_width,
                overlap_threshold=args.overlap,
                select_size=args.select_size,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
        )
    elif args.command == "score":
        path = evaluate_attack_outputs(
            AttackEvaluationConfig(
                selected_data_path=args.selected_data_path,
                predictions_path=args.predictions_path,
                output_path=args.output_path,
                output_format=args.output_format,
                batch_size=args.batch_size,
                lowercase=args.lowercase,
                report_defense_efficacy=args.defense_efficacy,
            )
        )
    elif args.command == "adaptive":
        path = run_adaptive_attack(
            AdaptiveAttackConfig(
                dataset=args.data,
                model_name=args.model_name,
                selected_data_path=args.selected_data_path,
                shadow_map_path=args.shadow_map_path,
                output_path=args.output_path,
                strategy=args.strategy,
                device=args.device,
                seed=args.seed,
                hf_token=args.hf_token,
                load_in_4bit=args.load_in_4bit,
                torch_dtype=args.torch_dtype,
                lowercase_metrics=args.lowercase_metrics,
            )
        )
    else:
        raise AssertionError(args.command)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
