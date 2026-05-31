from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
from tqdm import tqdm

from benchmark.model_assets import ensure_model
from benchmark.prompts import (
    build_prompt_1,
    build_prompt_2,
    build_prompt_3,
    build_prompt_4,
)

from benchmark.inference import LLMClient
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.metrics import compute_metrics, exact_match


ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "benchmark" / "test_utterances"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RAW_OUTPUTS_PATH = RESULTS_DIR / "raw_outputs.csv"

REQUIRED_RAW_COLUMNS = (
    "model",
    "student_utterance",
    "cefr",
    "reference",
    "error_type",
    "response_prompt4",
)


PROMPT_BUILDERS: dict[str, Callable] = {
    "prompt1": build_prompt_1,
    "prompt2": build_prompt_2,
    "prompt3": build_prompt_3,
}


#####################################################################
# INPUT
#####################################################################

VALID_CEFR_LEVELS = frozenset({"A1", "A2", "B1", "B2", "C1", "C2"})
PROMPT_CHOICES = {
    "all": ["prompt1", "prompt2", "prompt3"],
    "1": ["prompt1"],
    "2": ["prompt2"],
    "3": ["prompt3"],
}
EVAL_PROMPT_NAMES = ("prompt1", "prompt2", "prompt3")

T = TypeVar("T")


def _read_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        raise SystemExit(0) from None


def _ask_validated(
    prompt: str,
    validate: Callable[[str], T],
) -> T:
    while True:
        value = _read_input(prompt)
        try:
            return validate(value)
        except ValueError as exc:
            print(f"Error: {exc}")


def load_dataset(level: str) -> pd.DataFrame:

    if level.lower() == "all":

        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

        dfs = []

        for lvl in levels:
            path = DATASET_DIR / f"{lvl}.csv"

            if not path.exists():
                raise FileNotFoundError(path)

            dfs.append(pd.read_csv(path, sep=";"))

        return pd.concat(dfs, ignore_index=True)

    path = DATASET_DIR / f"{level.upper()}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path, sep=";")


def ask_api_key() -> str:
    def validate(value: str) -> str:
        if not value:
            raise ValueError("API key cannot be empty.")
        return value

    return _ask_validated("\nOpenRouter API key:\n> ", validate)


def ask_models() -> list[str]:
    def validate(value: str) -> list[str]:
        models = [x.strip() for x in value.split(",") if x.strip()]
        if not models:
            raise ValueError("Enter at least one model (comma-separated).")
        return models

    return _ask_validated("\nModels (comma separated):\n> ", validate)


def ask_judge_model() -> str:
    def validate(value: str) -> str:
        if not value:
            raise ValueError("Judge model cannot be empty.")
        return value

    return _ask_validated("\nJudge model:\n> ", validate)


def ask_prompts() -> list[str]:
    def validate(value: str) -> list[str]:
        choice = value.lower()
        if choice not in PROMPT_CHOICES:
            raise ValueError("Prompt must be 1, 2, 3, or all.")
        return PROMPT_CHOICES[choice]

    return _ask_validated("\nPrompt (1/2/3/all):\n> ", validate)


def ask_level() -> str:
    def validate(value: str) -> str:
        normalized = value.strip().lower()
        if normalized == "all":
            return "all"
        level = normalized.upper()
        if level not in VALID_CEFR_LEVELS:
            raise ValueError(
                "CEFR level must be A1, A2, B1, B2, C1, C2, or all."
            )
        return level

    return _ask_validated(
        "\nCEFR level (A1/A2/B1/B2/C1/C2/all):\n> ",
        validate,
    )


def ask_skip_generation() -> bool:
    def validate(value: str) -> bool:
        choice = value.strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        raise ValueError("Answer must be y/yes or n/no.")

    return _ask_validated(
        "\nSkip generation and use existing raw_outputs.csv? (y/n):\n> ",
        validate,
    )


def detect_prompts_from_columns(columns: pd.Index) -> list[str]:
    prompts = [
        name
        for name in EVAL_PROMPT_NAMES
        if f"response_{name}" in columns
    ]
    if not prompts:
        raise ValueError(
            "No evaluation prompts found in saved outputs. "
            "Expected at least one of: "
            + ", ".join(f"response_{name}" for name in EVAL_PROMPT_NAMES)
        )
    return prompts


def load_raw_outputs(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved outputs found at {path}. Run generation first or choose n."
        )

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )

    prompts = detect_prompts_from_columns(df.columns)
    return df, prompts


#####################################################################
# GENERATION
#####################################################################


def run_generation(
    df: pd.DataFrame,
    models: list[str],
    prompts: list[str],
    llm: LLMClient,
) -> pd.DataFrame:

    rows = []
    api_calls_per_row = len(prompts) + 1  # prompt4 is always generated
    total = len(df) * len(models) * api_calls_per_row

    progress = tqdm(
        total=total,
        desc="Generating responses",
        unit="req",
        dynamic_ncols=True,
    )

    for model_name in models:

        for _, row in df.iterrows():

            student_utterance = row["Student utterance"]
            cefr = row["CEFR"]
            reference = row["Corrected version"]
            error_type = row["Error Type"]

            result = {
                "model": model_name,
                "student_utterance": student_utterance,
                "cefr": cefr,
                "reference": reference,
                "error_type": error_type,
            }

            if "prompt1" in prompts:
                progress.set_postfix(model=model_name, prompt="1", refresh=False)
                result["response_prompt1"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_1(student_utterance, cefr),
                )
                progress.update(1)

            if "prompt2" in prompts:
                progress.set_postfix(model=model_name, prompt="2", refresh=False)
                result["response_prompt2"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_2(student_utterance, cefr),
                )
                progress.update(1)

            if "prompt3" in prompts:
                progress.set_postfix(model=model_name, prompt="3", refresh=False)
                result["response_prompt3"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_3(student_utterance, cefr),
                )
                progress.update(1)

            progress.set_postfix(model=model_name, prompt="4", refresh=False)
            result["response_prompt4"] = llm.generate(
                model_name=model_name,
                prompt=build_prompt_4(student_utterance),
            )
            progress.update(1)

            rows.append(result)

    progress.close()

    return pd.DataFrame(rows)


#####################################################################
# EVALUATION
#####################################################################


def run_evaluation(
    raw_df: pd.DataFrame,
    evaluator: BenchmarkEvaluator,
    prompts: list[str],
) -> pd.DataFrame:

    detailed_rows = []

    total = len(raw_df) * len(prompts)

    progress = tqdm(
        total=total,
        desc="Evaluating",
        unit="item",
        dynamic_ncols=True,
    )

    for _, row in raw_df.iterrows():

        for prompt_name in prompts:

            response_col = f"response_{prompt_name}"
            response = row[response_col]

            progress.set_postfix(
                model=row["model"],
                prompt=prompt_name,
                refresh=False,
            )

            scores = evaluator.evaluate_response(
                student_utterance=row["student_utterance"],
                reference=row["reference"],
                response=response,
                error_type=row["error_type"],
            )

            correction_match = exact_match(
                row["response_prompt4"],
                row["reference"],
            )

            detailed_rows.append(
                {
                    # META
                    "model": row["model"],
                    "prompt": prompt_name,
                    "cefr": row["cefr"],

                    # INPUT
                    "student_utterance": row["student_utterance"],
                    "reference": row["reference"],
                    "error_type": row["error_type"],

                    # RESPONSES
                    "response": response,
                    "response_prompt4": row["response_prompt4"],

                    # RAW MODEL OUTPUTS
                    "predicted_cefr": scores["predicted_cefr"],
                    "support": scores["support"],
                    "context": scores["context"],
                    "scaffold": scores["scaffold"],

                    # FINAL METRIC COMPONENT
                    "correction": correction_match,
                }
            )

            progress.update(1)

    progress.close()

    return pd.DataFrame(detailed_rows)


#####################################################################
# METRICS
#####################################################################


def aggregate_metrics(
    detailed_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = detailed_df.groupby(["model", "prompt"])

    for (model_name, prompt_name), sub in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Computing metrics",
        unit="group",
        dynamic_ncols=True,
    ):

        metrics = compute_metrics(
            df=sub,
            target_cefr_col="cefr",
            predicted_cefr_col="predicted_cefr",
            support_col="support",
            context_col="context",
            scaffold_col="scaffold",
            correction_pred_col="response_prompt4",
            correction_ref_col="reference",
        )

        rows.append(
            {
                "model": model_name,
                "prompt": prompt_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


#####################################################################
# LEADERBOARD
#####################################################################


def create_leaderboard(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:

    leaderboard = (
        evaluation_df
        .groupby("model")["total"]
        .mean()
        .reset_index()
    )

    leaderboard.columns = ["model", "total_avg"]

    leaderboard = (
        leaderboard
        .sort_values("total_avg", ascending=False)
        .reset_index(drop=True)
    )

    return leaderboard


#####################################################################
# MAIN
#####################################################################


def main():

    print("\n=== LLM Benchmark ===")

    skip_generation = ask_skip_generation()

    if skip_generation:
        api_key = ask_api_key()
        judge_model = ask_judge_model()

        print(f"\nLoading saved outputs from {RAW_OUTPUTS_PATH}...")
        raw_outputs, prompts = load_raw_outputs(RAW_OUTPUTS_PATH)
        print(f"Loaded rows: {len(raw_outputs)}")
        print(f"Detected prompts: {', '.join(prompts)}")
    else:
        api_key = ask_api_key()
        models = ask_models()
        judge_model = ask_judge_model()
        prompts = ask_prompts()
        level = ask_level()

        print("\nLoading dataset...")
        dataset = load_dataset(level)

        print(f"Loaded rows: {len(dataset)}")

        llm = LLMClient(
            api_key=api_key,
            request_delay=10,
            verbose=False,
        )

        print("\nGenerating model outputs...")

        raw_outputs = run_generation(
            df=dataset,
            models=models,
            prompts=prompts,
            llm=llm,
        )

        raw_outputs.to_csv(RAW_OUTPUTS_PATH, index=False)
        print(f"Saved: {RAW_OUTPUTS_PATH}")

    print("\nPreparing models...")
    model_paths = {}
    for model_name in tqdm(
        ["UPD_cefr_model_extended_dataset", "UPD_support_model"],
        desc="Models",
        unit="model",
        dynamic_ncols=True,
    ):
        model_paths[model_name] = ensure_model(model_name)

    evaluator = BenchmarkEvaluator(
        api_key=api_key,
        judge_model=judge_model,
        cefr_model_path=model_paths["UPD_cefr_model_extended_dataset"],
        support_model_path=model_paths["UPD_support_model"],
    )

    print("\nEvaluating...")

    detailed_df = run_evaluation(
        raw_outputs,
        evaluator,
        prompts,
    )

    detailed_path = RESULTS_DIR / "detailed_results.csv"
    detailed_df.to_csv(detailed_path, index=False)

    print(f"Saved: {detailed_path}")

    print("\nComputing metrics...")

    evaluation_df = aggregate_metrics(detailed_df)

    evaluation_path = RESULTS_DIR / "evaluation_results.csv"
    evaluation_df.to_csv(evaluation_path, index=False)

    print(f"Saved: {evaluation_path}")

    leaderboard_df = create_leaderboard(evaluation_df)

    leaderboard_path = RESULTS_DIR / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\n=== LEADERBOARD ===")
    print(leaderboard_df)

    print("\nResults saved to:")
    print(RAW_OUTPUTS_PATH)
    print(detailed_path)
    print(evaluation_path)
    print(leaderboard_path)


if __name__ == "__main__":
    main()