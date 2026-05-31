from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

import zipfile

from benchmark.prompts import (
    build_prompt_1,
    build_prompt_2,
    build_prompt_3,
    build_prompt_4,
)

from benchmark.inference import LLMClient
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.metrics import compute_metrics, exact_match


def ensure_model_extracted(model_dir: Path, zip_name: str) -> Path:
    """
    Проверяет наличие распакованной модели.
    Если папки нет, ищет ZIP-архив с именем zip_name и распаковывает его в model_dir.
    """
    model_path = MODEL_DIR / model_dir

    if model_path.exists() and model_path.is_dir():
        print(f"Model directory already exists: {model_path}")
        return model_path

    zip_path = MODEL_DIR / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Neither model directory {model_path} nor zip archive {zip_path} found."
        )

    print(f"Extracting {zip_path} to {model_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    # Проверка, что распаковалось именно в нужную папку
    if not model_path.exists():
        # Возможно, архив содержит корневую папку с другим именем – перемещаем
        extracted_items = list(MODEL_DIR.glob("*"))
        for item in extracted_items:
            if item.is_dir() and item.name.startswith("UPD_cefr"):
                item.rename(model_path)
                break

    print(f"Model extracted to {model_path}")
    return model_path

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "benchmark" / "test_utterances"
MODEL_DIR = ROOT_DIR / "benchmark" / "models"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


PROMPT_BUILDERS: dict[str, Callable] = {
    "prompt1": build_prompt_1,
    "prompt2": build_prompt_2,
    "prompt3": build_prompt_3,
}


#####################################################################
# INPUT
#####################################################################


def load_dataset(level: str) -> pd.DataFrame:

    if level.lower() == "all":

        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

        dfs = []

        for lvl in levels:
            path = DATASET_DIR / f"{lvl}.csv"

            if not path.exists():
                raise FileNotFoundError(path)

            dfs.append(pd.read_csv(path))

        return pd.concat(dfs, ignore_index=True)

    path = DATASET_DIR / f"{level.upper()}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def ask_models() -> list[str]:
    models = input("\nModels (comma separated):\n> ").strip()

    return [
        x.strip()
        for x in models.split(",")
        if x.strip()
    ]


def ask_prompts() -> list[str]:

    value = input("\nPrompt (1/2/3/all):\n> ").strip().lower()

    if value == "all":
        return ["prompt1", "prompt2", "prompt3"]

    if value == "1":
        return ["prompt1"]

    if value == "2":
        return ["prompt2"]

    if value == "3":
        return ["prompt3"]

    raise ValueError("Prompt must be 1,2,3 or all")


def ask_level() -> str:
    return input("\nCEFR level (A1/A2/B1/B2/C1/C2/all):\n> ").strip()


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

    total = len(df) * len(models)

    progress = tqdm(total=total, desc="Generating responses")

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
                result["response_prompt1"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_1(student_utterance, cefr),
                )

            if "prompt2" in prompts:
                result["response_prompt2"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_2(student_utterance, cefr),
                )

            if "prompt3" in prompts:
                result["response_prompt3"] = llm.generate(
                    model_name=model_name,
                    prompt=build_prompt_3(student_utterance, cefr),
                )

            # 🔴 ВСЕГДА считаем correction prompt
            result["response_prompt4"] = llm.generate(
                model_name=model_name,
                prompt=build_prompt_4(student_utterance),
            )

            rows.append(result)
            progress.update(1)

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

    progress = tqdm(total=total, desc="Evaluating")

    for _, row in raw_df.iterrows():

        for prompt_name in prompts:

            response_col = f"response_{prompt_name}"
            response = row[response_col]

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

    for (model_name, prompt_name), sub in grouped:

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

    api_key = input("\nOpenRouter API key:\n> ").strip()
    models = ask_models()
    judge_model = input("\nJudge model:\n> ").strip()
    prompts = ask_prompts()
    level = ask_level()

    print("\nLoading dataset...")
    dataset = load_dataset(level)

    print(f"Loaded rows: {len(dataset)}")

    llm = LLMClient(
        api_key=api_key,
        request_delay=10,
    )

    cefr_model_path = ensure_model_extracted(
    "UPD_cefr_model_extended_dataset",
    "UPD_cefr_model_extended_dataset.zip"
)
    support_model_path = ensure_model_extracted(
    "UPD_support_model",
    "UPD_support_model.zip"
)

    evaluator = BenchmarkEvaluator(
        api_key=api_key,
        judge_model=judge_model,
        cefr_model_path=cefr_model_path / "UPD_cefr_model_extended_dataset",
        support_model_path=support_model_path / "UPD_support_model",
    )

    print("\nGenerating model outputs...")

    raw_outputs = run_generation(
        df=dataset,
        models=models,
        prompts=prompts,
        llm=llm,
    )

    raw_path = RESULTS_DIR / "raw_outputs.csv"
    raw_outputs.to_csv(raw_path, index=False)

    print(f"Saved: {raw_path}")

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
    print(raw_path)
    print(detailed_path)
    print(evaluation_path)
    print(leaderboard_path)


if __name__ == "__main__":
    main()