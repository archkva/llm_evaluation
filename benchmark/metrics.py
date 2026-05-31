from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import re


#####################################################################
# CONSTANTS
#####################################################################

CEFR_MAP = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3,
    "C1": 4,
    "C2": 5,
}

MAX_SUPPORT_CLASS = 2


#####################################################################
# HELPERS
#####################################################################


def safe_mean(values: Iterable[Optional[float]]) -> float:
    clean = []

    for v in values:
        if v is None:
            continue
        if pd.isna(v):
            continue
        clean.append(float(v))

    if len(clean) == 0:
        return 0.0

    return float(np.mean(clean))


def cefr_to_numeric(level: str) -> Optional[int]:
    if level is None:
        return None

    return CEFR_MAP.get(str(level).strip())


#####################################################################
# ADAPTIVITY
#####################################################################


def adaptivity_score(
    true_levels: list[int],
    predicted_levels: list[int],
) -> float:
    """
    Точность с допущением ±1 CEFR.
    """

    if len(true_levels) == 0:
        return 0.0

    correct = 0

    for t, p in zip(true_levels, predicted_levels):
        if abs(p - t) <= 1:
            correct += 1

    return float(correct / len(true_levels))


def adaptivity_from_dataframe(
    df: pd.DataFrame,
    target_cefr_col: str,
    predicted_cefr_col: str,
) -> float:

    y_true = []
    y_pred = []

    for _, row in df.iterrows():

        target = cefr_to_numeric(row[target_cefr_col])
        pred = row[predicted_cefr_col]

        if target is None:
            continue

        if pd.isna(pred):
            continue

        y_true.append(target)
        y_pred.append(int(pred))

    return adaptivity_score(y_true, y_pred)


#####################################################################
# CORRECTION
#####################################################################


def normalize_text(text: str) -> str:
    text = str(text)

    # апострофы
    text = text.replace("’", "'").replace("`", "'")

    # кавычки
    text = text.replace('"', "")
    text = text.replace("“", "")
    text = text.replace("”", "")

    # пробелы
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def exact_match(pred: str, ref: str) -> int:
    return int(
        normalize_text(pred) == normalize_text(ref)
    )


def correction_score(
    preds: Iterable[str],
    refs: Iterable[str],
) -> float:

    matches = [
        exact_match(p, r)
        for p, r in zip(preds, refs)
    ]

    if len(matches) == 0:
        return 0.0

    return float(np.mean(matches))


def correction_from_dataframe(
    df: pd.DataFrame,
    pred_col: str,
    ref_col: str,
) -> float:

    preds = (
        df[pred_col]
        .astype(str)
        .fillna("")
        .tolist()
    )

    refs = (
        df[ref_col]
        .astype(str)
        .fillna("")
        .tolist()
    )

    return correction_score(preds, refs)


#####################################################################
# CONTEXT (LLM-as-a-judge)
#####################################################################


def context_score(values: Iterable[Optional[int]]) -> float:
    return safe_mean(values)


def context_from_dataframe(
    df: pd.DataFrame,
    context_col: str,
) -> float:

    return context_score(df[context_col].tolist())


#####################################################################
# SCAFFOLD (LLM-as-a-judge)
#####################################################################


def scaffold_score(values: Iterable[int]) -> float:

    vals = [
        int(v)
        for v in values
        if not pd.isna(v)
    ]

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))


def scaffold_from_dataframe(
    df: pd.DataFrame,
    scaffold_col: str,
) -> float:

    return scaffold_score(df[scaffold_col].tolist())


#####################################################################
# SUPPORT (DistilBERT)
#####################################################################


def support_score(values: Iterable[int]) -> float:

    vals = [
        int(v)
        for v in values
        if not pd.isna(v)
    ]

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals) / MAX_SUPPORT_CLASS)


def support_from_dataframe(
    df: pd.DataFrame,
    support_col: str,
) -> float:

    return support_score(df[support_col].tolist())


#####################################################################
# TOTAL
#####################################################################


def total_score(
    adaptivity: float,
    correction: float,
    context: float,
    scaffold: float,
    support: float,
) -> float:

    return float(
        0.20 * adaptivity
        + 0.30 * correction
        + 0.15 * context
        + 0.20 * scaffold
        + 0.15 * support
    )


#####################################################################
# FULL AGGREGATION
#####################################################################


def compute_metrics(
    df: pd.DataFrame,
    target_cefr_col: str,
    predicted_cefr_col: str,
    support_col: str,
    context_col: str,
    scaffold_col: str,
    correction_pred_col: str,
    correction_ref_col: str,
) -> dict:

    adaptivity = adaptivity_from_dataframe(
        df,
        target_cefr_col,
        predicted_cefr_col,
    )

    correction = correction_from_dataframe(
        df,
        pred_col=correction_pred_col,
        ref_col=correction_ref_col,
    )

    context = context_from_dataframe(
        df,
        context_col,
    )

    scaffold = scaffold_from_dataframe(
        df,
        scaffold_col,
    )

    support = support_from_dataframe(
        df,
        support_col,
    )

    total = total_score(
        adaptivity,
        correction,
        context,
        scaffold,
        support,
    )

    return {
        "adaptivity": adaptivity,
        "correction": correction,
        "context": context,
        "scaffold": scaffold,
        "support": support,
        "total": total,
    }