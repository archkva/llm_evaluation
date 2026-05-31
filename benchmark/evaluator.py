"""
benchmark/evaluator.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

from .inference import LLMClient
from .prompts import (
    build_context_integration_prompt,
    build_scaffold_prompt,
)


class BenchmarkEvaluator:
    """
    Основной класс оценки ответов моделей.
    """

    CEFR_MAP = {
        "A1": 0,
        "A2": 1,
        "B1": 2,
        "B2": 3,
        "C1": 4,
        "C2": 5,
    }

    INV_CEFR_MAP = {
        0: "A1",
        1: "A2",
        2: "B1",
        3: "B2",
        4: "C1",
        5: "C2",
    }

    MAX_SUPPORT_CLASS = 2

    def __init__(
        self,
        api_key: str,
        judge_model: str,
        cefr_model_path: str | Path,
        support_model_path: str | Path,
    ) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.judge_model = judge_model

        self.llm = LLMClient(
            api_key=api_key,
            verbose=False,
        )

        self.cefr_tokenizer = (
            DistilBertTokenizer.from_pretrained(
                str(cefr_model_path)
            )
        )

        self.cefr_model = (
            DistilBertForSequenceClassification
            .from_pretrained(
                str(cefr_model_path)
            )
            .to(self.device)
        )

        self.cefr_model.eval()

        self.support_tokenizer = (
            DistilBertTokenizer.from_pretrained(
                str(support_model_path)
            )
        )

        self.support_model = (
            DistilBertForSequenceClassification
            .from_pretrained(
                str(support_model_path)
            )
            .to(self.device)
        )

        self.support_model.eval()

        print("CEFR model loaded")
        print("Support model loaded")

    #################################################################
    # CEFR
    #################################################################

    def predict_cefr(
        self,
        text: str,
    ) -> int:
        """
        Возвращает класс CEFR:
        0=A1
        1=A2
        2=B1
        3=B2
        4=C1
        5=C2
        """

        inputs = self.cefr_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():

            logits = (
                self.cefr_model(
                    **inputs
                ).logits
            )

        prediction = int(
            torch.argmax(
                logits,
                dim=1
            ).item()
        )

        return prediction

    def predict_cefr_label(
        self,
        text: str,
    ) -> str:

        cls = self.predict_cefr(text)

        return self.INV_CEFR_MAP.get(
            cls,
            "UNKNOWN"
        )

    #################################################################
    # SUPPORT
    #################################################################

    def predict_support(
        self,
        text: str,
    ) -> int:
        """
        Возвращает класс:
        0
        1
        2
        """

        inputs = self.support_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():

            logits = (
                self.support_model(
                    **inputs
                ).logits
            )

        prediction = int(
            torch.argmax(
                logits,
                dim=1
            ).item()
        )

        return prediction

    #################################################################
    # CONTEXT INTEGRATION
    #################################################################

    def evaluate_context(
        self,
        student_utterance: str,
        reference: str,
        response: str,
        error_type: str,
    ) -> Optional[int]:
        """
        Возвращает:
        None -> NA
        0
        1
        """

        prompt = build_context_integration_prompt(
            student_utterance=student_utterance,
            reference=reference,
            response=response,
            error_type=error_type,
        )

        raw = self.llm.generate_judge_score(
            model_name=self.judge_model,
            prompt=prompt,
        )

        result = raw.strip()

        if result == "NA":
            return None

        if result == "0":
            return 0

        if result == "1":
            return 1

        print(
            f"[WARNING] "
            f"Unexpected Context output: "
            f"'{result}'"
        )

        return None

    #################################################################
    # SCAFFOLD
    #################################################################

    def evaluate_scaffold(
        self,
        student_utterance: str,
        reference: str,
        response: str,
        error_type: str,
    ) -> int:
        """
        Возвращает:
        0
        1
        """

        prompt = build_scaffold_prompt(
            student_utterance=student_utterance,
            reference=reference,
            response=response,
            error_type=error_type,
        )

        raw = self.llm.generate_judge_score(
            model_name=self.judge_model,
            prompt=prompt,
        )

        result = raw.strip()

        if result == "1":
            return 1

        if result == "0":
            return 0

        print(
            f"[WARNING] "
            f"Unexpected Scaffold output: "
            f"'{result}'"
        )

        return 0

    #################################################################
    # CORRECTION
    #################################################################

    @staticmethod
    def exact_match_correction(
        prediction: str,
        reference: str,
    ) -> int:
        """
        Exact Match.

        1 = совпадение
        0 = несовпадение
        """

        pred = str(
            prediction
        ).strip().lower()

        ref = str(
            reference
        ).strip().lower()

        return int(pred == ref)

    #################################################################
    # HELPERS
    #################################################################

    @classmethod
    def cefr_to_numeric(
        cls,
        level: str,
    ) -> Optional[int]:

        if level is None:
            return None

        return cls.CEFR_MAP.get(
            str(level).strip()
        )

    @classmethod
    def numeric_to_cefr(
        cls,
        value: int,
    ) -> str:

        return cls.INV_CEFR_MAP.get(
            value,
            "UNKNOWN"
        )

    #################################################################
    # ROW EVALUATION
    #################################################################

    def evaluate_response(
        self,
        student_utterance: str,
        reference: str,
        response: str,
        error_type: str,
    ) -> dict:
        """
        Полная оценка одного ответа.

        Используется для Prompt1-3.
        """

        predicted_cefr = self.predict_cefr(
            response
        )

        support = self.predict_support(
            response
        )

        context = self.evaluate_context(
            student_utterance=student_utterance,
            reference=reference,
            response=response,
            error_type=error_type,
        )

        scaffold = self.evaluate_scaffold(
            student_utterance=student_utterance,
            reference=reference,
            response=response,
            error_type=error_type,
        )

        return {
            "predicted_cefr": predicted_cefr,
            "support": support,
            "context": context,
            "scaffold": scaffold,
        }
