"""
benchmark/inference.py

Модуль для работы с OpenRouter через OpenAI SDK.

Функциональность:
- единый клиент для всех LLM вызовов
- retry-механизм
- логирование запросов и ответов
- поддержка любых моделей OpenRouter
"""

from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI


class LLMClient:
    """
    Универсальный клиент для работы с OpenRouter.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 5,
        retry_delay: int = 180,
        request_delay: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        api_key : str
            OpenRouter API key.

        base_url : str
            OpenRouter endpoint.

        max_retries : int
            Максимальное число попыток.

        retry_delay : int
            Задержка между попытками (сек).

        request_delay : int
            Задержка после успешного запроса.

        verbose : bool
            Выводить логи или нет.
        """

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.verbose = verbose

    def generate(
        self,
        model_name: str,
        prompt: str,
    ) -> str:
        """
        Отправляет запрос в LLM и возвращает ответ.

        Parameters
        ----------
        model_name : str
            OpenRouter model id.

        prompt : str
            Пользовательский промпт.


        Returns
        -------
        str
            Ответ модели.
        """

        for attempt in range(1, self.max_retries + 1):

            try:

                if self.verbose:
                    print("\n" + "=" * 100)
                    print(f"MODEL: {model_name}")
                    print(f"ATTEMPT: {attempt}/{self.max_retries}")
                    print("-" * 100)

                    preview = (
                        prompt[:1500]
                        if len(prompt) > 1500
                        else prompt
                    )

                    print(preview)

                    if len(prompt) > 1500:
                        print("\n...[PROMPT TRUNCATED]...")

                    print("=" * 100)

                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                response = (
                    completion
                    .choices[0]
                    .message
                    .content
                )

                if response is None:
                    raise ValueError(
                        "Model returned empty response."
                    )

                response = response.strip()

                if self.verbose:
                    print("\n" + "-" * 100)
                    print("MODEL RESPONSE:")
                    print(response)
                    print("-" * 100)

                if self.request_delay > 0:
                    time.sleep(self.request_delay)

                return response

            except Exception as exc:

                print("\n" + "#" * 100)
                print("REQUEST FAILED")
                print(f"MODEL: {model_name}")
                print(f"ATTEMPT: {attempt}/{self.max_retries}")
                print(f"ERROR TYPE: {type(exc).__name__}")
                print(f"ERROR MESSAGE: {exc}")

                if hasattr(exc, "response"):
                    try:
                        print("\nAPI RESPONSE:")
                        print(exc.response.text)
                    except Exception:
                        pass

                print("#" * 100)

                if attempt == self.max_retries:
                    raise

                print(
                    f"\nRetrying in "
                    f"{self.retry_delay} seconds..."
                )

                time.sleep(self.retry_delay)

        raise RuntimeError(
            "Unexpected retry loop termination."
        )

    def generate_judge_score(
        self,
        model_name: str,
        prompt: str,
    ) -> str:
        """
        Специализированный вызов judge-модели.

        Используется для:
        - Context Integration
        - Scaffold

        Возвращает сырой ответ модели.

        Parameters
        ----------
        model_name : str
            Judge model.

        prompt : str
            Judge prompt.

        Returns
        -------
        str
        """

        return self.generate(
            model_name=model_name,
            prompt=prompt,
        )

    def test_connection(
        self,
        model_name: str,
    ) -> bool:
        """
        Проверка работоспособности API.

        Parameters
        ----------
        model_name : str

        Returns
        -------
        bool
        """

        try:

            response = self.generate(
                model_name=model_name,
                prompt="Reply with: OK",
            )

            return len(response) > 0

        except Exception:
            return False