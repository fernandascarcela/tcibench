"""Client for OpenAI-compatible LLM endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible language model."""

    model: str
    base_url: str
    max_tokens: int
    api_key: str | None = None
    temperature: float = 0.7
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Relevant information returned by one generation request."""

    records: list[dict[str, Any]]
    elapsed_seconds: float
    requested_seed: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class LLMClient:
    """Send structured requests to an OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        config: LLMConfig,
    ) -> None:
        self.config = config

        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "local-api",
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        seed: int,
    ) -> LLMResponse:
        """
        Generate structured synthetic records.

        The requested seed is sent to the endpoint but is not assumed to
        guarantee deterministic generation.
        """

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            seed=seed,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "synthetic_records",
                    "strict": True,
                    "schema": schema,
                },
            },
            **self.config.generation_kwargs,
        )

        elapsed = time.perf_counter() - start

        choice = response.choices[0]

        if choice.finish_reason == "length":
            raise RuntimeError(
                "The LLM response was truncated because the token limit "
                "was reached. Increase max_tokens or request fewer records "
                "per generation call."
            )

        content = choice.message.content

        if not content:
            raise ValueError(
                "The LLM returned an empty response."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The LLM returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "The LLM response must be a JSON object."
            )

        records: list[dict[str, Any]] = []

        for field_name in schema["required"]:
            group_records = payload.get(field_name)

            if not isinstance(group_records, list):
                raise ValueError(
                    f"The LLM response must contain a "
                    f"'{field_name}' list."
                )

            records.extend(group_records)

        usage = response.usage

        return LLMResponse(
            records=records,
            elapsed_seconds=elapsed,
            requested_seed=seed,
            input_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),
            output_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            ),
            finish_reason=choice.finish_reason,
        )