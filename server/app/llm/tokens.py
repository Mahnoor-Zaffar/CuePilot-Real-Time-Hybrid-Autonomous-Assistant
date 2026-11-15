"""Token estimate calibration.

The mid-stream budget cutoff has to estimate tokens from characters: the
provider has not reported usage yet. A fixed `chars / 4` is a reasonable prior
for English prose but is wrong for code, JSON, and non-English text. This keeps
a per-process EWMA of observed tokens-per-character from provider-reported
usage and uses it for the next estimate.

The estimate is still an estimate. It is a bound on how much text to stream,
not a replacement for provider accounting: the terminal `usage` frame always
prefers the provider's numbers when they arrive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The prior: four characters per token. Ratios are clamped so one pathological
# turn cannot poison the estimate for every later turn.
DEFAULT_TOKENS_PER_CHAR = 0.25
MIN_TOKENS_PER_CHAR = 0.02
MAX_TOKENS_PER_CHAR = 2.0


@dataclass
class TokenCalibration:
    """Bounded EWMA of tokens-per-character, with error accounting."""

    alpha: float = 0.2
    prompt_ratio: float = DEFAULT_TOKENS_PER_CHAR
    completion_ratio: float = DEFAULT_TOKENS_PER_CHAR
    prompt_samples: int = 0
    completion_samples: int = 0
    samples: int = 0
    estimated_tokens: float = 0.0
    actual_tokens: float = 0.0
    abs_error_tokens: float = 0.0
    signed_error_tokens: float = 0.0

    def __post_init__(self) -> None:
        self.alpha = min(1.0, max(0.0, float(self.alpha)))
        self.prompt_ratio = _clamp(self.prompt_ratio)
        self.completion_ratio = _clamp(self.completion_ratio)

    def estimate_prompt(self, text: str) -> int:
        return max(0, math.ceil(len(text) * self.prompt_ratio))

    def estimate_completion(self, chars: int) -> int:
        return max(0, math.ceil(max(0, chars) * self.completion_ratio))

    def observe(
        self,
        *,
        prompt_chars: int,
        prompt_tokens: int,
        completion_chars: int,
        completion_tokens: int,
        estimated_tokens: int | None = None,
        actual_tokens: int | None = None,
    ) -> None:
        """Update ratios from provider-reported usage and record error.

        Ratios update only from a positive observation. Error accounting is
        separate: it compares the estimate that was actually used for this turn
        with the provider's total, so an improving ratio is visible as a falling
        mean absolute error.
        """
        if prompt_chars > 0 and prompt_tokens > 0:
            observed = prompt_tokens / prompt_chars
            self.prompt_ratio = _clamp(
                (1 - self.alpha) * self.prompt_ratio + self.alpha * observed
            )
            self.prompt_samples += 1
        if completion_chars > 0 and completion_tokens > 0:
            observed = completion_tokens / completion_chars
            self.completion_ratio = _clamp(
                (1 - self.alpha) * self.completion_ratio + self.alpha * observed
            )
            self.completion_samples += 1
        if estimated_tokens is not None and actual_tokens is not None and actual_tokens > 0:
            error = float(estimated_tokens - actual_tokens)
            self.samples += 1
            self.estimated_tokens += float(estimated_tokens)
            self.actual_tokens += float(actual_tokens)
            self.abs_error_tokens += abs(error)
            self.signed_error_tokens += error

    @property
    def mean_abs_error_ratio(self) -> float | None:
        if not self.samples or self.actual_tokens <= 0:
            return None
        return round(self.abs_error_tokens / self.actual_tokens, 4)

    @property
    def bias_ratio(self) -> float | None:
        if not self.samples or self.actual_tokens <= 0:
            return None
        return round(self.signed_error_tokens / self.actual_tokens, 4)

    def stats(self) -> dict[str, object]:
        return {
            "prompt_ratio": round(self.prompt_ratio, 4),
            "completion_ratio": round(self.completion_ratio, 4),
            "prompt_samples": self.prompt_samples,
            "completion_samples": self.completion_samples,
            "samples": self.samples,
            "mean_abs_error_ratio": self.mean_abs_error_ratio,
            "bias_ratio": self.bias_ratio,
        }


def _clamp(value: float) -> float:
    return min(MAX_TOKENS_PER_CHAR, max(MIN_TOKENS_PER_CHAR, float(value)))
