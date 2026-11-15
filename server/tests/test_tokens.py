from __future__ import annotations

from app.llm.tokens import (
    DEFAULT_TOKENS_PER_CHAR,
    MAX_TOKENS_PER_CHAR,
    MIN_TOKENS_PER_CHAR,
    TokenCalibration,
)


def test_calibration_starts_at_four_chars_per_token() -> None:
    calibration = TokenCalibration()
    assert calibration.prompt_ratio == DEFAULT_TOKENS_PER_CHAR
    assert calibration.completion_ratio == DEFAULT_TOKENS_PER_CHAR
    assert calibration.estimate_prompt("abcd") == 1
    assert calibration.estimate_completion(400) == 100


def test_calibration_adapts_to_observed_usage() -> None:
    calibration = TokenCalibration(alpha=1.0)
    calibration.observe(
        prompt_chars=1000,
        prompt_tokens=200,
        completion_chars=400,
        completion_tokens=200,
        estimated_tokens=300,
        actual_tokens=400,
    )

    assert calibration.prompt_ratio == 0.2
    assert calibration.completion_ratio == 0.5
    assert calibration.estimate_completion(400) == 200
    assert calibration.samples == 1
    assert calibration.mean_abs_error_ratio == 0.25
    assert calibration.bias_ratio == -0.25


def test_calibration_ignores_missing_observations_and_clamps_ratios() -> None:
    calibration = TokenCalibration(alpha=1.0)
    calibration.observe(
        prompt_chars=0,
        prompt_tokens=100,
        completion_chars=0,
        completion_tokens=100,
    )
    assert calibration.prompt_samples == 0
    assert calibration.completion_samples == 0
    assert calibration.samples == 0
    assert calibration.mean_abs_error_ratio is None

    calibration.observe(
        prompt_chars=1,
        prompt_tokens=10_000,
        completion_chars=1,
        completion_tokens=10_000,
    )
    assert calibration.prompt_ratio == MAX_TOKENS_PER_CHAR
    assert calibration.completion_ratio == MAX_TOKENS_PER_CHAR

    calibration.observe(
        prompt_chars=10_000,
        prompt_tokens=1,
        completion_chars=10_000,
        completion_tokens=1,
    )
    assert calibration.prompt_ratio == MIN_TOKENS_PER_CHAR
    assert calibration.completion_ratio == MIN_TOKENS_PER_CHAR
