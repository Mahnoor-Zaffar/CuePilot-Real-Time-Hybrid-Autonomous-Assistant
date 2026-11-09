import pytest
from app.ws import protocol
from app.ws.protocol import (
    AbortFrame,
    AudioFrame,
    FrameError,
    HelloFrame,
    PromptFrame,
    parse_client_frame,
)


def test_parse_hello_with_defaults() -> None:
    frame = parse_client_frame({"type": "hello"})
    assert isinstance(frame, HelloFrame)
    assert frame.protocol_version == protocol.PROTOCOL_VERSION
    assert frame.session_id is None


def test_parse_prompt_options() -> None:
    frame = parse_client_frame(
        {
            "type": "prompt",
            "id": "t1",
            "text": "hello",
            "options": {"top_k": 5, "mode": "bm25", "tools": False},
        }
    )
    assert isinstance(frame, PromptFrame)
    assert frame.options.top_k == 5
    assert frame.options.mode == "bm25"
    assert frame.options.tools is False


def test_parse_abort_allows_omitted_id() -> None:
    frame = parse_client_frame({"type": "abort"})
    assert isinstance(frame, AbortFrame)
    assert frame.id is None


def test_parse_audio_frame() -> None:
    frame = parse_client_frame({"type": "audio", "seq": 3, "pcm": "AAAA"})
    assert isinstance(frame, AudioFrame)
    assert frame.sample_rate == 16000


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "unknown"},
        {"type": "prompt", "id": "", "text": "x"},
        {"type": "prompt", "id": "t1"},
        {"type": "prompt", "id": "t1", "text": "x", "options": {"top_k": 99}},
        {"type": "audio", "seq": -1, "pcm": "AAAA"},
        {},
    ],
)
def test_invalid_frames_raise(payload: dict) -> None:
    with pytest.raises(FrameError):
        parse_client_frame(payload)


def test_server_frame_builders() -> None:
    ready = protocol.ready("s1", resumed=True, kb_version=4, model="m")
    assert ready["type"] == "ready"
    assert ready["resumed"] is True
    assert ready["protocol_version"] == protocol.PROTOCOL_VERSION

    delta = protocol.delta("t1", 2, "hello", coalesced=3)
    assert delta == {
        "type": "delta",
        "turn_id": "t1",
        "seq": 2,
        "text": "hello",
        "coalesced": 3,
    }

    err = protocol.error("nope", code="bad_frame", turn_id="t1", retryable=True)
    assert err["code"] == "bad_frame"
    assert err["turn_id"] == "t1"
    assert err["retryable"] is True

    assert protocol.aborted("t1", "cancelled")["reason"] == "cancelled"
