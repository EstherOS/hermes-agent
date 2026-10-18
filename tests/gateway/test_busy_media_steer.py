"""Busy media steer: opt-in text-mode preparation and gating."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

def _runner(*, mode: str = "text") -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._busy_media_steer = mode
    return runner

def _source(chat_id: str = "media-steer") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name="tester",
    )

def _photo(source: SessionSource, *, text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/busy-media.png"],
        media_types=["image/png"],
    )

@pytest.mark.asyncio
async def test_for_steer_prep_forces_text_images_and_keeps_native_buffer():
    runner = _runner()
    source = _source()
    session_key = build_session_key(source)
    runner._decide_image_input_mode = lambda **_: "native"
    runner._enrich_message_with_vision = AsyncMock(return_value="VISION DESCRIPTION")
    runner._session_state(session_key).persistent.native_image_paths = ["/tmp/prior.png"]

    event = _photo(source)
    prepared = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        session_key=session_key,
        for_steer=True,
    )

    assert prepared is not None
    assert "VISION DESCRIPTION" in prepared
    assert runner._consume_pending_native_image_paths(session_key) == ["/tmp/prior.png"]
    assert event._gateway_busy_steer_prepared_text == prepared

@pytest.mark.asyncio
async def test_busy_media_steer_reuses_prepared_text_on_fallback():
    runner = _runner()
    source = _source()
    event = _photo(source, text="look")

    async def _prepare(**kwargs):
        kwargs["event"]._gateway_busy_steer_prepared_text = "PREPARED"
        return "PREPARED"

    runner._prepare_inbound_message_text = AsyncMock(side_effect=_prepare)

    first = await runner._prepare_busy_steer_text(event, "key")
    second = await runner._prepare_busy_steer_text(event, "key")

    assert first == "PREPARED"
    assert second == "PREPARED"
    assert runner._prepare_inbound_message_text.await_count == 1

    event._gateway_busy_steer_prepared_text = "CACHED"
    third = await runner._prepare_busy_steer_text(event, "key")
    assert third == "CACHED"
    assert runner._prepare_inbound_message_text.await_count == 1

@pytest.mark.asyncio
async def test_busy_media_steer_gate_allows_media_when_enabled():
    runner = _runner(mode="text")
    runner._prepare_busy_steer_text = AsyncMock(return_value="PREPARED")
    runner._try_agent_verb = lambda *args, **kwargs: True
    runner._fold_into_running_turn = lambda *args, **kwargs: None
    event = _photo(_source())
    agent = type("Agent", (), {"steer": lambda self, text: True})()

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)

    assert outcome.steered is True
    assert outcome.effective_mode == "steer"

@pytest.mark.asyncio
async def test_busy_media_steer_gate_queues_when_off():
    runner = _runner(mode="off")
    event = _photo(_source())

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", object())

    assert outcome.steered is False
    assert outcome.effective_mode == "queue"
