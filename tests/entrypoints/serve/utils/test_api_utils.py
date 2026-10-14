# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from vllm.entrypoints.generate.base.protocol import StreamOptions
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    redact_sensitive_args,
    should_include_usage,
)


def _request_for_cancellation(headers=None, receive=None):
    if receive is None:

        async def receive():
            await asyncio.sleep(3600)

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(enable_server_load_tracking=False)),
        state=SimpleNamespace(),
        headers=headers or {},
        receive=receive,
        url=SimpleNamespace(path="/test"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tracking", [None, False, True])
@pytest.mark.parametrize("response_kind", ["plain", "json", "stream"])
@pytest.mark.parametrize("timeout", [None, "0", "-1"])
async def test_no_deadline_preserves_response_and_load(
    monkeypatch, tracking, response_kind, timeout
):
    """Opt-out requests keep their response and never start a deadline timer."""
    listener_started = asyncio.Event()
    listener_closed = asyncio.Event()
    background_calls: list[str] = []

    async def receive():
        listener_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            listener_closed.set()

    def unexpected_sleep(*args, **kwargs):
        pytest.fail("A request without a deadline started a timer")

    async def body():
        yield b"first"
        yield b"second"

    expected: dict[str, str] | JSONResponse | StreamingResponse = {"result": "ok"}
    if response_kind == "json":
        expected = JSONResponse(expected, status_code=201)
    elif response_kind == "stream":
        expected = StreamingResponse(body())
    if isinstance(expected, (JSONResponse, StreamingResponse)):
        expected.background = BackgroundTask(background_calls.append, "done")

    async def handler(_, raw_request):
        await listener_started.wait()
        return expected

    headers = {} if timeout is None else {"x-upstream-timeout-ms": timeout}
    request = _request_for_cancellation(headers=headers, receive=receive)
    if tracking is None:
        del request.app.state.enable_server_load_tracking
    else:
        request.app.state.enable_server_load_tracking = tracking
    request.app.state.server_load_metrics = 7
    monkeypatch.setattr(api_utils.asyncio, "sleep", unexpected_sleep)
    wrapped = api_utils.with_cancellation(api_utils.load_aware_call(handler))
    result = await wrapped(None, request)

    assert result is expected
    assert listener_closed.is_set()
    pending_response = tracking is True and response_kind != "plain"
    assert request.app.state.server_load_metrics == 7 + int(pending_response)
    if isinstance(result, StreamingResponse):
        assert [chunk async for chunk in result.body_iterator] == [b"first", b"second"]
    if isinstance(result, (JSONResponse, StreamingResponse)):
        assert result.background is not None
        await result.background()
        assert background_calls == ["done"]
    assert request.app.state.server_load_metrics == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("tracking", [False, True])
async def test_load_aware_without_cancellation_does_not_read_timeout(tracking):
    class UnreadableHeaders(dict):
        def get(self, *args, **kwargs):
            pytest.fail("A route without cancellation inspected timeout headers")

    request = _request_for_cancellation()
    request.headers = UnreadableHeaders({"x-upstream-timeout-ms": "1"})
    request.app.state.enable_server_load_tracking = tracking
    request.app.state.server_load_metrics = 7

    async def handler(_, raw_request):
        await asyncio.sleep(0.02)
        return "ok"

    assert await api_utils.load_aware_call(handler)(None, request) == "ok"
    assert request.app.state.server_load_metrics == 7


@pytest.mark.asyncio
async def test_with_cancellation_timeout_cleans_up_handler():
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def handler(_, raw_request):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.set()

    request = _request_for_cancellation(headers={"x-upstream-timeout-ms": "20"})
    response = await api_utils.with_cancellation(handler)(None, request)

    assert response.status_code == 504
    assert response.body == (
        b'{"error":{"message":"Upstream request deadline exceeded",'
        b'"type":"TimeoutError","code":504}}'
    )
    assert started.is_set()
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_with_cancellation_client_disconnect_propagates_cancellation():
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def handler(_, raw_request):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.set()

    request = _request_for_cancellation(receive=receive)
    with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
        await api_utils.with_cancellation(handler)(None, request)

    assert cleaned_up.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"x-upstream-timeout-ms": "3600000"}])
async def test_with_cancellation_outer_cancel_cleans_up_tasks(headers):
    started = asyncio.Event()
    handler_cleaned_up = asyncio.Event()
    listener_cleaned_up = asyncio.Event()

    async def receive():
        try:
            await asyncio.sleep(3600)
        finally:
            listener_cleaned_up.set()

    async def handler(_, raw_request):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            handler_cleaned_up.set()

    request = _request_for_cancellation(headers=headers, receive=receive)
    baseline_tasks = set(asyncio.all_tasks())
    wrapper_task = asyncio.create_task(
        api_utils.with_cancellation(handler)(None, request)
    )
    await started.wait()
    wrapper_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await wrapper_task

    await asyncio.sleep(0)
    leaked_tasks = [
        task
        for task in asyncio.all_tasks()
        if task not in baseline_tasks and task is not wrapper_task
    ]
    assert handler_cleaned_up.is_set()
    assert listener_cleaned_up.is_set()
    assert leaked_tasks == []


@pytest.mark.asyncio
async def test_with_cancellation_ignores_invalid_timeout_header(caplog):
    async def handler(_, raw_request):
        return "ok"

    request = _request_for_cancellation(
        headers={"x-upstream-timeout-ms": "not-an-integer"}
    )
    with caplog.at_level("WARNING", logger="vllm.entrypoints.serve.utils.api_utils"):
        result = await api_utils.with_cancellation(handler)(None, request)

    assert result == "ok"
    assert "Ignoring invalid x-upstream-timeout-ms header" in caplog.text


@pytest.mark.asyncio
async def test_with_cancellation_rejects_oversized_timeout_header(caplog):
    async def handler(_, raw_request):
        return "ok"

    request = _request_for_cancellation(headers={"x-upstream-timeout-ms": "9" * 310})
    with caplog.at_level("WARNING", logger="vllm.entrypoints.serve.utils.api_utils"):
        result = await api_utils.with_cancellation(handler)(None, request)

    assert result == "ok"
    assert "Ignoring invalid x-upstream-timeout-ms header" in caplog.text
    assert "9" * 310 not in caplog.text


@pytest.mark.asyncio
async def test_with_cancellation_timeout_releases_server_load():
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def handler(_, raw_request):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.set()

    request = _request_for_cancellation(headers={"x-upstream-timeout-ms": "20"})
    request.app.state.enable_server_load_tracking = True
    request.app.state.server_load_metrics = 0
    wrapped = api_utils.with_cancellation(api_utils.load_aware_call(handler))

    response = await wrapped(None, request)

    assert response.status_code == 504
    assert started.is_set()
    assert cleaned_up.is_set()
    assert request.app.state.server_load_metrics == 0
    assert request.state._vllm_server_load_active == 0


@pytest.mark.asyncio
async def test_with_cancellation_disconnect_releases_server_load():
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def handler(_, raw_request):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.set()

    request = _request_for_cancellation(receive=receive)
    request.app.state.enable_server_load_tracking = True
    request.app.state.server_load_metrics = 0
    wrapped = api_utils.with_cancellation(api_utils.load_aware_call(handler))

    with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
        await wrapped(None, request)

    assert cleaned_up.is_set()
    assert request.app.state.server_load_metrics == 0
    assert request.state._vllm_server_load_active == 0


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )


class TestRedactSensitiveArgs:
    API_KEY = "sk-test-secret-12345"

    def test_redact_replaces_sensitive_values_only(self):
        args = {"api_key": self.API_KEY, "hf_token": "hf_secret", "other": "visible"}
        redacted = redact_sensitive_args(args)
        assert redacted == {
            "api_key": "***",
            "hf_token": "***",
            "other": "visible",
        }
        # original dict must not be mutated
        assert args == {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "other": "visible",
        }

    def test_no_sensitive_fields_returns_original(self):
        args = {"model_tag": "org/model", "other": "visible"}
        assert redact_sensitive_args(args) is args

    def test_api_key_not_in_log(self, monkeypatch, caplog):
        non_default = {
            "model_tag": "org/model",
            "default_chat_template_kwargs": {"enable_thinking": False},
            "api_key": self.API_KEY,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(api_utils, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.serve.utils.api_utils"):
            api_utils.log_non_default_args(args=Namespace())
        message = caplog.text
        assert self.API_KEY not in message
        assert "'api_key': '***'" in message
        # non-sensitive args are still logged
        assert "org/model" in message
        assert "qwen3_coder" in message

    def test_hf_token_not_in_log(self, monkeypatch, caplog):
        hf_token = "hf_TESTTOKENVALUE123"
        non_default = {
            "model_tag": "org/model",
            "hf_token": hf_token,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(api_utils, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.serve.utils.api_utils"):
            api_utils.log_non_default_args(args=Namespace())
        message = caplog.text
        assert hf_token not in message
        assert "'hf_token': '***'" in message
        assert "org/model" in message
