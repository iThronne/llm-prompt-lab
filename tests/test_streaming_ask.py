"""页面内流式追问的回归测试，不调用真实模型 API。"""

import asyncio
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.models import call_model_stream
from src.report_server import create_report_server


class _FakeDelta:
    def __init__(self, content: str):
        self.content = content
        self.model_extra = {}


class _FakeChoice:
    def __init__(self, content: str, finish_reason=None):
        self.delta = _FakeDelta(content)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content: str, finish_reason=None):
        self.id = "chunk-id"
        self.created = 1
        self.model = "fake-model"
        self.usage = None
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeStream:
    def __init__(self):
        self._chunks = iter([
            _FakeChunk("你"),
            _FakeChunk("好", finish_reason="stop"),
        ])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    async def create(self, **_kwargs):
        return _FakeStream()


class StreamingModelTest(unittest.TestCase):
    def test_content_callback_receives_each_delta(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions()),
        )
        model_config = SimpleNamespace(call_params={"model": "fake-model"})
        deltas = []

        response, kwargs, _ = asyncio.run(call_model_stream(
            client,
            model_config,
            [{"role": "user", "content": "hi"}],
            on_content=deltas.append,
        ))

        self.assertEqual(deltas, ["你", "好"])
        self.assertEqual(response["choices"][0]["message"]["content"], "你好")
        self.assertTrue(kwargs["stream"])


class ReportServerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_path = Path(self.temp_dir.name) / "report.html"
        self.report_path.write_text("<html>report</html>", encoding="utf-8")
        advise_cfg = SimpleNamespace(
            model=SimpleNamespace(model="fake-model"),
        )
        self.server = create_report_server(
            "test-run",
            self.report_path,
            advise_cfg,
            "judge prompt",
            port=0,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def test_report_and_streaming_ask_endpoint(self):
        with urllib.request.urlopen(self.base_url + "/") as response:
            self.assertEqual(response.read(), b"<html>report</html>")

        async def fake_run_ask_stream(
            _run_name, _row_index, _question, _advise_cfg, _judge_prompt,
            on_content=None,
        ):
            on_content("你")
            on_content("好")
            return "你好", 1

        request = urllib.request.Request(
            self.base_url + "/api/ask",
            data=json.dumps({
                "row_index": 3,
                "question": "为什么？",
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": self.base_url,
            },
            method="POST",
        )
        with patch(
            "src.report_server.run_ask_stream",
            side_effect=fake_run_ask_stream,
        ):
            with urllib.request.urlopen(request) as response:
                events = [
                    json.loads(line)
                    for line in response.read().decode("utf-8").splitlines()
                ]

        self.assertEqual(
            [event["type"] for event in events],
            ["start", "delta", "delta", "done"],
        )
        self.assertEqual(
            [event["content"] for event in events if event["type"] == "delta"],
            ["你", "好"],
        )
        self.assertEqual(events[-1]["answer"], "你好")
        self.assertEqual(events[-1]["turn"], 1)


if __name__ == "__main__":
    unittest.main()
