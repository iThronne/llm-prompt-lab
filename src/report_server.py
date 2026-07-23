"""HTML 报告的本地交互服务。

只监听 127.0.0.1，负责托管报告并把页面内的追问请求转交给 asker。
浏览器端使用 NDJSON 接收增量文本，API Key 始终只存在于 Python 进程中。
"""

import asyncio
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.asker import run_ask_stream
from src.config import AdviseConfig, AdviseConfigLoader, EvalConfigLoader
from src.constants import RESULTS_DIR
from src.reporter import load_qa

MAX_REQUEST_BYTES = 1024 * 1024
MAX_QUESTION_CHARS = 20_000


class ReportHTTPServer(ThreadingHTTPServer):
    """携带当前 run 上下文的本地报告服务器。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        run_name: str,
        report_path: Path,
        advise_cfg: AdviseConfig,
        judge_prompt: str,
    ):
        super().__init__(server_address, ReportRequestHandler)
        self.run_name = run_name
        self.report_path = report_path
        self.qa_path = RESULTS_DIR / run_name / "qa.jsonl"
        self.advise_cfg = advise_cfg
        self.judge_prompt = judge_prompt
        self.ask_lock = threading.Lock()


class ReportRequestHandler(BaseHTTPRequestHandler):
    """提供报告、追问历史和流式追问接口。"""

    server: ReportHTTPServer
    protocol_version = "HTTP/1.0"
    server_version = "LLMPromptLab/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/report.html"):
            self._serve_report()
            return
        if parsed.path == "/api/qa":
            self._serve_qa_history(parsed.query)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/ask":
            self._send_json(404, {"error": "Not found"})
            return
        if not self._origin_is_allowed():
            self._send_json(403, {"error": "只允许从本地报告页面发起追问"})
            return

        payload = self._read_json_body()
        if payload is None:
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return

        try:
            row_index = int(payload.get("row_index"))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "row_index 必须是整数"})
            return

        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "追问内容不能为空"})
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send_json(
                400,
                {"error": f"追问内容不能超过 {MAX_QUESTION_CHARS} 个字符"},
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self._client_closed = False

        def emit(event: dict):
            if self._client_closed:
                return
            try:
                line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # 即使浏览器关闭面板，也继续完成模型调用并落盘。
                self._client_closed = True

        emit({
            "type": "start",
            "row_index": row_index,
            "model": self.server.advise_cfg.model.model,
        })

        try:
            # 串行化写 qa.jsonl，保证同一服务中的 turn 编号和历史顺序稳定。
            with self.server.ask_lock:
                answer, turn = asyncio.run(run_ask_stream(
                    self.server.run_name,
                    row_index,
                    question,
                    self.server.advise_cfg,
                    self.server.judge_prompt,
                    on_content=lambda text: emit({"type": "delta", "content": text}),
                ))
            emit({"type": "done", "turn": turn, "answer": answer})
        except Exception as exc:
            print(f"[error] report ask row={row_index}: {exc}")
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _serve_report(self):
        try:
            content = self.server.report_path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": "报告文件不存在"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _serve_qa_history(self, query: str):
        params = parse_qs(query)
        try:
            row_index = int(params.get("row_index", [""])[0])
        except (TypeError, ValueError):
            self._send_json(400, {"error": "row_index 必须是整数"})
            return

        history = load_qa(self.server.qa_path).get(row_index, [])
        self._send_json(200, {"row_index": row_index, "history": history})

    def _read_json_body(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Content-Length 非法"})
            return None
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(400, {"error": "请求体为空或过大"})
            return None
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "请求体必须是 UTF-8 JSON"})
            return None

    def _origin_is_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port == self.server.server_port
        )

    def _send_json(self, status: int, payload: dict):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args):
        """只保留错误请求日志，避免每次拉取历史都刷屏。"""
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def create_report_server(
    run_name: str,
    report_path: Path,
    advise_cfg: AdviseConfig,
    judge_prompt: str,
    port: int = 8765,
) -> ReportHTTPServer:
    """创建仅绑定本机回环地址的报告服务器，便于测试和 CLI 复用。"""
    return ReportHTTPServer(
        ("127.0.0.1", port),
        run_name,
        report_path,
        advise_cfg,
        judge_prompt,
    )


def serve_report(
    run_name: str,
    report_path: Path,
    port: int = 8765,
    open_browser: bool = True,
):
    """加载追问配置并阻塞运行本地报告服务，直到用户按 Ctrl+C。"""
    scores_path = RESULTS_DIR / run_name / "scores.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"该 run 尚未评测（{scores_path} 不存在），请先运行 eval。"
        )

    advise_cfg = AdviseConfigLoader().get_advise()
    judge_prompt = EvalConfigLoader().get_eval().prompt
    server = create_report_server(
        run_name, report_path, advise_cfg, judge_prompt, port=port,
    )
    actual_port = server.server_port
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"[serve] 交互报告已启动 → {url}")
    print("[serve] API Key 仅保留在本地 Python 进程；按 Ctrl+C 停止")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止")
    finally:
        server.server_close()
