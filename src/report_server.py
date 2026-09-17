"""HTML 报告的本地交互服务。

只监听 127.0.0.1，托管报告、人工评论管理页，并把追问请求转交给 asker。
浏览器端使用 NDJSON 接收增量文本，API Key 始终只存在于 Python 进程中。
"""

import asyncio
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.asker import run_ask_stream
from src.config import AdviseConfig, AdviseConfigLoader, EvalConfigLoader
from src.constants import RESULTS_DIR
from src.reporter import load_qa
from src.human_notes import (MAX_XLSX_BYTES, NoteConflict, answer_text, list_cases,
                             preview_import, save_notes, workbook_info)
from src.attribution import read_responses

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
        report_path: Path | None,
        advise_cfg: AdviseConfig | None,
        judge_prompt: str,
    ):
        super().__init__(server_address, ReportRequestHandler)
        self.run_name = run_name
        self.report_path = report_path
        self.qa_path = RESULTS_DIR / run_name / "qa.jsonl"
        self.advise_cfg = advise_cfg
        self.judge_prompt = judge_prompt
        self.ask_lock = threading.Lock()
        self.run_dir = RESULTS_DIR / run_name
        self.notes_token = secrets.token_urlsafe(32)
        self.notes_uploads = {}
        self.notes_upload_lock = threading.Lock()


class ReportRequestHandler(BaseHTTPRequestHandler):
    """提供报告、人工评论导入编辑、追问历史和流式追问接口。"""

    server: ReportHTTPServer
    protocol_version = "HTTP/1.0"
    server_version = "LLMPromptLab/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/notes" or parsed.path.startswith("/api/notes/") or (parsed.path == "/" and self.server.report_path is None):
            if not self._notes_host_allowed():
                self._send_json(403, {"error": "仅允许本机人工评论页面访问"})
                return
            self._notes_get(parsed)
            return
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
        if parsed.path.startswith("/api/notes/"):
            if (not self._notes_host_allowed() or not self.headers.get("Origin")
                    or not self._origin_is_allowed()
                    or not secrets.compare_digest(self.headers.get("X-Notes-Token", ""), self.server.notes_token)):
                self._send_json(403, {"error": "请从本机人工评论页面操作，或刷新页面后重试"})
                return
            self._notes_post(parsed)
            return
        if parsed.path != "/api/ask":
            self._send_json(404, {"error": "Not found"})
            return
        if not self._origin_is_allowed():
            self._send_json(403, {"error": "只允许从本地报告页面发起追问"})
            return
        if self.server.advise_cfg is None:
            self._send_json(400, {"error": "人工评论服务不提供模型追问，请使用 report --serve"})
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
        if self.server.report_path is None:
            self._send_json(404, {"error": "此服务仅提供人工评论页面，请访问 /notes"})
            return
        try:
            content = self.server.report_path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": "报告文件不存在"})
            return
        content = content.replace(b"<body>", '<body><div style="padding:12px 24px;background:#edf3fa"><a href="/notes">导入 / 编辑人工评测评论</a></div>'.encode("utf-8"), 1)
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

    def _read_json_body(self, limit=MAX_REQUEST_BYTES) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Content-Length 非法"})
            return None
        if content_length <= 0 or content_length > limit:
            self._send_json(400, {"error": "请求体为空或过大"})
            return None
        try:
            value = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if value is None:
                self._send_json(400, {"error": "请求体不能为 null"})
            return value
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "请求体必须是 UTF-8 JSON"})
            return None

    def _notes_host_allowed(self):
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def _notes_get(self, parsed):
        try:
            if parsed.path in ("/", "/notes"):
                from jinja2 import Environment, FileSystemLoader, select_autoescape
                environment = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                                          autoescape=select_autoescape(["html"]))
                content = environment.get_template("human_notes.html").render(
                    run_name=self.server.run_name, token=self.server.notes_token,
                    has_report=self.server.report_path is not None).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(content)
            elif parsed.path == "/api/notes/cases":
                self._send_json(200, {"run_name": self.server.run_name, "cases": list_cases(self.server.run_dir)})
            elif parsed.path == "/api/notes/case":
                index = int(parse_qs(parsed.query).get("row_index", [""])[0])
                row = next((r for r in read_responses(self.server.run_dir / "responses.jsonl") if r["row_index"] == index), None)
                if row is None:
                    self._send_json(404, {"error": "案例不存在"})
                    return
                self._send_json(200, {"query": row.get("query"), "answer": answer_text(row),
                                      "rendered_request": row.get("rendered_request")})
            else:
                self._send_json(404, {"error": "Not found"})
        except (ValueError, OSError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _notes_post(self, parsed):
        try:
            if parsed.path == "/api/notes/workbook":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_XLSX_BYTES:
                    raise ValueError("上传文件为空或超过 20 MB")
                data = self.rfile.read(length)
                sheets = workbook_info(data)
                token = secrets.token_urlsafe(24)
                with self.server.notes_upload_lock:
                    uploads = self.server.notes_uploads
                    for key in list(uploads):
                        if uploads[key][0] < time.monotonic() - 1800:
                            del uploads[key]
                    if len(uploads) >= 3:
                        del uploads[min(uploads, key=lambda key: uploads[key][0])]
                    uploads[token] = (time.monotonic(), data)
                self._send_json(200, {"upload_id": token, "sheets": sheets})
                return
            payload = self._read_json_body(limit=MAX_XLSX_BYTES)
            if payload is None:
                return
            if not isinstance(payload, dict):
                raise ValueError("请求体必须为 JSON 对象")
            if parsed.path == "/api/notes/preview":
                upload_id = payload.get("upload_id")
                if not isinstance(upload_id, str):
                    raise ValueError("请先上传 XLSX")
                with self.server.notes_upload_lock:
                    upload = self.server.notes_uploads.get(upload_id)
                if not upload or upload[0] < time.monotonic() - 1800:
                    raise ValueError("上传已过期，请重新选择文件")
                preview = preview_import(upload[1], list_cases(self.server.run_dir), payload.get("sheet"),
                                         payload.get("query_column"), payload.get("note_column"))
                self._send_json(200, preview)
            elif parsed.path == "/api/notes/save":
                self._send_json(200, save_notes(self.server.run_dir, payload.get("changes")))
            else:
                self._send_json(404, {"error": "Not found"})
        except NoteConflict as exc:
            self._send_json(409, {"error": str(exc), "conflict_rows": exc.rows})
        except (ValueError, OSError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            print(f"[error] notes: {type(exc).__name__}")
            self._send_json(400, {"error": "文件或请求无法处理，请核对 XLSX 格式后重试"})

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
    report_path: Path | None = None,
    advise_cfg: AdviseConfig | None = None,
    judge_prompt: str = "",
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


def serve_notes(run_name: str, port: int = 8765, open_browser: bool = True):
    """无需评分、模型配置或 API Key 的本地人工评论编辑页。"""
    if not (RESULTS_DIR / run_name / "responses.jsonl").exists():
        raise FileNotFoundError("该实验没有 responses.jsonl，请先运行或导入数据")
    server = create_report_server(run_name, port=port)
    url = f"http://127.0.0.1:{server.server_port}/notes"
    print(f"[serve] 人工评论管理 → {url}")
    print("[serve] 保存到当前实验 human_notes.jsonl；按 Ctrl+C 停止")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止")
    finally:
        server.server_close()


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
