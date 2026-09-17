from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import AsyncMock, patch

from openpyxl import Workbook

from src.attribution import build_input, prepare_cases, read_responses, run_attribution
from src.config import AttributionConfig, ModelConfig
from src.human_notes import (NoteConflict, apply_human_notes, list_cases, load_notes,
                             preview_import, save_notes, workbook_info)
from src.report_server import create_report_server


def xlsx(rows):
    book = Workbook()
    sheet = book.active
    sheet.title = "评论"
    for row in rows:
        sheet.append(row)
    buffer = BytesIO()
    book.save(buffer)
    book.close()
    return buffer.getvalue()


def row(index, query, answer="示例答复", **extra):
    return {"row_index": index, "query": query, "rendered_request": {"messages": [
        {"role": "user", "content": query}]},
        "response": {"choices": [{"message": {"content": answer}}]}, **extra}


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


class HumanNotesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.source = self.directory / "responses.jsonl"
        self.rows = [row(0, "独有问题"), row(1, "重复问题", "答复 A"), row(2, "重复问题", "答复 B"),
                     row(3, "已有评论", human_note="原评论"), row(4, "表内重复")]
        write_rows(self.source, self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, index, note):
        state = next(c for c in list_cases(self.directory) if c["row_index"] == index)
        return {"row_index": index, "human_note": note, "revision": state["revision"]}

    def test_preview_duplicates_unmatched_existing_and_blank(self):
        data = xlsx([["Query", "human_note"], [" 独有问题\r\n", "新评论"], ["重复问题", "需选择对应答复"],
                     ["表内重复", "评论甲"], ["表内重复", "评论乙"], ["不存在", "无法匹配"],
                     ["已有评论", "替换评论"], ["独有问题", None]])
        info = workbook_info(data)
        self.assertEqual(info[0]["name"], "评论")
        self.assertEqual(info[0]["columns"][1]["label"], "human_note")
        result = preview_import(data, list_cases(self.directory), "评论", 0, 1)
        self.assertFalse(result["rows"][0]["auto_select"])  # 空评论行也造成源 Query 重复，必须明确选择。
        self.assertEqual([c["row_index"] for c in result["rows"][1]["candidates"]], [1, 2])
        self.assertFalse(result["rows"][1]["auto_select"])
        self.assertEqual(result["summary"]["unmatched_rows"], 1)
        self.assertEqual(result["summary"]["existing_note_rows"], 1)
        self.assertTrue(result["rows"][-1]["blocked"])
        self.assertFalse((self.directory / "human_notes.jsonl").exists())

    def test_unique_match_normalizes_only_edges_and_line_endings(self):
        data = xlsx([["query", "note"], ["  独有问题\r\n", "日本語のコメント"], ["独有 问题", "不应模糊匹配"]])
        result = preview_import(data, list_cases(self.directory), "评论", 0, 1)
        self.assertTrue(result["rows"][0]["auto_select"])
        self.assertEqual(result["rows"][1]["candidates"], [])

    def test_formula_and_same_column_are_rejected(self):
        data = xlsx([["query", "note"], ["独有问题", '=HYPERLINK("https://example.com")']])
        result = preview_import(data, list_cases(self.directory), "评论", 0, 1)
        self.assertTrue(result["rows"][0]["blocked"])
        with self.assertRaises(ValueError):
            preview_import(data, list_cases(self.directory), "评论", 0, 0)
        with self.assertRaises(ValueError):
            workbook_info(b"not xlsx")

    def test_save_edit_clear_preserves_source_and_history(self):
        before = self.source.read_bytes()
        note = "日本語のコメント\n" + "超长评论" * 12000
        save_notes(self.directory, [self.change(0, note)])
        self.assertEqual(list_cases(self.directory)[0]["human_note"], note)
        applied = apply_human_notes(read_responses(self.source), self.directory)
        self.assertEqual(applied[0]["human_note"], note)
        cfg = AttributionConfig(model=ModelConfig(provider="test", model="fake", base_url="https://example.invalid"), prompt="test")
        _, quality, _ = build_input(applied[0], cfg)
        self.assertFalse(any("未绑定" in q for q in quality))
        old_hash = prepare_cases(applied, cfg, "r")[0]["input_hash"]
        save_notes(self.directory, [self.change(0, "修改后的评论")])
        new_hash = prepare_cases(apply_human_notes(read_responses(self.source), self.directory), cfg, "r")[0]["input_hash"]
        self.assertNotEqual(old_hash, new_hash)
        save_notes(self.directory, [self.change(0, ""), self.change(3, "")])
        self.assertEqual(list_cases(self.directory)[0]["human_note"], "")
        self.assertEqual(apply_human_notes(read_responses(self.source), self.directory)[3]["human_note"], "")
        self.assertEqual(len(load_notes(self.directory / "human_notes.jsonl")), 4)
        self.assertEqual(self.source.read_bytes(), before)

    def test_multi_page_conflict_is_all_or_nothing(self):
        stale = self.change(0, "页面 A")
        save_notes(self.directory, [self.change(0, "页面 B")])
        second = self.change(3, "不应被写入")
        with self.assertRaises(NoteConflict) as error:
            save_notes(self.directory, [stale, second])
        self.assertEqual(error.exception.rows, [0])
        self.assertEqual(list_cases(self.directory)[3]["human_note"], "原评论")
        self.assertEqual(list_cases(self.directory)[0]["human_note"], "页面 B")

    def test_changed_answer_or_context_does_not_apply_old_note(self):
        save_notes(self.directory, [self.change(0, "旧版评论")])
        old = self.change(0, "过期修改")
        self.rows[0]["rendered_request"]["messages"].append({"role": "system", "content": "新规则"})
        write_rows(self.source, self.rows)
        self.assertEqual(list_cases(self.directory)[0]["stale_note"], "旧版评论")
        applied = apply_human_notes(read_responses(self.source), self.directory)
        self.assertNotIn("human_note", applied[0])
        self.assertIn("旧版", applied[0]["human_note_warning"])
        with self.assertRaises(NoteConflict):
            save_notes(self.directory, [old])


class NotesHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "demo"
        self.directory.mkdir()
        write_rows(self.directory / "responses.jsonl", [row(0, "重复问题", "答复 A"), row(1, "重复问题", "答复 B"), row(2, "独有问题")])
        with patch("src.report_server.RESULTS_DIR", self.root):
            self.server = create_report_server("demo", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, value=None, raw=False, headers=None):
        request_headers = {"Origin": self.base, "X-Notes-Token": self.server.notes_token}
        request_headers.update(headers or {})
        data = None if value is None else (value if raw else json.dumps(value).encode())
        request = urllib.request.Request(self.base + path, data=data, headers=request_headers)
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read()
                return json.loads(body) if response.headers["Content-Type"].startswith("application/json") else body.decode()
        except urllib.error.HTTPError as error:
            error.close()
            raise

    def test_upload_preview_manual_save_and_reload_without_scores_or_keys(self):
        html = self.request("/notes")
        self.assertIn("人工评测评论", html)
        self.assertIn("notes-token", html)
        self.assertIn("beforeunload", html)
        upload = self.request("/api/notes/workbook", xlsx([["query", "note"], ["重复问题", "导入评论"], ["独有问题", "独有评论"]]), raw=True)
        preview = self.request("/api/notes/preview", {"upload_id": upload["upload_id"], "sheet": "评论", "query_column": 0, "note_column": 1})
        self.assertFalse(preview["rows"][0]["auto_select"])
        self.assertTrue(preview["rows"][1]["auto_select"])
        cases = self.request("/api/notes/cases")["cases"]
        self.assertEqual(cases[0]["query_match_count"], 2)
        # 人工仅选择重复 Query 对应的答复 B，并手动修改评论。
        payload = {"changes": [{"row_index": 1, "revision": cases[1]["revision"], "human_note": "手动修改后的评论"}]}
        self.assertEqual(self.request("/api/notes/save", payload)["saved"], 1)
        saved = self.request("/api/notes/cases")["cases"]
        self.assertEqual(saved[0]["human_note"], "")
        self.assertEqual(saved[1]["human_note"], "手动修改后的评论")
        self.assertEqual(self.request("/api/notes/case?row_index=1")["answer"], "答复 B")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/notes/save", payload)
        self.assertEqual(error.exception.code, 409)
        self.assertFalse((self.directory / "scores.jsonl").exists())

    def test_cross_origin_token_and_invalid_payload_rejected(self):
        for headers in ({"Origin": "http://evil.example"}, {"X-Notes-Token": "wrong"}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/notes/save", {"changes": []}, headers=headers)
            self.assertEqual(error.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/notes/cases", headers={"Host": "evil.example"})
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/notes/save", b"null", raw=True)
        self.assertEqual(error.exception.code, 400)


class AttributionNotesIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_attribution_receives_saved_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "r"
            directory.mkdir()
            write_rows(directory / "responses.jsonl", [row(0, "问题")])
            state = list_cases(directory)[0]
            save_notes(directory, [{"row_index": 0, "revision": state["revision"], "human_note": "新增评论"}])
            cfg = AttributionConfig(model=ModelConfig(provider="test", model="fake", base_url="https://example.invalid"), prompt="test", retries=0)
            api = AsyncMock(side_effect=ValueError("测试停止，不调用真实模型"))
            with patch("src.attribution.RESULTS_DIR", root), patch("src.attribution.create_client", return_value=AsyncMock()), patch("src.attribution.call_model", api):
                await run_attribution("r", cfg, formats=())
            payload = json.loads(api.call_args.args[2][1]["content"])
            self.assertEqual(payload["input_snapshot"]["human_note"], "新增评论")


if __name__ == "__main__":
    unittest.main()
