import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from src.answer_context import answer_notice
from src.attribution import build_input
from src.config import AttributionConfig, EvalConfig, ModelConfig
from src.evaluator import build_score_messages, compute_eval_hash, run_evaluation, score_input_hash
from src.importer import import_data
from src.reporter import _extract_search_queries, _extract_search_results
from src.search_context import embedded_search_contexts


CONTEXT = (
    "<web_search_context>\n<search_queries>甲\n乙</search_queries>\n"
    "<search_results>[3] 原始来源内容</search_results>\n</web_search_context>"
    "\n<current_query>问题</current_query>"
)


def model():
    return ModelConfig(provider="test", model="fake", base_url="https://example.invalid")


class ProductionImportTests(unittest.TestCase):
    def test_extract_new_legacy_and_mixed_formats_without_mutation(self):
        embedded = {"role": "user", "content": CONTEXT}
        legacy = [
            {"role": "assistant", "tool_calls": [
                {"function": {"arguments": '{"queries":["旧关键词"]}'}}]},
            {"role": "tool", "content": "旧结果"},
        ]
        for messages, queries, results in [
            ([embedded], "甲\n乙", "[3] 原始来源内容"),
            (legacy, '{"queries":["旧关键词"]}', "旧结果"),
            (legacy + [embedded], '{"queries":["旧关键词"]}\n\n甲\n乙',
             "旧结果\n\n[3] 原始来源内容"),
        ]:
            before = deepcopy(messages)
            self.assertEqual(_extract_search_queries({"messages": messages}), queries)
            self.assertEqual(_extract_search_results({"messages": messages}), results)
            self.assertEqual(messages, before)

    def test_only_complete_user_contexts_with_real_content(self):
        self.assertEqual(embedded_search_contexts([
            {"role": "system", "content": CONTEXT},
            {"role": "assistant", "content": CONTEXT},
            {"role": "user", "content": "<search_results>单独标签</search_results>"},
            {"role": "user", "content": CONTEXT.replace("</web_search_context>", "")},
            {"role": "user", "content": "<web_search_context><search_queries> </search_queries></web_search_context>"},
        ]), [])
        blocks = embedded_search_contexts([
            {"role": "user", "content": [{"type": "text", "text": CONTEXT}]}])
        self.assertEqual(blocks[0]["path"], "/messages/0/content/0/text")

    def test_import_preserves_artifacts_but_evaluates_plain_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = {"query": "问题", "response": "正文", "human_note": "评论",
                   "response_with_cites": "正文<rsup>1</rsup>",
                   "cites": [{"title": "来源", "content": "引用存档" * 40000}],
                   "api_json": [{"role": "user", "content": CONTEXT}]}
            source = root / "source.jsonl"
            source.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            with patch("src.importer.RESULTS_DIR", root / "runs"):
                import_data(str(source), "sample")
            run = root / "runs" / "sample"
            row = json.loads((run / "responses.jsonl").read_text(encoding="utf-8"))
            saved = json.loads((run / source.name).read_text(encoding="utf-8"))
            for key in ("response_with_cites", "cites"):
                self.assertEqual(row[key], raw[key])
                self.assertEqual(saved[key], raw[key])
            self.assertEqual(row["response"]["choices"][0]["message"]["content"], "正文")
            cfg = EvalConfig(model=model(), prompt="评分")
            score_messages = build_score_messages(row, cfg)
            self.assertIn("不评显式引用准确性", score_messages[1]["content"])
            self.assertNotIn("<rsup>", score_messages[1]["content"])
            self.assertNotIn("引用存档", score_messages[1]["content"])
            snapshot, quality, trace = build_input(row, AttributionConfig(model=model(), prompt="归因"))
            self.assertEqual(snapshot["answer"], "正文")
            self.assertTrue(any("不评显式引用准确性" in item for item in quality))
            self.assertFalse(any("当前回答不同" in item for item in quality))
            self.assertEqual(trace["calls"], [])
            self.assertEqual(trace["embedded_search_contexts"][0]["path"], "/messages/0/content")
            changed = deepcopy(row)
            changed["cites"] = []
            self.assertEqual(score_input_hash(row), score_input_hash(changed))
            changed.pop("answer_representation")
            self.assertNotEqual(score_input_hash(row), score_input_hash(changed))

    def test_legacy_import_keeps_original_answer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"query": "q", "response": "answer [1]",
                                         "api_json": []}), encoding="utf-8")
            with patch("src.importer.RESULTS_DIR", root):
                import_data(str(source), "sample")
            row = json.loads((root / "sample" / "responses.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(answer_notice(row), "")
            self.assertNotIn("cites", row)

    def test_changed_input_is_rescored_and_unchanged_input_resumes(self):
        cfg = EvalConfig(model=model(), prompt="评分", dimensions=["overall"])
        row = {"row_index": 0, "query": "q", "rendered_request": {"messages": []},
               "response": {"choices": [{"message": {"content": "正文"}}]},
               "answer_representation": "production_plain_response"}
        old = deepcopy(row)
        old.pop("answer_representation")
        response = {"choices": [{"message": {"content": json.dumps(
            {"analysis": "正文评分", "answer_trace": None, "overall": 4})}}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "sample"
            run.mkdir()
            (run / "responses.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run / "scores.jsonl").write_text(json.dumps({
                "row_index": 0, "overall": 1, "score_input_hash": score_input_hash(old)
            }) + "\n", encoding="utf-8")
            (run / "eval_meta.json").write_text(json.dumps({
                "eval_config_hash": compute_eval_hash(cfg)
            }), encoding="utf-8")
            with patch("src.evaluator.RESULTS_DIR", root), \
                    patch("src.evaluator.create_client"), \
                    patch("src.evaluator.call_model", new=AsyncMock(return_value=(response, {}))) as call:
                asyncio.run(run_evaluation("sample", cfg))
                self.assertEqual(call.await_count, 1)
                asyncio.run(run_evaluation("sample", cfg))
                self.assertEqual(call.await_count, 1)
            saved = json.loads((run / "scores.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["overall"], 4)
