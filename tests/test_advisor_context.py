import json
import unittest
from copy import deepcopy

from src.advisor import _build_advise_user_content, _select_low_score_samples
from src.config import AdviseConfig, ModelConfig


class AdvisorContextTests(unittest.TestCase):
    def test_selected_samples_keep_search_evidence_and_answer_boundary(self):
        config = AdviseConfig(
            model=ModelConfig(provider="test", model="fake", base_url="https://example.invalid"),
            prompt="建议", min_low_samples=0, max_samples=1)
        for messages in (
            [{"role": "user", "content": "<web_search_context><search_queries>补贴 2025"
              "</search_queries><search_results>只有往年条件</search_results></web_search_context>"}],
            [{"role": "assistant", "tool_calls": [{"function": {
                "name": "WebSearch", "arguments": '{"queries":["补贴 2025"]}'}}]},
             {"role": "tool", "content": "只有往年条件"}],
        ):
            with self.subTest(messages=messages):
                row = {"row_index": 0, "query": "如何申请", "rendered_request": {"messages": messages},
                       "response": {"choices": [{"message": {"content": "回答"}}]},
                       "answer_representation": "production_plain_response",
                       "cites": ["不应送入模型的引用存档"]}
                before = deepcopy(row)
                samples = _select_low_score_samples([row], {0: {"overall": 2, "analysis": "线索"}}, config)
                content = _build_advise_user_content("原提示词", None, [], samples)
                self.assertIn(json.dumps(messages, ensure_ascii=False, indent=2), content)
                self.assertIn("不评显式引用准确性", content)
                self.assertNotIn("不应送入模型的引用存档", content)
                self.assertEqual(row, before)

    def test_old_sample_without_context_still_renders(self):
        content = _build_advise_user_content("原提示词", None, [], [{
            "row_index": 0, "overall": 2, "query": "问题", "response": "回答",
            "scores": {"overall": 2}, "analysis": "线索"}])
        self.assertIn("无记录不代表未搜索", content)
        self.assertIn("回答", content)
