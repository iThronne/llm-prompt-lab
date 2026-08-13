import unittest
from pathlib import Path

import yaml


PROMPT_PATH = Path(__file__).parents[1] / "config" / "prompts" / "judge-prompt.md"
EVAL_CONFIG_PATH = Path(__file__).parents[1] / "config" / "eval.yaml"


class JudgePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def test_contains_high_score_calibration(self):
        self.assertIn("5 分是严格保留分", self.prompt)
        self.assertIn("按最严重缺陷确定分数", self.prompt)
        self.assertIn("严重缺陷", self.prompt)

    def test_uses_controlled_external_evidence_and_observable_behavior(self):
        self.assertIn("仅在以下情况需要时使用联网搜索", self.prompt)
        self.assertIn("至少交叉核对两个相互独立的可靠来源", self.prompt)
        self.assertIn("来源名称、发布日期/更新时间或 URL", self.prompt)
        self.assertIn("相同问题、相同上下文、相同回复和相同证据必须适用相同标准", self.prompt)
        self.assertIn("不要输出或推测隐藏推理链", self.prompt)
        self.assertNotIn("推理链推断", self.prompt)

    def test_defines_missing_factuality_and_search_result_reliability(self):
        self.assertIn("不包含任何可客观核查的事实陈述", self.prompt)
        self.assertIn("来源可靠性，以及关键内容能否通过外部核验", self.prompt)
        self.assertIn("任一适用的回答质量维度为 2 分时，overall 不得超过 2", self.prompt)

    def test_search_diagnostics_do_not_directly_affect_overall(self):
        self.assertIn(
            "`overall` 只根据这些维度及最终回复的实际可用性确定",
            self.prompt,
        )
        self.assertIn(
            "`search_planning`、`search_relevance`、`search_utilization`",
            self.prompt,
        )
        self.assertIn("不直接参与 `overall`", self.prompt)
        self.assertIn("如果删去三个搜索辅助诊断维度的分数和分析，overall 应保持不变", self.prompt)
        self.assertIn('"search_relevance": 2, "search_utilization": 5, "overall": 5', self.prompt)
        self.assertNotIn("任一适用维度为 2 分时，overall 不得超过 2", self.prompt)

    def test_search_planning_scores_search_decision_even_without_search(self):
        start = self.prompt.index("### 搜索规划 (search_planning)")
        end = self.prompt.index("### 搜索结果相关性 (search_relevance)")
        section = self.prompt[start:end]

        self.assertIn("本维度**始终适用**", section)
        self.assertIn("需要搜索却没有搜索，或不需要搜索却发起搜索", section)
        self.assertIn("不需要搜索时，准确决定不搜索", section)
        self.assertNotIn("不适用条件", section)

    def test_removes_genre_creation_rules(self):
        removed_markers = [
            "创作意图识别",
            "附加格式检查清单",
            "小红书文案",
            "演讲稿",
            "邀请函",
            "写诗",
            "信件类",
        ]
        for marker in removed_markers:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, self.prompt)

    def test_eval_sampling_parameters_are_fixed(self):
        config = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
        model = config["model"]

        self.assertEqual(model["temperature"], 0)
        self.assertEqual(model["top_p"], 1)
        self.assertEqual(model["seed"], 7)
        self.assertIs(model["extra_body"]["enable_search"], True)


if __name__ == "__main__":
    unittest.main()
