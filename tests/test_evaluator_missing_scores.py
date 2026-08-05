import json
import unittest

from src.evaluator import (
    _build_skipped_score,
    _compute_summary,
    _extract_response_text,
    _is_empty_response,
    _parse_judge_output,
)
from src.reporter import _extract_response_text as _extract_report_response_text


DIMENSIONS = ["relevance", "timeliness", "overall"]


class MissingScoreTests(unittest.TestCase):
    def test_empty_response_detection(self):
        empty_responses = [
            None,
            {},
            {"choices": []},
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": " \n\t"}}]},
            {"choices": [{"message": {"content": "None"}}]},
            {"choices": [{"message": {"content": " none \n"}}]},
        ]
        for response in empty_responses:
            with self.subTest(response=response):
                self.assertTrue(_is_empty_response(response))

    def test_non_empty_response_detection_and_extraction(self):
        response = {"choices": [{"message": {"content": " answer "}}]}

        self.assertFalse(_is_empty_response(response))
        self.assertEqual(_extract_response_text(response), " answer ")

    def test_none_string_is_preserved_for_reports(self):
        response = {"choices": [{"message": {"content": "None"}}]}

        self.assertTrue(_is_empty_response(response))
        self.assertEqual(_extract_report_response_text(response), "None")

    def test_build_skipped_score_uses_null_dimensions(self):
        result = {"row_index": 7, "query": "q", "response": None}

        score = _build_skipped_score(result, DIMENSIONS)

        self.assertEqual(score["error"], "empty_response")
        self.assertEqual(score["analysis"], "候选模型回复为空，已跳过评测。")
        self.assertTrue(all(score[dimension] is None for dimension in DIMENSIONS))

    def test_parse_judge_output_accepts_null_and_normalizes_dash(self):
        content = json.dumps(
            {
                "analysis": "相关性（4分）：有效\n实时性（-）：不适用\n综合（4分）：良好",
                "relevance": 4,
                "timeliness": "-",
                "overall": 4,
            },
            ensure_ascii=False,
        )

        parsed = _parse_judge_output(content, DIMENSIONS)

        self.assertEqual(parsed["relevance"], 4)
        self.assertIsNone(parsed["timeliness"])
        self.assertEqual(parsed["overall"], 4)

    def test_parse_judge_output_rejects_invalid_scores(self):
        for invalid_score in [0, 6, True, "5"]:
            with self.subTest(invalid_score=invalid_score):
                content = json.dumps(
                    {
                        "analysis": "invalid",
                        "relevance": invalid_score,
                        "timeliness": None,
                        "overall": 4,
                    }
                )
                with self.assertRaisesRegex(ValueError, "relevance"):
                    _parse_judge_output(content, DIMENSIONS)

    def test_compute_summary_excludes_skipped_rows_and_null_dimensions(self):
        scores = [
            {"relevance": 4, "timeliness": None, "overall": 4},
            {"relevance": 2, "timeliness": 5, "overall": 3},
            {"error": "empty_response", "relevance": None, "timeliness": None, "overall": None},
        ]

        summary = _compute_summary(scores, DIMENSIONS)

        self.assertEqual(
            summary,
            {
                "total_items": 2,
                "total_records": 3,
                "count_relevance": 2,
                "avg_relevance": 3.0,
                "count_timeliness": 1,
                "avg_timeliness": 5.0,
                "count_overall": 2,
                "avg_overall": 3.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
