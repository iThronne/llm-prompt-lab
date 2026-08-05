"""HTML 报告手动排除评分统计的回归测试。"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.reporter import generate_html_report


class ReportStatisticsExclusionsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.temp_dir.name)
        self.run_name = "statistics-exclusions"
        self.run_dir = self.results_dir / self.run_name
        self.run_dir.mkdir()

        responses = [
            {
                "row_index": row_index,
                "query": f"query {row_index}",
                "response": {
                    "choices": [{
                        "message": {"content": f"answer {row_index}"},
                        "finish_reason": "stop",
                    }],
                },
                "latency_ms": 100 + row_index,
            }
            for row_index in (1, 2, 3)
        ]
        scores = [
            {"row_index": 1, "relevance": 5, "overall": 4, "analysis": ""},
            {"row_index": 2, "relevance": 3, "overall": 2, "analysis": ""},
            {"row_index": 3, "relevance": None, "overall": None, "analysis": ""},
        ]
        summary = {
            "dimensions": ["relevance", "overall"],
            "summary": {
                "avg_relevance": 4,
                "avg_overall": 3,
                "count_relevance": 2,
                "count_overall": 2,
            },
        }

        self._write_jsonl(self.run_dir / "responses.jsonl", responses)
        self._write_jsonl(self.run_dir / "scores.jsonl", scores)
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, entries: list[dict]):
        path.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
            encoding="utf-8",
        )

    def test_report_contains_persistent_exclusion_controls_and_recalculation(self):
        with patch("src.reporter.RESULTS_DIR", self.results_dir):
            report_path = generate_html_report(self.run_name)

        html = report_path.read_text(encoding="utf-8")

        self.assertIn("计入统计", html)
        self.assertIn('id="includedStatsCount"', html)
        self.assertIn("function setRowIncludedInStatistics", html)
        self.assertIn("function recomputeScoreStatistics", html)
        self.assertIn("function resetStatisticsExclusions", html)
        self.assertIn("localStorage.setItem", html)
        self.assertIn("const DIM_NAMES =", html)
        self.assertIn("相关性（4分）", html)
        self.assertIn("综合（3分）", html)
        self.assertIn("scoreRadarChart.update()", html)
        self.assertIn("rowHasNumericScore(row)", html)

        node = shutil.which("node")
        if node:
            inline_script = html.split("<script>", 1)[1].split("</script>", 1)[0]
            result = subprocess.run(
                [node, "--check", "-"],
                input=inline_script,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
