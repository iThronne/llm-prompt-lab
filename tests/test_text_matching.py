"""Unicode 查询匹配：真实执行 Python 导入及浏览器共用的 JS 搜索函数。"""

from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from openpyxl import Workbook

from src.human_notes import preview_import
from src.text_matching import normalize_query, question_key


def preview(queries, imported):
    cases = [{"row_index": i, "query": q, "answer_preview": "answer", "has_note": False,
              "revision": str(i)} for i, q in enumerate(queries)]
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["query", "note"])
    for query in imported:
        sheet.append([query, "人工评论"])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return preview_import(buffer.getvalue(), cases, sheet.title, 0, 1)


class TextMatchingTests(unittest.TestCase):
    def test_spanish_question_glyphs_accents_and_invisible_spaces(self):
        source = "¿Cuál es la capital？"
        imported = "\ufeff¿Cua\u0301l\u00a0es\u200b la capital ?\u2060"
        self.assertEqual(normalize_query(source), normalize_query(imported))
        result = preview([source], [imported])["rows"][0]
        self.assertEqual(result["match_type"], "normalized")
        self.assertEqual(result["candidates"][0]["query"], source)
        self.assertEqual(result["query"], imported)
        self.assertFalse(result["auto_select"])

    def test_question_omission_fallback_requires_manual_selection(self):
        result = preview(["Cuál es la capital"], ["¿Cuál es la capital?"])["rows"][0]
        self.assertEqual(result["match_type"], "question_punctuation")
        self.assertFalse(result["auto_select"])
        self.assertEqual(len(result["candidates"]), 1)

    def test_normalization_collisions_and_exact_priority(self):
        result = preview(["¿Qué hora？", "¿Que\u0301 hora?"], ["¿Qué hora﹖"])["rows"][0]
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue(any("目标 Query 重复" in w for w in result["warnings"]))
        exact = preview(["¿Qué hora？", "¿Qué hora?"], ["¿Qué hora?"])["rows"][0]
        self.assertEqual(exact["match_type"], "exact")
        self.assertEqual([c["row_index"] for c in exact["candidates"]], [1])

    def test_source_duplicates_use_question_compatible_key(self):
        result = preview(["Qué hora"], ["¿Qué hora?", "Qué hora？"])
        self.assertEqual(result["summary"]["duplicate_source_rows"], 2)
        self.assertTrue(all(not row["auto_select"] for row in result["rows"]))

    def test_no_accent_case_math_or_internal_question_erasure(self):
        for original, different in [("año?", "ano?"), ("Sí", "Si"), ("A", "a"),
                                    ("x²", "x2"), ("a?b", "ab"), ("foo bar", "foobar")]:
            with self.subTest(original=original):
                self.assertNotEqual(question_key(original), question_key(different))
                self.assertEqual(preview([original], [different])["rows"][0]["candidates"], [])
        self.assertEqual(preview(["???"], ["¿?"])["rows"][0]["candidates"], [])

    def test_javascript_matches_python_and_search_question_suffix(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required to execute browser search regression tests")
        module = Path("src/templates/text_matching.js").read_text(encoding="utf-8")
        values = ["¿Cua\u0301l\u00a0es？", "\ufeff¿Dónde\u200b?\u2060", " a\r\n b ﹖ ", "sí", "si", "x²", "a?b", "???"]
        checks = [["¿Cuál es la capital？", "¿Cuál es la capital?"],
                  ["¿Que\u0301 hora es?", "¿Qué hora es?"],
                  ["Dónde está", "¿Dónde está?"],
                  ["¿Dónde\u00a0 está﹖", "DÓNDE está?"],
                  ["año?", "ano?"], ["hello", "?"], ["hello?", "?"], ["a?b", "ab"]]
        script = module + "\nconst values=" + json.dumps(values) + ";const checks=" + json.dumps(checks) + ";\n" + (
            "console.log(JSON.stringify({normalized:values.map(normalizeMatchText),"
            "questions:values.map(questionMatchKey),search:checks.map(([v,q])=>matchesSearchText(v,q))}));")
        completed = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", check=True)
        output = json.loads(completed.stdout)
        self.assertEqual(output["normalized"], [normalize_query(v) for v in values])
        self.assertEqual(output["questions"], [question_key(v) for v in values])
        self.assertEqual(output["search"], [True, True, True, True, False, False, True, False])

    def test_all_search_pages_use_shared_normalizer(self):
        for filename in ("report.html", "human_notes.html", "attribution.html"):
            template = Path("src/templates", filename).read_text(encoding="utf-8")
            self.assertIn("include 'text_matching.js'", template)
            self.assertIn("matchesSearchText(", template)


if __name__ == "__main__":
    unittest.main()
