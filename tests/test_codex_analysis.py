import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from trendradar.commands.codex_analysis import (
    ANALYSIS_FIELDS,
    _snapshot_id,
    import_result,
    load_result,
)


def _result(snapshot_id="snapshot"):
    return {
        "snapshot_id": snapshot_id,
        **{field: "分析" for field in ANALYSIS_FIELDS},
        "standalone_summaries": {},
    }


class CodexAnalysisTest(unittest.TestCase):
    def test_load_result_rejects_invalid_json_and_snapshot_mismatch(self):
        with TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是有效 JSON"):
                load_result(invalid, "snapshot")

            invalid.write_text(json.dumps(_result("other")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "与快照不匹配"):
                load_result(invalid, "snapshot")

    def test_import_renders_ai_section_without_overwriting_source(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            report = {
                "data": {
                    "stats": [], "new_titles": [], "failed_ids": [],
                    "total_new_count": 0,
                },
                "total_titles": 0,
                "rss_items": [],
                "rss_new_items": [],
                "standalone_data": None,
                "region_order": ["ai_analysis"],
                "display_mode": "keyword",
                "show_new_section": False,
                "rendered_at": "2026-10-19 18:02:00",
            }
            payload = {
                "schema_version": 1,
                "source": {
                    "date": "2026-10-19", "crawl_time": "18-02", "mode": "current",
                },
                "prompt": {"system": "system", "user": "user"},
                "analysis_counts": {
                    "hotlist_total": 0, "rss_total": 0, "analyzed_news": 0,
                    "hotlist_analyzed": 0, "rss_analyzed": 0,
                    "standalone_analyzed": 0, "max_news_limit": 150,
                    "include_rss": False, "include_standalone": False,
                },
                "report": report,
            }
            snapshot_id = _snapshot_id(payload)
            snapshot = tmp_path / "snapshot.json"
            snapshot.write_text(json.dumps({**payload, "snapshot_id": snapshot_id}), encoding="utf-8")
            result = tmp_path / "result.json"
            bad_output = tmp_path / "bad-report.html"
            result.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是有效 JSON"):
                import_result(snapshot, result, bad_output)
            self.assertFalse(bad_output.exists())

            result.write_text(json.dumps(_result("other")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "与快照不匹配"):
                import_result(snapshot, result, bad_output)
            self.assertFalse(bad_output.exists())

            result.write_text(json.dumps(_result(snapshot_id), ensure_ascii=False), encoding="utf-8")
            output = tmp_path / "report.html"
            snapshot_before = snapshot.read_bytes()
            result_before = result.read_bytes()

            import_result(snapshot, result, output)

            html = output.read_text(encoding="utf-8")
            self.assertIn("AI 热点分析", html)
            self.assertIn("分析", html)
            self.assertEqual(snapshot.read_bytes(), snapshot_before)
            self.assertEqual(result.read_bytes(), result_before)
            output_before = output.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖已有报告"):
                import_result(snapshot, result, output)
            self.assertEqual(output.read_bytes(), output_before)


if __name__ == "__main__":
    unittest.main()
