# coding=utf-8
"""Prepare a frozen TrendRadar prompt and import a Codex-written analysis."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# LiteLLM otherwise fetches a price table merely while importing TrendRadar's AI types.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from trendradar.__main__ import NewsAnalyzer
from trendradar.ai import AIAnalysisResult, AIAnalyzer
from trendradar.ai.prompt_loader import load_prompt_template
from trendradar.core import load_config
from trendradar.core.analyzer import count_rss_frequency
from trendradar.report.html import render_html_content


SCHEMA_VERSION = 1
ANALYSIS_FIELDS = (
    "core_trends",
    "sentiment_controversy",
    "signals",
    "rss_insights",
    "outlook_strategy",
)


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _snapshot_id(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _news_maps(news_data, platform_ids):
    results, names, title_info = {}, {}, {}
    for source_id, items in news_data.items.items():
        if source_id not in platform_ids:
            continue
        names[source_id] = news_data.id_to_name.get(source_id, source_id)
        results[source_id], title_info[source_id] = {}, {}
        for item in items:
            ranks = item.ranks or [item.rank]
            results[source_id][item.title] = {
                "ranks": ranks,
                "url": item.url or "",
                "mobileUrl": item.mobile_url or "",
            }
            title_info[source_id][item.title] = {
                "first_time": item.first_time or item.crawl_time,
                "last_time": item.last_time or item.crawl_time,
                "count": item.count,
                "ranks": ranks,
                "url": item.url or "",
                "mobileUrl": item.mobile_url or "",
                "rank_timeline": item.rank_timeline,
            }
    return results, names, title_info


def prepare_snapshot(date: str, output: Path) -> Dict[str, Any]:
    config = load_config()
    if config["REPORT_MODE"] != "current":
        raise ValueError("Codex 手动分析当前仅支持 report.mode=current")
    config["STORAGE"]["BACKEND"] = "local"
    config["STORAGE"]["PULL"]["ENABLED"] = False
    app = NewsAnalyzer(config=config)
    storage = app.storage_manager
    mode = config["REPORT_MODE"]
    news_data = storage.get_latest_crawl_data(date)
    if not news_data or not news_data.items:
        raise ValueError(f"{date} 没有可用热榜数据")

    results, id_to_name, title_info = _news_maps(news_data, app.ctx.platform_ids)
    word_groups, filter_words, global_filters = app.ctx.load_frequency_words()
    stats, hotlist_total = app.ctx.count_frequency(
        results,
        word_groups,
        filter_words,
        id_to_name,
        title_info,
        {},
        mode=mode,
        global_filters=global_filters,
        quiet=True,
    )

    rss_data = storage.get_latest_rss_data(date)
    raw_rss = []
    rss_stats = []
    if rss_data:
        raw_rss = app._convert_rss_items_to_list(rss_data.items, rss_data.id_to_name)
        rss_stats, _ = count_rss_frequency(
            raw_rss,
            word_groups,
            filter_words,
            global_filters=global_filters,
            max_news_per_keyword=config.get("MAX_NEWS_PER_KEYWORD", 0),
            sort_by_position_first=config.get("SORT_BY_POSITION_FIRST", False),
            timezone=app.ctx.timezone,
            rank_threshold=app.rank_threshold,
            quiet=True,
        )

    standalone = app._prepare_standalone_data(results, id_to_name, title_info, raw_rss)
    analysis_config = config["AI_ANALYSIS"]
    analyzer = AIAnalyzer.__new__(AIAnalyzer)
    analyzer.max_news = analysis_config.get("MAX_NEWS_FOR_ANALYSIS", 50)
    analyzer.include_rss = analysis_config.get("INCLUDE_RSS", True)
    analyzer.include_rank_timeline = analysis_config.get("INCLUDE_RANK_TIMELINE", False)
    analyzer.include_standalone = analysis_config.get("INCLUDE_STANDALONE", False)
    analyzer.language = analysis_config.get("LANGUAGE", "Chinese")
    analyzer.system_prompt, analyzer.user_prompt_template = load_prompt_template(
        analysis_config.get("PROMPT_FILE", "ai_analysis_prompt.txt"), label="AI"
    )
    prepared = analyzer._prepare_news_content(stats, rss_stats)
    standalone_content, standalone_count = "", 0
    if analyzer.include_standalone and standalone:
        standalone_content, standalone_count = analyzer._prepare_standalone_content(standalone)

    prepared_at = app.ctx.get_time().replace(microsecond=0)
    analysis_time = prepared_at.strftime("%Y-%m-%d %H:%M:%S")
    prompt = analyzer.user_prompt_template
    replacements = {
        "{report_mode}": mode,
        "{report_type}": NewsAnalyzer.MODE_STRATEGIES[mode]["report_type"],
        "{current_time}": analysis_time,
        "{news_count}": str(prepared.hotlist_total),
        "{rss_count}": str(prepared.rss_analyzed),
        "{platforms}": ", ".join(id_to_name.values()),
        "{keywords}": ", ".join(s.get("word", "") for s in stats if s.get("word")) or "无",
        "{news_content}": prepared.news_content,
        "{rss_content}": prepared.rss_content,
        "{standalone_content}": standalone_content,
        "{language}": analyzer.language,
    }
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)

    report_data = app.ctx.prepare_report(stats, news_data.failed_ids, {}, id_to_name, mode)
    report_data.update({
        "hotlist_total": hotlist_total,
        "platform_total": len(id_to_name),
        "rss_matched_count": sum(s.get("count", 0) for s in rss_stats),
        "rss_total_count": rss_data.get_total_count() if rss_data else 0,
        "rss_source_total": len(app.ctx.rss_feeds),
        "rss_source_failed": len(rss_data.failed_ids) if rss_data else 0,
    })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "date": date,
            "crawl_time": news_data.crawl_time,
            "collected_at": f"{date} {news_data.crawl_time.replace('-', ':')}:00",
            "mode": mode,
        },
        "prepared_at": prepared_at.isoformat(sep=" "),
        "prompt": {"system": analyzer.system_prompt, "user": prompt},
        "analysis_counts": {
            "hotlist_total": prepared.hotlist_total,
            "rss_total": prepared.rss_total,
            "analyzed_news": prepared.analyzed_count,
            "hotlist_analyzed": prepared.hotlist_analyzed,
            "rss_analyzed": prepared.rss_analyzed,
            "standalone_analyzed": standalone_count,
            "max_news_limit": analyzer.max_news,
            "include_rss": analyzer.include_rss,
            "include_standalone": analyzer.include_standalone,
        },
        "report": {
            "data": report_data,
            "total_titles": hotlist_total,
            "rss_items": rss_stats,
            "rss_new_items": [],
            "standalone_data": standalone if app.ctx.config["DISPLAY"]["REGIONS"]["STANDALONE"] else None,
            "region_order": app.ctx.region_order,
            "display_mode": app.ctx.display_mode,
            "show_new_section": app.ctx.show_new_section,
            "rendered_at": prepared_at.isoformat(sep=" "),
        },
    }
    snapshot = {**payload, "snapshot_id": _snapshot_id(payload)}
    prompt_path = output.with_suffix(".prompt.txt")
    if output.exists() or prompt_path.exists():
        raise FileExistsError(f"拒绝覆盖已有分析快照或提示词: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(
        f"[system]\n{analyzer.system_prompt}\n\n[user]\n{prompt}\n",
        encoding="utf-8",
    )
    return snapshot


def _load_snapshot(path: Path) -> Dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的分析快照格式")
    expected = snapshot.get("snapshot_id")
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if not isinstance(expected, str) or expected != _snapshot_id(payload):
        raise ValueError("分析快照校验失败：文件内容已变化")
    return snapshot


def load_result(path: Path, snapshot_id: str) -> AIAnalysisResult:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"分析结果不是有效 JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("分析结果必须是 JSON object")
    if data.get("snapshot_id") != snapshot_id:
        raise ValueError("分析结果与快照不匹配")
    for field in ANALYSIS_FIELDS:
        if not isinstance(data.get(field, ""), str):
            raise ValueError(f"分析结果字段 {field} 必须是字符串")
    summaries = data.get("standalone_summaries", {})
    if not isinstance(summaries, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in summaries.items()
    ):
        raise ValueError("分析结果字段 standalone_summaries 必须是字符串映射")
    if not any(data.get(field, "").strip() for field in ANALYSIS_FIELDS):
        raise ValueError("分析结果至少需要一个非空主板块")
    return AIAnalysisResult(
        **{field: data.get(field, "") for field in ANALYSIS_FIELDS},
        standalone_summaries=summaries,
        raw_response=path.read_text(encoding="utf-8"),
        success=True,
    )


def import_result(snapshot_path: Path, result_path: Path, output: Path) -> str:
    if output.resolve() in (snapshot_path.resolve(), result_path.resolve()):
        raise ValueError("输出路径不能与快照或分析结果路径相同")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有报告: {output}")
    snapshot = _load_snapshot(snapshot_path)
    result = load_result(result_path, snapshot["snapshot_id"])
    counts = snapshot["analysis_counts"]
    count_fields = {
        "hotlist_total": "hotlist_count",
        "rss_total": "rss_count",
        "analyzed_news": "analyzed_news",
        "hotlist_analyzed": "hotlist_analyzed",
        "rss_analyzed": "rss_analyzed",
        "standalone_analyzed": "standalone_analyzed",
        "max_news_limit": "max_news_limit",
        "include_rss": "include_rss",
        "include_standalone": "include_standalone",
    }
    for source_field, result_field in count_fields.items():
        setattr(result, result_field, counts[source_field])
    result.total_news = counts["hotlist_total"] + counts["rss_total"]
    result.ai_mode = snapshot["source"]["mode"]

    report = snapshot["report"]
    rendered_at = datetime.fromisoformat(report["rendered_at"])
    html = render_html_content(
        report_data=report["data"],
        total_titles=report["total_titles"],
        mode=snapshot["source"]["mode"],
        update_info=None,
        region_order=report["region_order"],
        get_time_func=lambda: rendered_at,
        rss_items=report["rss_items"],
        rss_new_items=report["rss_new_items"],
        display_mode=report["display_mode"],
        ai_analysis=result,
        show_new_section=report["show_new_section"],
        standalone_data=report["standalone_data"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrendRadar Codex 手动分析导入")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="冻结本地数据并生成分析提示词")
    prepare.add_argument("--date", required=True, help="SQLite 数据日期 (YYYY-MM-DD)")
    prepare.add_argument("--output", type=Path, required=True, help="快照 JSON 路径")
    importer = subparsers.add_parser("import", help="校验分析结果并渲染新报告")
    importer.add_argument("--snapshot", type=Path, required=True)
    importer.add_argument("--result", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        snapshot = prepare_snapshot(args.date, args.output)
        print(f"分析快照: {args.output}")
        print(f"分析提示词: {args.output.with_suffix('.prompt.txt')}")
        print(f"snapshot_id: {snapshot['snapshot_id']}")
    else:
        print(f"AI 分析报告: {import_result(args.snapshot, args.result, args.output)}")


if __name__ == "__main__":
    main()
