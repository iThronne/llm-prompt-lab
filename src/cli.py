"""CLI 入口模块。

子命令：
  run              运行实验（基于 experiment.yaml 配置）
  eval [run]       评测实验结果 (LLM-as-Judge)，默认评测最新实验
  import           从 Excel 导入已有数据（用于评测现网数据）
  list             列出已有 run
  show <run>       查看实验结果摘要
  report <run>     生成 HTML 可视化报告
  export <run>     导出 Excel 文件
"""

import argparse
import asyncio
import json
from pathlib import Path

from src.config import Config
from src.constants import RESULTS_DIR, META_FILE
from src.evaluator import run_evaluation
from src.experiment import run_experiment
from src.importer import import_excel
from src.reporter import generate_html_report, export_excel


def _resolve_run_name(run_name: str | None) -> str | None:
    """解析 run 名称，如果未指定则返回最新的实验目录名。

    Returns:
        run 名称，如果找不到任何实验则返回 None
    """
    if run_name:
        return run_name

    if not RESULTS_DIR.exists():
        print("[error] 没有找到任何实验结果目录")
        return None

    runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "responses.jsonl").exists()]
    if not runs:
        print("[error] 没有找到任何已完成的实验")
        return None

    latest_run = max(runs, key=lambda p: p.stat().st_mtime)
    run_name = latest_run.name
    print(f"[info] 未指定 run，使用最新实验: {run_name}")
    return run_name


def main():
    parser = argparse.ArgumentParser(prog="llm-lab", description="LLM Prompt Lab — 大模型 API 实验框架")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行实验（断点续跑）")
    run_p.add_argument("--name", help="自定义 run 名称（默认根据配置自动生成，相同配置可断点续跑）")
    run_p.add_argument("--profile", "-p", help="选择 experiment.yaml 中的 profile（默认 default）")

    eval_p = sub.add_parser("eval", help="评测实验结果")
    eval_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")
    eval_p.add_argument("--concurrency", "-c", type=int, default=1, help="并发评测数（默认 1，即串行）")

    import_p = sub.add_parser("import", help="从 Excel 导入已有数据（用于评测现网数据）")
    import_p.add_argument("excel", help="Excel 文件路径")
    import_p.add_argument("--name", required=True, help="生成的 run 名称")
    import_p.add_argument("--profile", "-p", help="选择 experiment.yaml 中的 profile（默认 default）")
    import_p.add_argument("--query-col", default="query", help="Query 列名（默认 query）")
    import_p.add_argument("--response-col", default="response", help="模型回答列名（默认 response）")
    import_p.add_argument("--api-json-col", default="api_json", help="api_json 列名（默认 api_json）")
    import_p.add_argument("--domain-col", default="domain", help="垂域分类列名（默认 domain，可选）")

    list_p = sub.add_parser("list", help="列出实验定义与已有 run")
    list_p.add_argument("--profile", "-p", help="选择 experiment.yaml 中的 profile")

    sub.add_parser("show", help="查看结果摘要").add_argument("run", help="run 名称（YAML key 或自动生成名）")

    report_p = sub.add_parser("report", help="生成 HTML 可视化报告")
    report_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")
    report_p.add_argument("--no-open", action="store_true", help="不自动打开浏览器")

    export_p = sub.add_parser("export", help="导出 Excel 文件")
    export_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")

    args = parser.parse_args()

    if args.command == "run":
        config = Config(profile=getattr(args, "profile", None))
        exp = config.get_experiment()
        run_name = args.name if args.name else Config.generate_run_name(
            exp.candidate, exp.prompt_name, exp.prompt,
            exp.dataset, Config.hash_file(Path(exp.dataset)),
            profile_name=config.profile_name,
        )
        asyncio.run(run_experiment(config, run_name))
    elif args.command == "eval":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return

        asyncio.run(run_evaluation(run_name, concurrency=args.concurrency))
        # 评测完成后自动生成报告和导出
        try:
            html_path = generate_html_report(run_name)
            print(f"[done] HTML 报告已生成 → {html_path}")
        except Exception as e:
            print(f"[warn] HTML 报告生成失败: {e}")
        try:
            xlsx_path = export_excel(run_name)
            print(f"[done] Excel 已导出 → {xlsx_path}")
        except Exception as e:
            print(f"[warn] Excel 导出失败: {e}")
    elif args.command == "import":
        config = Config(profile=getattr(args, "profile", None))
        import_excel(
            excel_path=args.excel,
            run_name=args.name,
            config=config,
            query_col=args.query_col,
            response_col=args.response_col,
            api_json_col=args.api_json_col,
            domain_col=args.domain_col,
            profile_name=config.profile_name,
        )
    elif args.command == "list":
        config = Config(profile=getattr(args, "profile", None))
        # 显示 profiles 信息
        if config.available_profiles:
            print("=== 可用 Profiles ===")
            for p in config.available_profiles:
                marker = " ← 当前" if p == config.profile_name else ""
                print(f"  {p}{marker}")
            print()
        exp = config.get_experiment()
        print("=== 当前实验配置 ===")
        print(f"  candidate: {exp.candidate.model}  prompt: {exp.prompt_name}  dataset: {exp.dataset}")
        if exp.judge:
            print(f"  judge: {exp.judge.model.model}  dims: {exp.judge.dimensions}")
        print()
        if RESULTS_DIR.exists():
            runs = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "responses.jsonl").exists()]
            if runs:
                print("=== 已有 Run ===")
                for r in sorted(runs):
                    meta_path = RESULTS_DIR / r / META_FILE
                    tag = ""
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        profile = meta.get("profile", "")
                        profile_tag = f"  profile={profile}" if profile else ""
                        candidate_info = meta.get("candidate", {})
                        tag = f"  candidate={candidate_info.get('model', '?')}  prompt={meta.get('prompt_name', '?')}{profile_tag}"
                    print(f"  {r}{tag}")
    elif args.command == "show":
        _show_experiment(args.run)
    elif args.command == "report":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return

        try:
            path = generate_html_report(run_name, open_browser=not args.no_open)
            print(f"[done] HTML 报告已生成 → {path}")
        except FileNotFoundError as e:
            print(f"[error] {e}")
    elif args.command == "export":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return

        try:
            path = export_excel(run_name)
            print(f"[done] Excel 已导出 → {path}")
        except FileNotFoundError as e:
            print(f"[error] {e}")


def _show_experiment(run_name: str):
    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    summary_path = result_dir / "summary.json"

    if not responses_path.exists():
        print(f"No results for '{run_name}' at {responses_path}")
        return

    total = 0
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"Experiment: {run_name}")
    print(f"  Completed rows: {total}")

    if summary_path.exists():
        ev = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"  Judge model: {ev.get('judge_model', 'N/A')}")
        summary = ev.get("summary", {})
        for k, v in summary.items():
            if k != "total_items":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
