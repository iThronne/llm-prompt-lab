"""CLI 入口模块。

子命令：
  run              运行实验（基于 experiment.yaml 配置）
  eval [run]       评测实验结果 (LLM-as-Judge)，默认评测最新实验
  import           从 Excel 导入已有数据（用于评测现网数据）
  show <run>       查看实验结果摘要
  report <run>     生成 HTML 可视化报告，可用 --serve 在页面内流式追问
  export <run>     导出 Excel 文件
  notes [run]      按 Query 导入 XLSX 人工评论，支持页面编辑保存
  calibrate [run]  对比人工评分与 Judge 评分，生成校准报告
  advise [run]     读取 run 结果，由大模型给出 System Prompt 优化建议
  ask [run] -r N   就某个 case 向大模型追问（基于评分标准与 case 上下文）
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.advisor import run_advise
from src.asker import run_ask, run_ask_interactive
from src.calibrate import generate_calibration_report
from src.config import ExperimentConfigLoader, EvalConfigLoader, AdviseConfigLoader, AttributionConfigLoader
from src.attribution import run_attribution
from src.constants import RESULTS_DIR
from src.evaluator import run_evaluation
from src.experiment import run_experiment
from src.importer import import_data
from src.reporter import generate_html_report, export_excel, export_responses
from src.report_server import serve_report, serve_notes


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


async def _run_both(run_name: str, concurrency: int, force: bool):
    """配置加载及执行也分别隔离，某一流程失败不阻止另一流程。"""
    async def evaluate():
        config = EvalConfigLoader().get_eval()
        return await run_evaluation(run_name, config, concurrency=concurrency, force=force)

    async def attribute():
        config = AttributionConfigLoader().get_attribution()
        return await run_attribution(run_name, config)

    outcomes = await asyncio.gather(evaluate(), attribute(), return_exceptions=True)
    for name, outcome in zip(("eval", "attribute"), outcomes):
        if isinstance(outcome, Exception):
            print(f"[error] {name} 执行失败: {outcome}")
    return outcomes


def main():
    parser = argparse.ArgumentParser(prog="llm-lab", description="LLM Prompt Lab — 大模型 API 实验框架")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行实验（断点续跑）")
    run_p.add_argument("--name", help="自定义 run 名称（默认根据配置自动生成，相同配置可断点续跑）")
    run_p.add_argument("--profile", "-p", help="选择 experiment.yaml 中的 profile（默认 default）")

    eval_p = sub.add_parser("eval", help="评测实验结果")
    eval_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")
    eval_p.add_argument("--concurrency", "-c", type=int, default=1, help="并发评测数（默认 1，即串行）")
    eval_p.add_argument("--force", action="store_true", help="评测配置变更时强制重新评测（清空旧结果）")
    eval_p.add_argument("--attribute", action="store_true", help="同时独立执行归因（不读取评分结果）")

    attribute_p = sub.add_parser("attribute", help="独立归因并生成 JSONL/HTML/Excel，无需评分")
    attribute_p.add_argument("run", nargs="?", help="run 名称（默认最新实验）")
    attribute_p.add_argument("--rows", type=int, nargs="+", help="仅分析指定 row_index，默认全量")
    attribute_p.add_argument("--concurrency", "-c", type=int, help="并发数（默认 attribution.yaml）")
    attribute_p.add_argument("--force", action="store_true", help="重新归因选定案例；保留历史记录")
    attribute_p.add_argument("--report-only", action="store_true", help="不调用模型，重建当前配置报告")
    attribute_p.add_argument("--format", nargs="+", choices=["html", "xlsx"], default=["html", "xlsx"], help="报告格式（JSONL 始终保存）")
    attribute_p.add_argument("--config-dir", type=Path, help="归因配置目录（含 attribution.yaml 和 prompts/）")

    notes_p = sub.add_parser("notes", help="打开人工评论管理页面，按 Query 导入 XLSX、编辑并保存")
    notes_p.add_argument("run", nargs="?", help="run 名称（默认最新实验）")
    notes_p.add_argument("--port", type=int, default=8765, help="本地服务端口（默认 8765）")
    notes_p.add_argument("--no-open", action="store_true", help="不自动打开浏览器")

    import_p = sub.add_parser("import", help="从 Excel/JSONL 导入已有数据（用于评测现网数据）")
    import_p.add_argument("data", help="数据文件路径（.xlsx/.jsonl/.csv）")
    import_p.add_argument("--name", required=True, help="生成的 run 名称")
    import_p.add_argument("--query-col", default="query", help="Query 列名（默认 query）")
    import_p.add_argument("--response-col", default="response", help="模型回答列名（默认 response）")
    import_p.add_argument("--api-json-col", default="api_json", help="api_json 列名（默认 api_json）")
    import_p.add_argument("--note-col", default="human_note", help="人工评论列（默认 human_note，缺失时兼容 note）")

    sub.add_parser("show", help="查看结果摘要").add_argument("run", help="run 名称（YAML key 或自动生成名）")

    report_p = sub.add_parser("report", help="生成 HTML 可视化报告")
    report_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")
    report_p.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    report_p.add_argument("--serve", action="store_true", help="启动本地交互服务，在报告内流式追问")
    report_p.add_argument("--port", type=int, default=8765, help="本地交互服务端口（默认 8765）")

    export_p = sub.add_parser("export", help="导出 Excel 文件")
    export_p.add_argument("run", nargs="?", help="run 名称（可选，默认为最新的实验）")

    calibrate_p = sub.add_parser("calibrate", help="对比人工评分与 Judge 评分")
    calibrate_p.add_argument("run", nargs="?", help="run 名称（可选，默认最新）")

    advise_p = sub.add_parser("advise", help="读取 run 结果，由大模型给出 System Prompt 优化建议")
    advise_p.add_argument("run", nargs="?", help="run 名称（可选，默认最新已评测的 run）")

    ask_p = sub.add_parser("ask", help="就某个 case 向大模型追问（基于评分标准与 case 上下文）")
    ask_p.add_argument("run", nargs="?", help="run 名称（可选，默认最新已评测的 run）")
    ask_p.add_argument("--row", "-r", type=int, required=True, help="case 的 row_index")
    ask_p.add_argument("--question", "-q", help="单次追问内容；省略则进入交互式多轮")

    args = parser.parse_args()

    if args.command == "run":
        loader = ExperimentConfigLoader(profile=getattr(args, "profile", None))
        exp = loader.get_experiment()
        run_name = args.name if args.name else ExperimentConfigLoader.generate_run_name(
            exp.candidate, exp.prompt_name, exp.prompt,
            exp.dataset, ExperimentConfigLoader.hash_file(exp.dataset_path),
            profile_name=loader.profile_name,
        )
        asyncio.run(run_experiment(loader, run_name))
        try:
            path = export_responses(run_name)
            print(f"[done] responses.xlsx 已导出 → {path}")
        except Exception as e:
            print(f"[warn] responses.xlsx 导出失败: {e}")
    elif args.command == "notes":
        run_name = _resolve_run_name(args.run)
        if run_name:
            try:
                serve_notes(run_name, port=args.port, open_browser=not args.no_open)
            except (FileNotFoundError, ValueError, OSError) as exc:
                parser.exit(1, f"[error] {exc}\n")
    elif args.command == "attribute":
        if args.report_only and args.force:
            parser.error("--report-only 与 --force 不能同时使用")
        if args.concurrency is not None and args.concurrency < 1:
            parser.error("--concurrency 必须大于 0")
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return
        try:
            cfg = AttributionConfigLoader(args.config_dir).get_attribution()
            if args.concurrency is not None:
                cfg.concurrency = args.concurrency
            asyncio.run(run_attribution(run_name, cfg, rows=args.rows, force=args.force,
                                       report_only=args.report_only, formats=tuple(args.format)))
        except (FileNotFoundError, ValueError, OSError) as exc:
            parser.exit(1, f"[error] {exc}\n")
    elif args.command == "eval":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return

        if args.attribute:
            asyncio.run(_run_both(run_name, args.concurrency, args.force))
        else:
            try:
                eval_cfg = EvalConfigLoader().get_eval()
            except (FileNotFoundError, ValueError) as e:
                print(f"[error] {e}")
                return
            asyncio.run(run_evaluation(
                run_name, eval_cfg,
                concurrency=args.concurrency,
                force=args.force,
            ))
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
        import_data(
            data_path=args.data,
            run_name=args.name,
            query_col=args.query_col,
            response_col=args.response_col,
            api_json_col=args.api_json_col,
            note_col=args.note_col,
        )
    elif args.command == "show":
        _show_experiment(args.run)
    elif args.command == "report":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return

        try:
            path = generate_html_report(
                run_name,
                open_browser=not args.no_open and not args.serve,
            )
            print(f"[done] HTML 报告已生成 → {path}")
            if args.serve:
                serve_report(
                    run_name,
                    path,
                    port=args.port,
                    open_browser=not args.no_open,
                )
        except (FileNotFoundError, OSError, ValueError) as e:
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
    elif args.command == "calibrate":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return
        try:
            path = generate_calibration_report(run_name)
            print(f"[done] 校准报告已更新 → {path}")
        except FileNotFoundError as e:
            print(f"[error] {e}")

    elif args.command == "advise":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return
        try:
            loader = AdviseConfigLoader()
            advise_cfg = loader.get_advise()
        except (FileNotFoundError, ValueError) as e:
            print(f"[error] {e}")
            return
        # 前置检查：advise 依赖评分，未评测的 run 无意义
        if not (RESULTS_DIR / run_name / "scores.jsonl").exists():
            print(f"[error] 该 run 尚未评测，请先运行：python -m src.cli eval {run_name}")
            return
        try:
            path = asyncio.run(run_advise(run_name, advise_cfg))
            print(f"[done] 优化建议已生成 → {path}")
        except FileNotFoundError as e:
            print(f"[error] {e}")
        except Exception as e:
            print(f"[error] advise 失败: {e}")

    elif args.command == "ask":
        run_name = _resolve_run_name(args.run)
        if not run_name:
            return
        # ask 依赖评分上下文，未评测的 run 无意义
        if not (RESULTS_DIR / run_name / "scores.jsonl").exists():
            print(f"[error] 该 run 尚未评测，请先运行：python -m src.cli eval {run_name}")
            return
        try:
            advise_cfg = AdviseConfigLoader().get_advise()
            judge_prompt = EvalConfigLoader().get_eval().prompt
        except (FileNotFoundError, ValueError) as e:
            print(f"[error] {e}")
            return
        try:
            if args.question:
                answer = asyncio.run(run_ask(
                    run_name, args.row, args.question, advise_cfg, judge_prompt,
                ))
                print(answer)
            else:
                asyncio.run(run_ask_interactive(
                    run_name, args.row, advise_cfg, judge_prompt,
                ))
        except FileNotFoundError as e:
            print(f"[error] {e}")
        except Exception as e:
            print(f"[error] ask 失败: {e}")


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
