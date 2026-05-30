"""CLI 入口模块。

子命令：
  run              运行实验（基于 experiment.yaml 配置）
  eval <run>       评测实验结果 (LLM-as-Judge)
  import           从 Excel 导入已有数据（用于评测现网数据）
  list             列出已有 run
  show <run>       查看实验结果摘要
"""

import argparse
import asyncio
import json
from pathlib import Path

from src.config import Config
from src.evaluator import run_evaluation
from src.experiment import run_experiment, RESULTS_DIR, META_FILE
from src.importer import import_excel


def main():
    parser = argparse.ArgumentParser(prog="llm-lab", description="LLM Prompt Lab — 大模型 API 实验框架")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行实验（断点续跑）")
    run_p.add_argument("--name", help="自定义 run 名称（默认根据配置自动生成，相同配置可断点续跑）")

    sub.add_parser("eval", help="评测实验结果").add_argument("run", help="run 名称（YAML key 或自动生成名）")

    import_p = sub.add_parser("import", help="从 Excel 导入已有数据（用于评测现网数据）")
    import_p.add_argument("excel", help="Excel 文件路径")
    import_p.add_argument("--name", required=True, help="生成的 run 名称")
    import_p.add_argument("--query-col", default="query", help="Query 列名（默认 query）")
    import_p.add_argument("--response-col", default="response", help="模型回答列名（默认 response）")

    sub.add_parser("list", help="列出实验定义与已有 run")

    sub.add_parser("show", help="查看结果摘要").add_argument("run", help="run 名称（YAML key 或自动生成名）")

    args = parser.parse_args()

    if args.command == "run":
        config = Config()
        exp = config.get_experiment()
        run_name = args.name if args.name else Config.generate_run_name(
            exp.model, exp.prompt_name, exp.prompt,
            exp.dataset, Config.hash_file(Path(exp.dataset)),
        )
        asyncio.run(run_experiment(config, run_name))
    elif args.command == "eval":
        config = Config()
        asyncio.run(run_evaluation(config, args.run))
    elif args.command == "import":
        config = Config()
        import_excel(
            excel_path=args.excel,
            run_name=args.name,
            config=config,
            query_col=args.query_col,
            response_col=args.response_col,
        )
    elif args.command == "list":
        config = Config()
        exp = config.get_experiment()
        print("=== 当前实验配置 ===")
        print(f"  model: {exp.model.model}  prompt: {exp.prompt_name}  dataset: {exp.dataset}")
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
                        tag = f"  model={meta.get('model', {}).get('model', '?')}  prompt={meta.get('prompt_name', '?')}"
                    print(f"  {r}{tag}")
    elif args.command == "show":
        _show_experiment(args.run)


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
