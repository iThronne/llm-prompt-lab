"""CLI 入口模块。

子命令：
  run  <experiment>  运行实验（自动断点续跑）
  eval <experiment>  评测实验结果 (LLM-as-Judge)
  list               列出所有实验定义
  show <experiment>  查看实验结果摘要
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.config import Config
from src.experiment import run_experiment, RESULTS_DIR
from src.evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser(prog="llm-lab", description="LLM Prompt Lab — 大模型 API 实验框架")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="运行实验（断点续跑）").add_argument("experiment", help="实验名称")

    sub.add_parser("eval", help="评测实验结果").add_argument("experiment", help="实验名称")

    sub.add_parser("list", help="列出所有实验")

    sub.add_parser("show", help="查看结果摘要").add_argument("experiment", help="实验名称")

    args = parser.parse_args()

    if args.command == "run":
        config = Config()
        asyncio.run(run_experiment(config, args.experiment))
    elif args.command == "eval":
        config = Config()
        asyncio.run(run_evaluation(config, args.experiment))
    elif args.command == "list":
        config = Config()
        for name, exp in config.experiments.items():
            print(f"  {name}")
            print(f"    model: {exp.model}  template: {exp.template}  dataset: {exp.dataset}")
            if exp.judge:
                print(f"    judge: {exp.judge.model}  dims: {exp.judge.dimensions}")
            print()
    elif args.command == "show":
        _show_experiment(args.experiment)


def _show_experiment(experiment_name: str):
    result_dir = RESULTS_DIR / experiment_name
    results_path = result_dir / "results.jsonl"
    eval_path = result_dir / "evaluation.json"

    if not results_path.exists():
        print(f"No results for '{experiment_name}' at {results_path}")
        return

    total = 0
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"Experiment: {experiment_name}")
    print(f"  Completed rows: {total}")

    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        print(f"  Judge model: {ev.get('judge_model', 'N/A')}")
        summary = ev.get("summary", {})
        for k, v in summary.items():
            if k != "total_items":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()