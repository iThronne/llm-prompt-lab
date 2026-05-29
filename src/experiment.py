"""实验运行器模块。

核心功能：
- 断点续跑（SQLite checkpoint）
- 指数退避重试
- tqdm 进度显示
- 结果写入 results.jsonl
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.config import Config, ExperimentConfig
from src.dataset import load_dataset, render_request
from src.models import create_client, call_model

RESULTS_DIR = Path("results")
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class CheckpointManager:
    """使用 SQLite 管理实验进度，支持断点续跑。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint (
                    experiment_name TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'done',
                    PRIMARY KEY (experiment_name, row_index)
                )
            """)
            conn.commit()

    def is_done(self, experiment_name: str, row_index: int) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM checkpoint WHERE experiment_name = ? AND row_index = ? AND status = 'done'",
                (experiment_name, row_index),
            ).fetchone()
            return row is not None

    def mark_done(self, experiment_name: str, row_index: int):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoint (experiment_name, row_index, status) VALUES (?, ?, 'done')",
                (experiment_name, row_index),
            )
            conn.commit()


def _append_result(results_path: Path, record: dict):
    """追加一条结果到 JSONL 文件。"""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_experiment(config: Config, experiment_name: str):
    """运行指定实验（支持断点续跑）。"""
    exp = config.get_experiment(experiment_name)
    model_cfg = config.get_model(exp.model)
    template_cfg = config.get_template(exp.template)

    client = create_client(model_cfg)
    rows = load_dataset(exp.dataset)
    total = len(rows)

    result_dir = RESULTS_DIR / experiment_name
    checkpoint = CheckpointManager(result_dir / "checkpoint.db")
    results_path = result_dir / "results.jsonl"

    # 统计已完成的条目
    done_count = sum(1 for i in range(total) if checkpoint.is_done(experiment_name, i))
    if done_count > 0:
        print(f"[resume] {done_count}/{total} already done, resuming...")

    pbar = tqdm(total=total, initial=done_count, desc=experiment_name, unit="item")

    for idx, row in enumerate(rows):
        if checkpoint.is_done(experiment_name, idx):
            continue

        # Jinja2 模板变量：模板中的 system_prompt + 数据行中的 query
        variables = {"system_prompt": template_cfg.system_prompt, "query": row["query"]}
        try:
            request = render_request(row["api_json"], variables)
        except Exception as e:
            tqdm.write(f"[error] row {idx}: template render failed: {e}")
            continue

        start = time.monotonic()
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await call_model(client, model_cfg.model, request, exp.params)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    tqdm.write(f"[retry] row {idx} attempt {attempt + 1} failed: {e} — retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    tqdm.write(f"[error] row {idx} failed after {MAX_RETRIES} retries: {e}")
                    pbar.update(1)
                    continue
        else:
            pbar.update(1)
            continue

        latency_ms = (time.monotonic() - start) * 1000
        result = {
            "experiment": experiment_name,
            "row_index": idx,
            "query": row["query"],
            "rendered_request": request,
            "response": response,
            "model": model_cfg.model,
            "latency_ms": round(latency_ms, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_result(results_path, result)
        checkpoint.mark_done(experiment_name, idx)
        pbar.update(1)

    pbar.close()
    print(f"[done] experiment '{experiment_name}' completed, results → {results_path}")