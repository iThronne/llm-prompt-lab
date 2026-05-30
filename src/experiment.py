"""实验运行器模块。

核心功能：
- 断点续跑（从 responses.jsonl 派生已完成状态）
- 指数退避重试
- tqdm 进度显示
- 结果写入 responses.jsonl
"""

import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.config import Config, ExperimentConfig
from src.constants import RESULTS_DIR, META_FILE
from src.dataset import load_dataset, render_request
from src.models import create_client, call_model

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def save_run_meta(run_name: str, experiment: ExperimentConfig, profile_name: str = ""):
    """Save run metadata (full experiment config snapshot) for reproducibility."""
    result_dir = RESULTS_DIR / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(experiment.dataset)
    shutil.copy2(dataset_path, result_dir / dataset_path.name)
    meta = {
        "run_name": run_name,
        "profile": profile_name,
        "candidate": experiment.candidate.model_dump(),
        "prompt_name": experiment.prompt_name,
        "prompt_content": experiment.prompt,
        "dataset": experiment.dataset,
        "dataset_content_hash": Config.hash_file(dataset_path),
        "judge": experiment.judge.model_dump() if experiment.judge else None,
    }
    meta_path = result_dir / META_FILE
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_run_meta(run_name: str) -> dict:
    """Load run metadata from meta.json snapshot."""
    meta_path = RESULTS_DIR / run_name / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json for run '{run_name}' at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


class CheckpointManager:
    """从 responses.jsonl 派生已完成状态，单一数据源，无一致性问题。"""

    def __init__(self, responses_path: Path, run_name: str):
        self.responses_path = responses_path
        self.run_name = run_name
        self._done: set[int] = set()
        self._load()

    def _load(self):
        """启动时扫描 responses.jsonl，加载已完成的 row_index 集合。"""
        if not self.responses_path.exists():
            return
        with open(self.responses_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("experiment") == self.run_name:
                        self._done.add(record["row_index"])

    @property
    def done_count(self) -> int:
        return len(self._done)

    def is_done(self, row_index: int) -> bool:
        return row_index in self._done

    def mark_done(self, row_index: int):
        """更新内存中的完成状态（持久化由 _append_result 保证）。"""
        self._done.add(row_index)


def _append_response(path: Path, record: dict):
    """追加一条响应到 responses.jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_experiment(config: Config, run_name: str):
    """运行实验（支持断点续跑）。

    Args:
        config: 配置对象
        run_name: 输出目录使用的名称
    """
    exp = config.get_experiment()
    model_cfg = exp.candidate

    save_run_meta(run_name, exp, profile_name=config.profile_name)

    client = create_client(model_cfg)
    rows = load_dataset(exp.dataset)
    total = len(rows)

    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    checkpoint = CheckpointManager(responses_path, run_name)

    # 统计已完成的条目
    done_count = checkpoint.done_count
    if done_count > 0:
        print(f"[resume] {done_count}/{total} already done, resuming...")

    pbar = tqdm(total=total, initial=done_count, desc=run_name, unit="item")

    for idx, row in enumerate(rows):
        if checkpoint.is_done(idx):
            continue

        # Jinja2 模板变量：prompt 内容 + 数据行中的 query
        variables = {"system_prompt": exp.prompt, "query": row["query"]}
        try:
            request = render_request(row["api_json"], variables)
        except Exception as e:
            tqdm.write(f"[error] row {idx}: template render failed: {e}")
            continue

        start = time.monotonic()
        for attempt in range(1 + MAX_RETRIES):
            try:
                response, actual_request = await call_model(client, model_cfg, request)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    tqdm.write(f"[retry] row {idx} attempt {attempt + 1} failed: {e} — retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    tqdm.write(f"[error] row {idx} failed after {MAX_RETRIES} retries: {e}")
        else:
            # for loop exhausted without break → all retries failed, skip this row
            pbar.update(1)
            continue

        latency_ms = (time.monotonic() - start) * 1000
        result = {
            "experiment": run_name,
            "row_index": idx,
            "model": model_cfg.model,
            "query": row["query"],
            "rendered_request": actual_request,
            "response": response,
            "latency_ms": round(latency_ms, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_response(responses_path, result)
        checkpoint.mark_done(idx)
        pbar.update(1)

    pbar.close()
    print(f"[done] experiment '{run_name}' completed, results → {responses_path}")
