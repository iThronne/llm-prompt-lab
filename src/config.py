"""配置加载与校验模块。

从 config/ 目录加载 experiment.yaml 以及 config/prompts/ 目录下的 .md 文件，
使用 Pydantic 做 schema 校验，提供类型安全的配置对象。

experiment.yaml 定义一组实验配置（含模型连接信息），每次运行基于该配置，
运行时快照保存到 results/<run_name>/meta.json 保证可复现。
输出文件：responses.jsonl（模型响应）、scores.jsonl（评分）、summary.json（汇总）。
"""

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path("config")


class ModelConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024

    INFRA_PARAMS: ClassVar[set[str]] = {"provider", "base_url", "api_key_env"}

    @property
    def call_params(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if k not in self.INFRA_PARAMS}


class PromptConfig(BaseModel):
    content: str


class JudgeConfig(BaseModel):
    model: ModelConfig
    prompt: str
    dimensions: list[str] = Field(
        default_factory=lambda: ["relevance", "factuality", "fluency", "structure", "overall"])


class ExperimentConfig(BaseModel):
    model: ModelConfig
    prompt: str
    prompt_name: str = ""
    dataset: str
    judge: Optional[JudgeConfig] = None


class Config:
    """聚合加载所有配置文件，提供便捷的查找方法。"""

    def __init__(self, config_dir: Optional[Path] = None):
        base = config_dir or CONFIG_DIR
        self.prompts = self._load_prompts(base / "prompts")
        self.experiment = self._load_experiment(base / "experiment.yaml")

    def _load_prompts(self, path: Path) -> dict[str, PromptConfig]:
        """扫描 config/prompts/ 目录，读取 .md 文件作为 prompt。
        prompt 名 = 文件名去掉 .md 后缀。
        """
        prompts: dict[str, PromptConfig] = {}
        for md_file in sorted(path.glob("*.md")):
            name = md_file.stem
            prompts[name] = PromptConfig(content=md_file.read_text(encoding="utf-8"))
        return prompts

    def _load_experiment(self, path: Path) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        exp = ExperimentConfig(**data)
        # prompt 字段如果对应 config/prompts/ 下的 .md 文件名，则解析为实际内容
        if exp.prompt in self.prompts:
            exp.prompt_name = exp.prompt
            exp.prompt = self.prompts[exp.prompt].content
        if exp.judge and exp.judge.prompt in self.prompts:
            exp.judge.prompt = self.prompts[exp.judge.prompt].content
        return exp

    def get_prompt(self, name: str) -> PromptConfig:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self.prompts.keys())}")
        return self.prompts[name]

    def get_experiment(self) -> ExperimentConfig:
        return self.experiment

    @staticmethod
    def hash_file(path: Path) -> str:
        """计算文件内容的 MD5 hash，用于检测数据集等文件是否发生变化。"""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:8]

    @staticmethod
    def generate_run_name(
            model_config: ModelConfig, prompt_name: str, prompt_content: str,
            dataset: str, dataset_content_hash: str,
    ) -> str:
        """Generate a deterministic run name: {model}-{prompt}-{dataset_stem}-{hash}

        hash covers model call params + prompt content + dataset file content,
        so any substantive change produces a different run name.
        The same config always produces the same run name, enabling resume/checkpoint.
        """
        payload = {
            "model_config": model_config.call_params,
            "prompt_content": prompt_content,
            "dataset_content_hash": dataset_content_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        h = hashlib.md5(canonical.encode()).hexdigest()[:8]
        dataset_stem = Path(dataset).stem
        return f"{model_config.model}-{prompt_name}-{dataset_stem}-{h}"
