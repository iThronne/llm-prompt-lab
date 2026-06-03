"""配置加载与校验模块。

从 config/ 目录加载 experiment.yaml 以及 config/prompts/ 目录下的 prompt 文件，
使用 Pydantic 做 schema 校验，提供类型安全的配置对象。

experiment.yaml 结构：
  - dataset: 数据集路径（所有 profile 共享）
  - judge: 评测配置（所有 profile 共享，可选）
  - profiles: 命名实验配置列表，每个 profile 包含完整的 candidate 和 prompt，
    通过 --profile 选择。

运行时快照保存到 results/<run_name>/meta.json 保证可复现。
输出文件：responses.jsonl（模型响应）、scores.jsonl（评分）、summary.json（汇总）。
"""

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path("config")


class ModelConfig(BaseModel):
    """模型连接配置。

    标准 OpenAI SDK 参数直接定义（model, temperature, max_tokens 等）。
    通过指定 extra="allow"，未显式定义的采样参数（top_p, min_p 等）也会透传给 API。
    非标准 API 参数（如 chat_template_kwargs）放在 extra_body 下，
    SDK 会将其展开到 HTTP 请求体顶层，与标准参数并列。
    """

    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    extra_body: Optional[dict] = None
    stream: bool = False
    use_proxy: bool = False

    INFRA_PARAMS: ClassVar[set[str]] = {"provider", "base_url", "api_key_env", "stream", "use_proxy"}

    @property
    def call_params(self) -> dict:
        """返回 API 调用所需的参数（排除基础设施字段和 None 值）。

        包括标准字段（model, temperature, max_tokens）以及 extra_body（如已配置）。
        """
        return {
            k: v for k, v in self.model_dump().items()
            if k not in self.INFRA_PARAMS and v is not None
        }


class PromptConfig(BaseModel):
    content: str


class JudgeConfig(BaseModel):
    model: ModelConfig
    prompt: str
    dimensions: list[str] = Field(
        default_factory=lambda: ["relevance", "factuality", "fluency", "structure", "timeliness", "localization", "search_quality", "overall"])


class ExperimentConfig(BaseModel):
    candidate: ModelConfig
    prompt: str
    prompt_name: str = ""
    dataset: str
    judge: Optional[JudgeConfig] = None


class Config:
    """聚合加载所有配置文件，提供便捷的查找方法。

    支持两种 experiment.yaml 格式：
      - 单配置模式：整个文件为一组实验配置（简单场景）。
      - 多 profile 模式：profiles 节定义多个完整配置，dataset/judge 共享。
    """

    def __init__(self, config_dir: Optional[Path] = None, profile: Optional[str] = None):
        base = config_dir or CONFIG_DIR
        self.prompts = self._load_prompts(base / "prompts")
        self.domain_prompts = self._load_domain_prompts(base / "prompts" / "judge-domains")
        self.profile_name: str = ""
        self.available_profiles: list[str] = []
        self.experiment = self._load_experiment(base / "experiment.yaml", profile)

    PROMPT_EXTENSIONS = {".md", ".txt"}

    def _load_prompts(self, path: Path) -> dict[str, PromptConfig]:
        """扫描 config/prompts/ 目录，读取 prompt 文件。

        支持 .md / .txt 等扩展名。key 为完整文件名（如 test.md），
        YAML 中必须写带扩展名的文件名来引用，否则报错。
        """
        prompts: dict[str, PromptConfig] = {}
        for ext in self.PROMPT_EXTENSIONS:
            for prompt_file in sorted(path.glob(f"*{ext}")):
                prompts[prompt_file.name] = PromptConfig(content=prompt_file.read_text(encoding="utf-8"))
        return prompts

    def _load_domain_prompts(self, path: Path) -> dict[str, str]:
        """扫描 config/prompts/judge-domains/ 目录，加载垂域评测标准。

        key 为文件名去掉扩展名（如 coding.md → coding），
        value 为文件内容。数据集的 domain 字段值需与 key 匹配。
        """
        if not path.exists():
            return {}
        domain_prompts: dict[str, str] = {}
        for ext in self.PROMPT_EXTENSIONS:
            for f in sorted(path.glob(f"*{ext}")):
                domain_prompts[f.stem] = f.read_text(encoding="utf-8")
        return domain_prompts

    def _load_experiment(self, path: Path, profile: Optional[str]) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        if "profiles" in data:
            # 多 profile 模式：profiles 内每个 profile 是完整配置，共享 dataset/judge
            profiles_data: dict = data.pop("profiles")
            self.available_profiles = sorted(profiles_data.keys())

            selected = profile or "default"
            if selected not in profiles_data:
                raise ValueError(
                    f"Profile '{selected}' not found. Available: {self.available_profiles}"
                )
            self.profile_name = selected

            # 共享配置（dataset, judge）+ 所选 profile（candidate, prompt）
            shared = {k: v for k, v in data.items() if k in ("dataset", "judge")}
            profile_data = profiles_data[selected]
            merged = {**shared, **profile_data}
            exp = ExperimentConfig(**merged)
        else:
            # 单配置模式（简单场景）
            self.available_profiles = []
            self.profile_name = ""
            exp = ExperimentConfig(**data)

        # prompt 字段必须匹配 config/prompts/ 下的文件名（含扩展名）
        exp.prompt_name = exp.prompt
        if exp.prompt not in self.prompts:
            raise ValueError(
                f"Prompt '{exp.prompt}' not found in config/prompts/. "
                f"Available: {list(self.prompts.keys())}"
            )
        exp.prompt = self.prompts[exp.prompt].content
        if exp.judge:
            if exp.judge.prompt not in self.prompts:
                raise ValueError(
                    f"Judge prompt '{exp.judge.prompt}' not found in config/prompts/. "
                    f"Available: {list(self.prompts.keys())}"
                )
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
            profile_name: str = "",
    ) -> str:
        """Generate a deterministic run name: [{profile}-]{provider}-{model}-{prompt}-{dataset_stem}-{hash}

        hash covers provider + model call params + prompt content + dataset file content,
        so any substantive change produces a different run name.
        The same config always produces the same run name, enabling resume/checkpoint.
        """
        payload = {
            "provider": model_config.provider,
            "model_config": model_config.call_params,
            "prompt_content": prompt_content,
            "dataset_content_hash": dataset_content_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        h = hashlib.md5(canonical.encode()).hexdigest()[:8]
        dataset_stem = Path(dataset).stem

        parts = []
        if profile_name:
            parts.append(profile_name)
        # model 名可能含 /（如 meta/llama-3），替换为 _ 避免目录路径问题
        safe_model = model_config.model.replace("/", "_")
        parts.extend([model_config.provider, safe_model, prompt_name, dataset_stem, h])
        return "@".join(parts)
