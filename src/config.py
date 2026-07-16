"""配置加载与校验模块。

从 config/ 目录加载配置文件，使用 Pydantic 做 schema 校验，提供类型安全的配置对象。

配置文件：
  - experiment.yaml: 实验配置（dataset + profiles），run 命令使用
  - eval.yaml: 评测配置（judge model + prompt + dimensions），eval 命令使用
  - prompts/: 候选模型和评测模型的 prompt 文件

两个配置文件完全独立：run 只关心 experiment.yaml，eval 只关心 eval.yaml。
运行时快照分别保存到 results/<run_name>/meta.json（run）和 eval_meta.json（eval）。
"""

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
DATASET_PROMPT_TEMPLATE_FILENAME = "dataset-prompt-template.md"


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


class SanitizeRule(BaseModel):
    """单条脱敏规则：用 pattern 正则匹配，替换为 replacement。"""

    pattern: str
    replacement: str = ""


class EvalConfig(BaseModel):
    """评测配置（LLM-as-Judge），从 eval.yaml 加载。"""

    model: ModelConfig
    prompt: str
    dimensions: list[str] = Field(
        default_factory=lambda: ["relevance", "factuality", "fluency", "structure", "timeliness", "localization", "search_planning", "search_relevance", "search_utilization", "overall"])
    sanitize: list[SanitizeRule] = Field(default_factory=list)


class AdviseConfig(BaseModel):
    """优化建议配置（读 run 结果 → 建议），从 advise.yaml 加载。

    独立于 experiment.yaml / eval.yaml，advise 命令运行时实时加载。
    """

    model: ModelConfig
    prompt: str  # 加载后由 Loader 替换为文件内容
    # 低分样本选取：overall <= low_score_threshold 视为低分；
    # 不足 min_low_samples 时按 overall 升序补足；总样本上限 max_samples。
    low_score_threshold: int = 3
    min_low_samples: int = 5
    max_samples: int = 20


class ExperimentConfig(BaseModel):
    candidate: ModelConfig
    prompt: str
    prompt_name: str = ""
    # 数据集目录名（位于 data/ 下）。约定结构：
    #   data/<dataset>/<dataset>.jsonl|.csv|.xlsx   数据集本体（.jsonl/.csv 优先，无长度上限）
    #   data/<dataset>/dataset-prompt-template.md  可选的 candidate prompt 模板（jinja2）
    dataset: str
    # 是否使用数据集自带的 prompt 模板做反推+重渲染。
    # 默认 True：dataset-prompt-template.md 存在则启用；不存在则自动退回静态注入。
    # 显式置 False 可强制走静态注入路径（即使文件存在）。
    use_dataset_prompt_template: bool = True
    # 加载后由 ExperimentConfigLoader 注入（用户不在 yaml 里写这两项）：
    dataset_prompt_template: str = ""       # 模板内容
    dataset_prompt_template_name: str = ""  # 模板文件名（用于 meta.json 追溯）

    @property
    def dataset_dir(self) -> Path:
        return DATA_DIR / self.dataset

    @property
    def dataset_path(self) -> Path:
        """数据集文件：优先 .jsonl/.csv（字段无长度上限），回退 .xlsx。"""
        for ext in (".jsonl", ".csv"):
            candidate = self.dataset_dir / f"{self.dataset}{ext}"
            if candidate.exists():
                return candidate
        return self.dataset_dir / f"{self.dataset}.xlsx"


PROMPT_EXTENSIONS = {".md", ".txt"}


def _load_prompts(path: Path) -> dict[str, PromptConfig]:
    """扫描 config/prompts/ 目录，读取 prompt 文件。

    支持 .md / .txt 等扩展名。key 为完整文件名（如 test.md），
    YAML 中必须写带扩展名的文件名来引用，否则报错。
    """
    prompts: dict[str, PromptConfig] = {}
    for ext in PROMPT_EXTENSIONS:
        for prompt_file in sorted(path.glob(f"*{ext}")):
            prompts[prompt_file.name] = PromptConfig(content=prompt_file.read_text(encoding="utf-8"))
    return prompts


class ExperimentConfigLoader:
    """加载实验配置（experiment.yaml + prompts）。

    run 命令使用。支持两种 experiment.yaml 格式：
      - 单配置模式：整个文件为一组实验配置（简单场景）。
      - 多 profile 模式：profiles 节定义多个完整配置，dataset 共享。
    """

    def __init__(self, config_dir: Optional[Path] = None, profile: Optional[str] = None):
        base = config_dir or CONFIG_DIR
        self.prompts = _load_prompts(base / "prompts")
        self.profile_name: str = ""
        self.available_profiles: list[str] = []
        self.experiment = self._load_experiment(base / "experiment.yaml", profile)

    def _load_experiment(self, path: Path, profile: Optional[str]) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        if "profiles" in data:
            # 多 profile 模式：profiles 内每个 profile 是完整配置，共享 dataset
            profiles_data: dict = data.pop("profiles")
            self.available_profiles = sorted(profiles_data.keys())

            selected = profile or "default"
            if selected not in profiles_data:
                raise ValueError(
                    f"Profile '{selected}' not found. Available: {self.available_profiles}"
                )
            self.profile_name = selected

            # 共享配置（dataset）+ 所选 profile（candidate, prompt）
            shared = {k: v for k, v in data.items() if k == "dataset"}
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

        # 检查 dataset 约定：目录必须存在，且目录下有数据集文件（.jsonl/.csv/.xlsx）
        if not exp.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {exp.dataset_dir}")
        if not exp.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {exp.dataset_path} "
                f"(expected: data/<dataset>/<dataset>.{{jsonl,csv,xlsx}})"
            )

        # 加载数据集自带的 prompt 模板（可选）：从 data/<dataset>/dataset-prompt-template.md
        template_path = exp.dataset_dir / DATASET_PROMPT_TEMPLATE_FILENAME
        if exp.use_dataset_prompt_template and template_path.exists():
            exp.dataset_prompt_template = template_path.read_text(encoding="utf-8")
            exp.dataset_prompt_template_name = DATASET_PROMPT_TEMPLATE_FILENAME
        else:
            exp.dataset_prompt_template = ""
            exp.dataset_prompt_template_name = ""

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
        """Generate a deterministic run name: {dataset_stem}@{provider}@{model}@[{profile}@]{prompt}@{hash}

        稳定的部分在前（dataset > provider > model > profile > prompt），
        方便目录浏览时同一数据集和模型的实验自然聚集。
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

        # model 名可能含 /（如 meta/llama-3），替换为 _ 避免目录路径问题
        safe_model = model_config.model.replace("/", "_")
        parts = [dataset_stem, model_config.provider, safe_model]
        if profile_name:
            parts.append(profile_name)
        parts.extend([prompt_name, h])
        return "@".join(parts)


class EvalConfigLoader:
    """加载评测配置（eval.yaml + prompts + domain_prompts）。

    eval 命令使用。独立于 experiment.yaml，不加载实验配置。
    """

    def __init__(self, config_dir: Optional[Path] = None):
        base = config_dir or CONFIG_DIR
        self.prompts = _load_prompts(base / "prompts")
        self.eval_config = self._load_eval_config(base / "eval.yaml")

    def _load_eval_config(self, path: Path) -> EvalConfig:
        """加载并解析 eval.yaml，将 prompt 文件名替换为内容。"""
        if not path.exists():
            raise FileNotFoundError(
                f"评测配置文件不存在: {path}\n"
                f"请创建 config/eval.yaml 来配置评测模型。"
            )

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        eval_cfg = EvalConfig(**data)

        # 解析 prompt 文件名 → 内容
        if eval_cfg.prompt not in self.prompts:
            raise ValueError(
                f"Judge prompt '{eval_cfg.prompt}' not found in config/prompts/. "
                f"Available: {list(self.prompts.keys())}"
            )
        eval_cfg.prompt = self.prompts[eval_cfg.prompt].content
        return eval_cfg

    def get_eval(self) -> EvalConfig:
        return self.eval_config


class AdviseConfigLoader:
    """加载优化建议配置（advise.yaml + prompts）。

    advise 命令使用。独立于 experiment.yaml / eval.yaml，运行时实时加载。
    """

    def __init__(self, config_dir: Optional[Path] = None):
        base = config_dir or CONFIG_DIR
        self.prompts = _load_prompts(base / "prompts")
        self.advise_config = self._load_advise_config(base / "advise.yaml")

    def _load_advise_config(self, path: Path) -> AdviseConfig:
        """加载并解析 advise.yaml，将 prompt 文件名替换为内容。"""
        if not path.exists():
            raise FileNotFoundError(
                f"优化建议配置文件不存在: {path}\n"
                f"请创建 config/advise.yaml 来配置建议模型。"
            )

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        advise_cfg = AdviseConfig(**data)

        # 解析 prompt 文件名 → 内容
        if advise_cfg.prompt not in self.prompts:
            raise ValueError(
                f"Advise prompt '{advise_cfg.prompt}' not found in config/prompts/. "
                f"Available: {list(self.prompts.keys())}"
            )
        advise_cfg.prompt = self.prompts[advise_cfg.prompt].content
        return advise_cfg

    def get_advise(self) -> AdviseConfig:
        return self.advise_config
