"""配置加载与校验模块。

从 config/ 目录加载 models.yaml, templates.yaml, experiments.yaml，
使用 Pydantic 做 schema 校验，提供类型安全的配置对象。
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


CONFIG_DIR = Path("config")


class ModelConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key_env: str = ""


class TemplateConfig(BaseModel):
    system_prompt: str


class JudgeConfig(BaseModel):
    model: str
    prompt: str
    dimensions: list[str] = Field(default_factory=lambda: ["accuracy", "completeness", "clarity"])


class ExperimentConfig(BaseModel):
    model: str
    template: str
    dataset: str
    params: dict = Field(default_factory=dict)
    judge: Optional[JudgeConfig] = None


class ModelsFile(BaseModel):
    models: dict[str, ModelConfig]


class TemplatesFile(BaseModel):
    templates: dict[str, TemplateConfig]


class ExperimentsFile(BaseModel):
    experiments: dict[str, ExperimentConfig]


class Config:
    """聚合加载所有配置文件，提供便捷的查找方法。"""

    def __init__(self, config_dir: Optional[Path] = None):
        base = config_dir or CONFIG_DIR
        self.models = self._load_models(base / "models.yaml")
        self.templates = self._load_templates(base / "templates.yaml")
        self.experiments = self._load_experiments(base / "experiments.yaml")

    def _load_models(self, path: Path) -> dict[str, ModelConfig]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ModelsFile(**data).models

    def _load_templates(self, path: Path) -> dict[str, TemplateConfig]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TemplatesFile(**data).templates

    def _load_experiments(self, path: Path) -> dict[str, ExperimentConfig]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ExperimentsFile(**data).experiments

    def get_model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_template(self, name: str) -> TemplateConfig:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[name]

    def get_experiment(self, name: str) -> ExperimentConfig:
        if name not in self.experiments:
            raise KeyError(f"Experiment '{name}' not found. Available: {list(self.experiments.keys())}")
        return self.experiments[name]