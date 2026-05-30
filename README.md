# LLM Prompt Lab

大模型 API 实验框架 — 支持不同模型、Prompt 模板、数据集和参数的系统化实验，内置结果记录、LLM-as-Judge 评测与断点续跑。

## 快速开始

```bash
# 安装依赖
poetry install

# 激活虚拟环境
poetry shell

# 配置 API Key
export OPENAI_API_KEY=sk-xxx

# 运行实验
python -m src.cli run

# 评测结果
python -m src.cli eval <run_name>

# 查看摘要
python -m src.cli show <run_name>
```

## 项目结构

```
llm-prompt-lab/
├── config/
│   ├── experiment.yaml      # 实验配置（模型、prompt、数据集、评测）
│   └── prompts/             # Prompt 模板（.md 文件）
│       ├── default.md       # 默认 system prompt
│       └── judge-default.md # Judge 评分标准
├── data/                    # 原始 Excel 数据集
├── src/
│   ├── config.py            # Pydantic 配置加载与校验
│   ├── dataset.py           # Excel 读取 + Jinja2 动态模板替换
│   ├── models.py            # OpenAI 兼容接口客户端
│   ├── experiment.py        # 实验运行器（断点续跑）
│   ├── evaluator.py         # LLM-as-Judge 评测
│   ├── importer.py          # 从 Excel 导入现网数据
│   └── cli.py               # CLI 入口
└── results/
    └── {run_name}/          # 实验结果
        ├── meta.json        # 实验配置快照
        ├── responses.jsonl  # 模型逐条响应（含完整请求参数）
        ├── scores.jsonl     # Judge 逐条评分
        └── summary.json     # 评测汇总统计
```

## 配置指南

### 实验配置 (`config/experiment.yaml`)

所有配置集中在一个文件中，每次修改后运行实验即可：

```yaml
# 模型配置
model:
  provider: openai
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.7
  max_tokens: 1024

# Prompt（引用 config/prompts/ 下的 .md 文件名）
prompt: default

# 数据集路径
dataset: data/example.xlsx

# 评测配置（可选）
judge:
  model:
    provider: openai
    model: qwen3.7-max
    base_url: https://api.deepseek.com/v1
    api_key_env: OPENAI_API_KEY
  prompt: judge-default     # 引用 config/prompts/judge-default.md
  dimensions: [relevance, factuality, fluency, structure, overall]
```

`api_key_env` 指定从哪个环境变量读取 API Key，留空表示无需认证。

### Prompt 模板 (`config/prompts/`)

Prompt 以 `.md` 文件存放，文件名即为 prompt 名称。实验的 `prompt` 字段引用文件名（不含 `.md`），框架自动加载文件内容，在运行时通过 `{{ system_prompt }}` 注入到 API 请求中。

数据集的 `api_json` 字段中可使用 Jinja2 占位符：
- `{{ system_prompt }}`：自动注入 `prompt` 对应的 .md 文件内容
- `{{ query }}`：自动注入当前行的用户问题

## 数据集格式

Excel 文件需包含以下两列：

| query | api_json |
|-------|----------|
| 什么是机器学习？ | `{"messages": [{"role": "system", "content": "{{system_prompt}}"}, {"role": "user", "content": "{{query}}"}]}` |

- **query**：用户问题文本
- **api_json**：完整的 API 请求 JSON 字符串，支持 Jinja2 占位符（`{{ system_prompt }}`、`{{ query }}` 等）

运行时框架会自动将占位符替换为模板和数据集中的实际值，无需预先生成中间文件。

## CLI 命令

| 命令 | 说明 |
|------|------|
| `python -m src.cli list` | 列出当前实验配置与已有 run |
| `python -m src.cli run [--name <run>]` | 运行实验（自动断点续跑，不指定则自动生成） |
| `python -m src.cli eval <run>` | LLM-as-Judge 评测 |
| `python -m src.cli import <excel> --name <run>` | 从 Excel 导入已有数据 |
| `python -m src.cli show <run>` | 查看实验结果摘要 |

### 导入现网数据

如果你有一批现网数据（Excel 格式，包含 Query 和模型回答），可以直接导入并评测，无需重新调用模型 API：

```bash
# 导入数据（Excel 需包含 query 和 response 列）
python -m src.cli import data/production_data.xlsx --name prod-eval-20240530

# 如果列名不同，可自定义
python -m src.cli import data/production_data.xlsx --name prod-eval-20240530 \
    --query-col "用户问题" --response-col "模型回答"

# 评测导入的数据
python -m src.cli eval prod-eval-20240530
```

导入命令会创建 `results/<run_name>/` 目录，生成 `responses.jsonl` 和 `meta.json`，之后即可像正常实验一样进行评测。

## 断点续跑

实验运行中如果被中断（Ctrl+C、网络故障等），重新执行相同命令即可从断点继续：

```
[resume] 12/50 already done, resuming...
████████████████░░░░░░░░░░ 50/50 [done]
```

断点续跑通过 `responses.jsonl` 中已有的记录自动派生，无需额外的状态文件。

## 评测

评测采用 LLM-as-Judge 模式：用另一个模型对实验结果逐条打分。逐条评分写入 `scores.jsonl`，汇总统计写入 `summary.json`。

### 评分维度

默认评测五个维度（每个维度 1-5 分），可在 `experiment.yaml` 的 `dimensions` 中自定义：

| 维度 | 说明 |
|------|------|
| relevance（相关性） | 回复是否紧扣用户问题 |
| factuality（事实性） | 信息是否准确可靠 |
| fluency（流畅性） | 语言表达是否自然通顺 |
| structure（结构化） | 回复组织是否清晰合理 |
| overall（综合评分） | 整体质量评价 |

### 稳定性保障

为保证跨次评测的可比性：
- 调用 Judge 模型时固定 `temperature=0` + `seed` 参数
- Judge prompt 作为 system 消息（评分标准），待评测内容作为 user 消息，支持 prompt cache 复用
- 要求 Judge 先逐维度分析再给出评分（chain-of-thought），减少随机性

评测本身也支持断点续评 — 已评分的条目不会重复调用 Judge 模型。