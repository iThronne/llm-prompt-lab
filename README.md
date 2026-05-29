# LLM Prompt Lab

大模型 API 实验框架 — 支持不同模型、Prompt 模板、数据集和参数的系统化实验，内置结果记录、LLM-as-Judge 评测与断点续跑。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
export OPENAI_API_KEY=sk-xxx

# 运行实验
python -m src.cli run example

# 评测结果
python -m src.cli eval example

# 查看摘要
python -m src.cli show example
```

## 项目结构

```
llm-prompt-lab/
├── config/
│   ├── models.yaml          # 模型配置（API endpoint、认证方式）
│   ├── templates.yaml       # Prompt 模板（Jinja2 语法）
│   └── experiments.yaml     # 实验定义（模型 × 模板 × 数据集 × 参数）
├── data/                    # 原始 Excel 数据集
├── src/
│   ├── config.py            # Pydantic 配置加载与校验
│   ├── dataset.py           # Excel 读取 + Jinja2 动态模板替换
│   ├── models.py            # OpenAI 兼容接口客户端
│   ├── experiment.py        # 实验运行器（SQLite 断点续跑）
│   ├── evaluator.py         # LLM-as-Judge 评测
│   └── cli.py               # CLI 入口
└── results/
    └── {experiment}/        # 实验结果
        ├── checkpoint.db    # 断点续跑状态
        ├── results.jsonl    # 实验输出
        └── evaluation.json  # 评测结果
```

## 配置指南

### 1. 模型配置 (`config/models.yaml`)

```yaml
models:
  gpt-4o:
    provider: openai
    model: gpt-4o
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY

  deepseek-chat:
    provider: openai
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY

  qwen2.5-local:           # Ollama 本地模型
    provider: openai
    model: qwen2.5:7b
    base_url: http://localhost:11434/v1
    api_key_env: ""
```

`api_key_env` 指定从哪个环境变量读取 API Key，留空表示无需认证。

### 2. Prompt 模板 (`config/templates.yaml`)

```yaml
templates:
  chinese-expert:
    system_prompt: "你是一位资深的中文技术专家，请用专业、准确的中文回答用户问题。"

  concise:
    system_prompt: "Be concise. Answer in 1-2 sentences."
```

模板变量 `{{ system_prompt }}` 在运行时自动注入。数据集的 `api_json` 字段中还可使用 `{{ query }}` 引用当前行的用户问题。

### 3. 实验定义 (`config/experiments.yaml`)

```yaml
experiments:
  my-experiment:
    model: gpt-4o-mini          # 引用 models.yaml 中的模型
    template: default           # 引用 templates.yaml 中的模板
    dataset: data/my_data.xlsx  # 数据集路径
    params:
      temperature: 0.7
      max_tokens: 1024
    judge:                      # 评测配置（可选）
      model: gpt-4o-mini        # Judge 模型
      prompt: |
        Score the response on accuracy (1-5), completeness (1-5), clarity (1-5).
        Question: {{ query }}
        Response: {{ response }}
        Output ONLY JSON: {"accuracy": X, "completeness": X, "clarity": X}
      dimensions: [accuracy, completeness, clarity]
```

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
| `python -m src.cli list` | 列出所有实验定义 |
| `python -m src.cli run <name>` | 运行实验（自动断点续跑） |
| `python -m src.cli eval <name>` | LLM-as-Judge 评测 |
| `python -m src.cli show <name>` | 查看实验结果摘要 |

## 断点续跑

实验运行中如果被中断（Ctrl+C、网络故障等），重新执行相同命令即可从断点继续：

```
[resume] 12/50 already done, resuming...
████████████████░░░░░░░░░░ 50/50 [done]
```

每个实验的进度状态存储在 `results/{experiment}/checkpoint.db`（SQLite）中。

## 评测

评测采用 LLM-as-Judge 模式：用另一个模型对实验结果逐条打分。评分维度可在实验配置中自定义（如准确性、完整性、流畅性）。评测结果包含每条分数和汇总统计，写入 `results/{experiment}/evaluation.json`。

Judge prompt 支持 `{{ query }}` 和 `{{ response }}` 模板变量，框架自动注入对应内容。评测本身也支持断点续评 — 已评分的条目不会重复调用 Judge 模型。