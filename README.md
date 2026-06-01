# LLM Prompt Lab

大模型 API 实验框架 -- 支持多 Profile 切换、Prompt 模板、数据集和参数的系统化实验，内置 LLM-as-Judge 评测、流式调用、代理支持、HTML 可视化报告与断点续跑。

## 快速开始

```bash
# 安装依赖
poetry install
poetry shell

# 配置 API Key（candidate 和 judge 可分别配置）
export CANDIDATE_OPENAI_API_KEY=sk-xxx
export JUDGE_OPENAI_API_KEY=sk-xxx

# 运行实验（默认 default profile）
python -m src.cli run

# 运行指定 profile 的实验
python -m src.cli run --profile think

# 评测结果（自动生成 HTML 报告和 Excel 导出）
python -m src.cli eval

# 查看摘要
python -m src.cli show
```

## 项目结构

```
llm-prompt-lab/
├── config/
│   ├── experiment.yaml                    # 实验配置（profiles + judge + dataset）
│   └── prompts/                           # Prompt 模板文件
│       ├── candidate-system-prompt-default.md   # 被测模型 system prompt
│       └── judge-system-prompt-default.md        # Judge 评分标准
├── data/                                  # 数据集（Excel）
├── src/
│   ├── cli.py                             # CLI 入口
│   ├── config.py                          # Pydantic 配置加载与校验
│   ├── constants.py                       # 共享常量（RESULTS_DIR 等）
│   ├── dataset.py                         # Excel 读取 + Jinja2 模板替换
│   ├── evaluator.py                       # LLM-as-Judge 评测
│   ├── experiment.py                      # 实验运行器（断点续跑）
│   ├── importer.py                        # 从 Excel 导入现网数据
│   ├── models.py                          # OpenAI 兼容接口客户端
│   ├── reporter.py                        # HTML 报告生成与 Excel 导出
│   └── templates/                         # HTML 模板
│       └── report.html                    # 报告模板（Chart.js 图表 + 交互表格）
└── results/
    └── {run_name}/                        # 每次实验的输出
        ├── meta.json                      # 配置快照（含 profile、模型参数、prompt 全文）
        ├── responses.jsonl                # 模型逐条响应
        ├── scores.jsonl                   # Judge 逐条评分
        ├── summary.json                   # 评测汇总统计
        ├── report.html                    # HTML 可视化报告
        └── report.xlsx                    # Excel 导出文件
```

## 配置指南

### 实验配置 (`config/experiment.yaml`)

采用多 Profile 模式：`dataset` 和 `judge` 为所有 Profile 共享，`profiles` 节定义多个完整的被测模型配置。

```yaml
# ── 共享配置 ──────────────────────────────────────────────────────
dataset: data/example.xlsx

judge:
  model:
    provider: ali
    model: qwen3.7-max
    base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    api_key_env: JUDGE_OPENAI_API_KEY
  prompt: judge-system-prompt-default.md
  dimensions: ["relevance", "factuality", "fluency", "structure", "timeliness", "localization", "overall"]

# ── Profiles ─────────────────────────────────────────────────────
profiles:
  default:
    candidate:
      provider: ali
      model: deepseek-v4-flash
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.7
      max_tokens: 1024
    prompt: candidate-system-prompt-default.md

  think:
    candidate:
      provider: ali
      model: deepseek-r1
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.3
      max_tokens: 8192
      extra_body:                        # 非标准 API 参数
        chat_template_kwargs:
          enable_thinking: true
    prompt: candidate-system-prompt-default.md

  default-stream:
    candidate:
      provider: ali
      model: deepseek-v4-flash
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.7
      max_tokens: 1024
      stream: true                       # 启用流式调用
    prompt: candidate-system-prompt-default.md

  default-proxy:
    candidate:
      provider: ali
      model: deepseek-v4-flash
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.7
      max_tokens: 1024
      use_proxy: true                    # 使用代理（从环境变量 PROXY_URL 读取）
    prompt: candidate-system-prompt-default.md
```

#### 核心概念

| 概念 | 说明 |
|------|------|
| **candidate** | 被测模型 -- 正在实验的模型及其参数 |
| **judge** | 评测模型 -- 用于对 candidate 的输出打分（LLM-as-Judge） |
| **profile** | 一组完整的 candidate 配置 + prompt，通过 `--profile` 切换 |

`candidate` 和 `judge` 各自独立配置 `api_key_env`，可以使用不同供应商的 API Key。

#### 非标准 API 参数

如果模型 API 不遵循 OpenAI 规范（例如需要在请求体顶层传 `chat_template_kwargs`），将其放在 `extra_body` 下。OpenAI SDK 会将 `extra_body` 的内容展开到 HTTP 请求体顶层，与 `model`、`messages` 等标准参数并列：

```yaml
candidate:
  model: deepseek-r1
  temperature: 0.3
  max_tokens: 8192
  extra_body:
    chat_template_kwargs:
      enable_thinking: true
```

#### 额外采样参数

通过 `ConfigDict(extra="allow")`，模型配置支持任意额外参数（如 `top_p`、`min_p`、`top_k` 等），自动透传给 API：

```yaml
candidate:
  model: deepseek-v4-flash
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.9
  min_p: 0.1
```

#### 流式调用

启用 `stream: true` 后，模型逐 token 返回响应，适合长文本生成场景。流式调用额外记录：

- `ttft_ms`：首 token 延迟（Time To First Token）
- `reasoning_content`：思考模型的推理过程（如果存在）

#### 代理支持

启用 `use_proxy: true` 后，从环境变量 `PROXY_URL` 读取代理地址，创建 `httpx.AsyncClient` 并禁用 SSL 验证：

```bash
# 配置代理地址
export PROXY_URL=http://127.0.0.1:7890
```

#### 单配置模式

如果不需要多 Profile，可以不写 `profiles` 节，直接将完整配置放在顶层：

```yaml
candidate:
  provider: ali
  model: deepseek-v4-flash
  base_url: https://...
  api_key_env: CANDIDATE_OPENAI_API_KEY
  temperature: 0.7
  max_tokens: 1024
prompt: candidate-system-prompt-default.md
dataset: data/example.xlsx
judge:
  model: ...
  prompt: judge-system-prompt-default.md
```

### Prompt 模板 (`config/prompts/`)

Prompt 以文件形式存放，支持 `.md` 和 `.txt` 扩展名。配置中的 `prompt` 字段必须写完整文件名（含扩展名），否则报错。

数据集的 `api_json` 字段中使用 Jinja2 占位符引用 prompt：

- `{{ system_prompt }}` -- 自动注入 `prompt` 字段指向的文件内容
- `{{ query }}` -- 自动注入当前数据行的用户问题

## 数据集格式

Excel 文件需包含以下两列：

| 列名 | 说明 |
|------|------|
| `query` | 用户问题文本 |
| `api_json` | 完整的 API 请求 JSON 字符串，支持 Jinja2 占位符 |

可选列：

| 列名 | 说明 |
|------|------|
| `language` | 用户语言（如 `zh-CN`），用于本地化评测 |
| `location` | 用户位置（如 `China`），用于本地化评测 |

`api_json` 示例：

```json
{"messages": [{"role": "system", "content": "{{system_prompt}}"}, {"role": "user", "content": "{{query}}"}]}
```

运行时框架自动将 `{{system_prompt}}` 和 `{{query}}` 替换为实际值，与 `candidate` 的 `call_params` 合并后发送请求。

如果提供了 `language` 和 `location` 列，框架会自动将其注入到 `system_prompt` 末尾：

```
当前用户信息：
语言：zh-CN
位置：China
```

## CLI 命令

```bash
# 列出可用 profiles 和已有实验 run
python -m src.cli list
python -m src.cli list --profile think       # 预览指定 profile 的配置

# 运行实验（自动断点续跑）
python -m src.cli run                        # 使用 default profile
python -m src.cli run --profile think        # 使用 think profile
python -m src.cli run --name my-exp          # 自定义 run 名称

# 评测实验结果（LLM-as-Judge，从 meta.json 读取配置，不依赖当前 experiment.yaml）
# 省略 run_name 时自动评测最新实验，完成后自动生成 HTML 报告和 Excel 导出
python -m src.cli eval
python -m src.cli eval <run_name>

# 生成 HTML 可视化报告（省略 run_name 时使用最新实验）
python -m src.cli report
python -m src.cli report <run_name>
python -m src.cli report --no-open           # 不自动打开浏览器

# 导出 Excel 文件（省略 run_name 时使用最新实验）
python -m src.cli export
python -m src.cli export <run_name>

# 从 Excel 导入现网数据用于评测
python -m src.cli import <excel> --name <run_name>
python -m src.cli import data/prod.xlsx --name prod-20240530 \
    --query-col "用户问题" --response-col "模型回答"

# 查看实验结果摘要
python -m src.cli show <run_name>
```

| 命令 | 说明 |
|------|------|
| `list` | 列出 profiles 和已有 run |
| `run` | 运行实验（支持断点续跑） |
| `eval` | LLM-as-Judge 评测（自动生成报告和导出） |
| `report` | 生成 HTML 可视化报告 |
| `export` | 导出 Excel 文件 |
| `import` | 从 Excel 导入现网数据 |
| `show` | 查看结果摘要 |

`run` 和 `list` 支持 `--profile` / `-p` 参数选择 profile。`eval`、`report`、`export` 和 `show` 支持省略 `run_name`，自动使用最新实验（按文件夹修改时间排序）。

### 导入现网数据

如果已有现网数据（Excel 格式，含 Query 和模型回答），可直接导入评测，无需重新调用模型 API：

```bash
# 导入（Excel 需包含 query 和 response 列）
python -m src.cli import data/production.xlsx --name prod-eval

# 评测导入的数据
python -m src.cli eval prod-eval
```

导入命令在 `results/<run_name>/` 下生成 `responses.jsonl` 和 `meta.json`（标记 `"source": "imported"`），之后可像正常实验一样评测。

## 断点续跑

实验或评测中断后，重新执行相同命令即可从断点继续：

```
[resume] 12/50 already done, resuming...
████████████████░░░░░░░░░░ 50/50 [done]
```

断点续跑通过扫描 `responses.jsonl` / `scores.jsonl` 中已有记录实现，无额外状态文件。run 名称由配置自动生成（相同配置产生相同名称），因此相同命令天然支持续跑。

## 评测

评测采用 LLM-as-Judge 模式：用 Judge 模型对 candidate 的输出逐条打分。

### 评分维度

默认评测七个维度（每个维度 1-5 分），可在 `experiment.yaml` 的 `dimensions` 中自定义：

| 维度 | 说明 |
|------|------|
| relevance（相关性） | 回复是否紧扣用户问题 |
| factuality（事实性） | 信息是否准确可靠 |
| fluency（流畅性） | 语言表达是否自然通顺 |
| structure（结构化） | 回复组织是否清晰合理 |
| timeliness（实时性） | 回复是否考虑了时间敏感性（新闻、金价、天气等） |
| localization（本地化） | 回复是否适配了用户的语言和位置偏好 |
| overall（综合评分） | 整体质量评价 |

评分标准详见 `config/prompts/judge-system-prompt-default.md`。

#### 实时性评测

用户问题中可能包含请求时间信息（如 `[Current Time: 2024-06-01 14:30:00 CST]`），Judge 会以此作为判断回复实时性的时间依据。对于不涉及时间敏感性的问题，默认给 5 分。

#### 本地化评测

如果数据集提供了 `language` 和 `location` 列，Judge 会评估回复是否适配了用户的语言和位置偏好（如中国推荐微信、美国推荐 Facebook、日本推荐 LINE）。对于未提供语言和位置信息的情况，默认给 5 分。

### 稳定性保障

- 调用 Judge 模型时固定 `temperature=0` + `seed` 参数
- Judge prompt 作为 system 消息，待评测内容作为 user 消息，支持 prompt cache
- 要求 Judge 先逐维度分析再给出评分（chain-of-thought），减少随机性
- 评测本身也支持断点续评，已评分条目不会重复调用 Judge 模型
- 失败时采用指数退避重试

### 评测独立性

评测配置（Judge 模型、评分 prompt、评分维度）在实验运行时冻结到 `meta.json`。执行 `eval` 时从快照读取，不依赖当前 `experiment.yaml` 的内容 -- 即使修改了配置或重命名了 prompt 文件，之前的实验仍可正常评测。

## 可视化报告

### HTML 报告

`eval` 命令完成后自动生成 HTML 报告（也可手动执行 `report` 命令），包含：

- **概览卡片**：总行数、平均延迟、平均首 token 延迟、总 token 数
- **图表区**：
  - 首 token 延迟 vs 总延迟散点图（如有流式数据）
  - 评分雷达图（显示各维度平均分）
  - Token 用量柱状图
- **数据表格**：
  - 可搜索、可分页
  - 点击行展开查看完整 query / response / reasoning / 评分分析
  - 支持按评分维度筛选和排序

### Excel 导出

`eval` 命令完成后自动生成 Excel 文件（也可手动执行 `export` 命令），包含多个 sheet：

- **Summary**：实验元信息 + 评分汇总
- **Responses**：逐条响应数据（query、response、reasoning、token 用量、延迟等）
- **Scores**：逐条评分数据（如有评测）

## Run 名称生成

Run 名称由配置自动生成，格式为 `{profile}@{provider}@{model}@{prompt}@{dataset_stem}@{hash}`，使用 `@` 分隔各部分。例如：

```
default@ali@deepseek-v4-flash@candidate-system-prompt-default.md@example@0c3361be
```

相同配置产生相同名称，因此相同命令天然支持断点续跑。
