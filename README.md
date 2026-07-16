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

# 评测结果（从 eval.yaml 读取评测配置，自动生成 HTML 报告和 Excel 导出）
python -m src.cli eval

# 评测配置变更后强制重新评测
python -m src.cli eval --force

# 查看摘要
python -m src.cli show <run_name>
```

## 项目结构

```
llm-prompt-lab/
├── config/
│   ├── experiment.yaml                    # 实验配置（profiles + dataset）
│   ├── eval.yaml                          # 评测配置（judge 模型 + prompt + dimensions）
│   ├── advise.yaml                        # 优化建议配置（建议模型 + prompt + 低分采样）
│   └── prompts/                           # Prompt 模板文件
│       ├── candidate-prompt.md            # 被测模型 system prompt
│       ├── judge-prompt.md                # Judge 评分标准
│       └── advise-prompt.md               # 建议模型 system prompt
├── data/                                  # 数据集目录（每个数据集一个子目录）
│   └── example/                           # 一个数据集 = 一个包
│       ├── example.jsonl|.xlsx            # 数据集本体（与目录同名；.jsonl 优先）
│       └── dataset-prompt-template.md     # 可选：生产模板，用于反推 context
├── raw_data/                              # 原始日志数据
├── scripts/
│   └── import_summarybox.py               # Summarybox 日志导入工具
├── src/
│   ├── cli.py                             # CLI 入口
│   ├── config.py                          # Pydantic 配置加载与校验
│   ├── constants.py                       # 共享常量（RESULTS_DIR 等）
│   ├── dataset.py                         # Excel/JSONL 读取 + messages 构建
│   ├── prompt_rewriter.py                 # 从渲染串反推 context + jinja2 重渲染
│   ├── advisor.py                         # 读取 run 结果，由模型给出 System Prompt 优化建议
│   ├── evaluator.py                       # LLM-as-Judge 评测
│   ├── experiment.py                      # 实验运行器（断点续跑）
│   ├── importer.py                        # 从 Excel/JSONL 导入现网数据
│   ├── models.py                          # OpenAI 兼容接口客户端
│   ├── reporter.py                        # HTML 报告生成与 Excel 导出
│   └── templates/                         # HTML 模板
│       └── report.html                    # 报告模板（Chart.js 图表 + 交互表格）
└── results/
    └── {run_name}/                        # 每次实验的输出
        ├── meta.json                      # 实验配置快照（candidate、prompt、dataset）
        ├── eval_meta.json                 # 评测配置快照（judge 模型、prompt、dimensions）
        ├── responses.jsonl                # 模型逐条响应
        ├── scores.jsonl                   # Judge 逐条评分
        ├── summary.json                   # 评测汇总统计
        ├── analysis_by_human.xlsx         # 人工评估结果（可选，用于校准）
        ├── report.html                    # HTML 可视化报告
        └── report.xlsx                    # Excel 导出文件
```

## 配置指南

### 实验配置 (`config/experiment.yaml`)

采用多 Profile 模式：`dataset` 为所有 Profile 共享，`profiles` 节定义多个完整的被测模型配置。

```yaml
# ── 共享配置 ──────────────────────────────────────────────────────
# dataset 是数据集"目录名"。约定结构：
#   data/<dataset>/<dataset>.jsonl|.xlsx       数据集本体（.jsonl 优先，无长度上限）
#   data/<dataset>/dataset-prompt-template.md  可选 candidate prompt 模板（jinja2）
dataset: example

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
    prompt: candidate-prompt.md

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
    prompt: candidate-prompt.md

  default-stream:
    candidate:
      provider: ali
      model: deepseek-v4-flash
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.7
      max_tokens: 1024
      stream: true                       # 启用流式调用
    prompt: candidate-prompt.md

  default-proxy:
    candidate:
      provider: ali
      model: deepseek-v4-flash
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      api_key_env: CANDIDATE_OPENAI_API_KEY
      temperature: 0.7
      max_tokens: 1024
      use_proxy: true                    # 使用代理（从环境变量 PROXY_URL 读取）
    prompt: candidate-prompt.md
```

### 评测配置 (`config/eval.yaml`)

评测配置独立于实验配置，`eval` 命令运行时实时加载，可随时修改后重新评测。

```yaml
# 评测配置 — LLM-as-Judge
model:
  provider: ali
  model: qwen3.7-max
  base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
  api_key_env: JUDGE_OPENAI_API_KEY
  max_tokens: 4096
prompt: judge-prompt.md
dimensions: ["relevance", "factuality", "fluency", "structure", "timeliness", "localization", "search_planning", "search_relevance", "search_utilization", "overall"]
```

### 优化建议配置 (`config/advise.yaml`)

独立于 `experiment.yaml` / `eval.yaml`，`advise` 命令运行时实时加载。配置建议模型及其 system prompt，以及低分样本的选取参数。

```yaml
model:
  provider: ali
  model: qwen3.7-max
  base_url: https://...
  api_key_env: ADVISE_OPENAI_API_KEY
  temperature: 0.3          # 偏低，保证建议稳定可复现
  max_tokens: 8192
prompt: advise-prompt.md

# 低分样本选取：overall <= low_score_threshold 视为低分；
# 不足 min_low_samples 时按 overall 升序补足；总样本上限 max_samples。
low_score_threshold: 3
min_low_samples: 5
max_samples: 20
```

`advise` 命令读取指定 run 的评测结果（`scores.jsonl` / `summary.json` / `meta.json` 中的当前 system prompt），选取低分样本，连同评分统计一起喂给建议模型，产出**诊断 + 修订版 system prompt + 修改理由**，写入 `results/<run>/advise.md`。当前优化范围仅限 System Prompt。

### 核心概念

| 概念 | 说明 |
|------|------|
| **candidate** | 被测模型 -- 正在实验的模型及其参数，配置在 `experiment.yaml` |
| **judge** | 评测模型 -- 用于对 candidate 的输出打分，配置在 `eval.yaml` |
| **profile** | 一组完整的 candidate 配置 + prompt，通过 `--profile` 切换 |

`candidate` 和 `judge` 各自独立配置 `api_key_env`，可以使用不同供应商的 API Key。两者配置文件完全独立，互不影响。

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
prompt: candidate-prompt.md
dataset: example
```

### Prompt 模板 (`config/prompts/`)

Prompt 以文件形式存放，支持 `.md` 和 `.txt` 扩展名。配置中的 `prompt` 字段必须写完整文件名（含扩展名），否则报错。

`prompt` 字段指向的文件即 candidate 的 system prompt。两种使用方式：

- **静态注入**（默认）：文件内容整段当作 system message 注入到每行样本，覆盖 `api_json[system].content`。
- **反推 + 重渲染**（启用 `dataset-prompt-template.md` 时）：文件视为 jinja2 模板，框架从每行 `api_json[system]` 反推出 context 并逐行渲染。详见下文 [数据集自带 prompt 模板](#数据集自带-prompt-模板)。

## 数据集格式

每个数据集是 `data/` 下的一个**目录**，约定结构：

```
data/<dataset>/
├── <dataset>.jsonl|.xlsx           # 数据集本体（与目录同名；.jsonl 优先）
└── dataset-prompt-template.md      # 可选：候选 prompt 模板，启用反推+重渲染
```

`experiment.yaml` 的 `dataset` 字段写**目录名**（如 `dataset: example`），框架按约定派生出数据集文件与 template 路径。数据集文件支持 `.jsonl`（每行一条 JSON 记录，字段无长度上限，**推荐用于超长 `api_json`**）与 `.xlsx`（Excel 单元格上限 32767 字符）；两者都存在时优先 `.jsonl`。

### 列约定

必选列：

| 列名 | 说明 |
|------|------|
| `query` | 用户问题文本 |
| `api_json` | 完整的 messages（生产环境真实请求体），形如 `{"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}]}` 或裸 messages 数组。**Excel 中须为 JSON 字符串**；**JSONL 中可为 JSON 字符串或直接写对象/数组**（框架自动归一化） |

运行时框架解析 `api_json` 取 messages，把 system 消息的 `content` 替换为当前 profile 的 candidate prompt（静态注入或反推渲染后的结果），其余消息（user / assistant / 多轮历史）保持原样发送给模型。

> **超长 api_json**：生产数据 `api_json`（含多轮历史、搜索结果等）可能超过 Excel 单元格 32767 字符上限。改用 `.jsonl` 数据集即可完整承载，不受限制。`export` 导出 Excel 时，超长文本字段（response、reasoning、search_results 等）会自动截断并标注 `[TRUNCATED]`，完整数据仍保留在 `responses.jsonl`。

可选列：

| 列名 | 说明 |
|------|------|
| `language` | 用户语言（如 `zh-CN`），自动追加到 system prompt 末尾，用于本地化评测 |
| `location` | 用户位置（如 `China`），自动追加到 system prompt 末尾，用于本地化评测 |

提供 `language` 和 `location` 时，会在 system prompt 末尾自动追加：

```
当前用户信息：
语言：zh-CN
位置：China
```

### 数据集自带 prompt 模板

如果数据集的 `api_json[system].content` 是由生产环境的某份 jinja2 模板渲染而成的，你可以把那份模板放在数据集目录下命名为 `dataset-prompt-template.md`，框架会自动启用 **反推 + 重渲染** 路径：

1. 用 jinja2 AST 把 `dataset-prompt-template.md` 拆成 `[字面量, 变量, ...]`，转成正则匹配每行的 `api_json[system].content`，反推出 context 字典
2. 把 `prompt` 字段指向的文件（如 `candidate-prompt.md`）当作**实验侧 jinja2 模板**，用上一步反推的 context 渲染出当前行的 system prompt

这样可以在保留每条样本真实上下文（用户类型、城市、tone 等）的前提下，仅替换 prompt 主体做对照实验。

**约束**：两份模板都只能含 `{{ var }}` 纯变量替换，不支持 filter / `{% if %}` / `{% for %}`。两边变量名必须**完全一致**。

**示例**：

```jinja2
{# data/example/dataset-prompt-template.md（与生产渲染时所用模板一致）#}
你是一个智能助手。
当前用户类型：{{ user_type }}
所在城市：{{ city }}
请用 {{ tone }} 的口吻回答问题。
```

```jinja2
{# config/prompts/candidate-prompt.md（实验候选模板，变量名一致）#}
你扮演一名{{ tone }}的 AI 助手，正在为身份为「{{ user_type }}」、所在「{{ city }}」的用户服务。
回答应当兼顾准确性与友好度，避免冗长。
```

**控制开关**：
- 默认 `use_dataset_prompt_template: true`，存在 `dataset-prompt-template.md` 就启用，不存在则退回静态注入
- 在 profile 下加 `use_dataset_prompt_template: false` 可强制禁用（即使文件存在）

**失败处理**：某行反推失败（模板与渲染串不匹配 / 含不支持节点 / 候选模板含 context 没有的变量）时，会 `tqdm.write` 一条 warn 并降级为模板原文，实验继续跑完，不会因个别行挂掉整批。

## CLI 命令

```bash
# 运行实验（自动断点续跑）
python -m src.cli run                        # 使用 default profile
python -m src.cli run --profile think        # 使用 think profile
python -m src.cli run --name my-exp          # 自定义 run 名称

# 评测实验结果（LLM-as-Judge，从 eval.yaml 实时加载评测配置）
# 省略 run_name 时自动评测最新实验，完成后自动生成 HTML 报告和 Excel 导出
python -m src.cli eval
python -m src.cli eval <run_name>
python -m src.cli eval --force               # 评测配置变更后强制重新评测

# 生成 HTML 可视化报告（省略 run_name 时使用最新实验）
python -m src.cli report
python -m src.cli report <run_name>
python -m src.cli report --no-open           # 不自动打开浏览器

# 导出 Excel 文件（省略 run_name 时使用最新实验）
python -m src.cli export
python -m src.cli export <run_name>

# 对比人工评分与 Judge 评分（省略 run_name 时使用最新实验）
python -m src.cli calibrate
python -m src.cli calibrate <run_name>

# 读取 run 结果，由大模型给出 System Prompt 优化建议（须先 eval）
python -m src.cli advise
python -m src.cli advise <run_name>

# 从 Excel/JSONL 导入现网数据用于评测
python -m src.cli import <data> --name <run_name>
python -m src.cli import data/prod.xlsx --name prod-20240530 \
    --query-col "用户问题" --response-col "模型回答"
# 超长 api_json 用 JSONL 导入（不受 Excel 32767 限制）
python -m src.cli import data/prod.jsonl --name prod-20240530

# 查看实验结果摘要
python -m src.cli show <run_name>
```

| 命令 | 说明 |
|------|------|
| `run` | 运行实验（支持断点续跑） |
| `eval` | LLM-as-Judge 评测（从 eval.yaml 读取配置，支持 `--force`） |
| `report` | 生成 HTML 可视化报告 |
| `export` | 导出 Excel 文件 |
| `calibrate` | 对比人工评分与 Judge 评分 |
| `advise` | 读取 run 结果，由大模型给出 System Prompt 优化建议（从 advise.yaml 读取配置） |
| `import` | 从 Excel/JSONL 导入现网数据 |
| `show` | 查看结果摘要 |

`run` 支持 `--profile` / `-p` 参数选择 profile。`eval` 支持 `--force` 参数强制覆盖已有评测结果。`eval`、`report`、`export`、`calibrate`、`advise` 和 `show` 支持省略 `run_name`，自动使用最新实验（按文件夹修改时间排序）。`advise` 要求该 run 已评测（存在 `scores.jsonl`）。

### 导入现网数据

如果已有现网数据（Excel 或 JSONL 格式，含 Query 和模型回答），可直接导入评测，无需重新调用模型 API：

```bash
# 导入（数据文件需包含 query 和 response 列）
python -m src.cli import data/production.xlsx --name prod-eval
# 或 JSONL（api_json 超长时推荐，不受 Excel 32767 字符限制）
python -m src.cli import data/production.jsonl --name prod-eval

# 评测导入的数据
python -m src.cli eval prod-eval
```

导入命令在 `results/<run_name>/` 下生成 `responses.jsonl` 和 `meta.json`（标记 `"source": "imported"`），之后可像正常实验一样评测。`api_json` 在 JSONL 中可写成 JSON 字符串或直接对象/数组，框架自动归一化。

### 从 Summarybox 日志导入

如果数据来源是 Summarybox 日志（`raw_data/summarybox_log.xlsx`），可使用 `scripts/import_summarybox.py` 将日志中的 QA 记录转换为标准数据集格式：

```bash
# 默认输出到 data/summarybox_import.xlsx
python scripts/import_summarybox.py

# 指定输出名
python scripts/import_summarybox.py my_dataset

# 指定输入路径
python scripts/import_summarybox.py my_dataset --input raw_data/summarybox_log.xlsx

# 生成后导入到 eval
python -m src.cli import my_dataset
```

脚本会筛选 `classification == "QA"` 的行，解码多层嵌套的 `log_json`（含 HTML unescape 与截断修复），提取 `query`、`answer`（→ `response`）、`prompt`（→ `api_json`）三列，输出与 `importer.py` 完全对齐的标准数据集。

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

默认评测十个维度（每个维度 1-5 分），可在 `eval.yaml` 的 `dimensions` 中自定义：

| 维度 | 说明 |
|------|------|
| relevance（相关性） | 回复是否紧扣用户问题 |
| factuality（事实性） | 信息是否准确可靠 |
| fluency（流畅性） | 语言表达是否自然通顺 |
| structure（结构化） | 回复组织是否清晰合理 |
| timeliness（实时性） | 回复是否考虑了时间敏感性（新闻、金价、天气等） |
| localization（本地化） | 回复是否适配了用户的语言和位置偏好 |
| search_planning（搜索规划） | 搜索关键词是否合理、能否有效服务于用户 Query |
| search_relevance（搜索结果相关性） | 搜索结果是否贴合 Query，是否包含无关结果，是否足以支撑回答 |
| search_utilization（搜索结果利用） | 模型对搜索结果的筛选判断与利用质量 |
| overall（综合评分） | 整体质量评价 |

评分标准详见 `config/prompts/judge-prompt.md`。

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

### 评测配置与断点续评

评测配置（Judge 模型、评分 prompt、评分维度）从 `eval.yaml` 实时加载，不依赖 `experiment.yaml` 或 `meta.json` 中的快照。这意味着修改评测配置后可以直接重新评测同一个 run，无需手动修改结果目录。

`eval` 命令通过 `eval_meta.json` 记录当次评测的配置 hash，实现智能断点续评：

- **配置未变（hash 相同）**：自动从断点继续，已评分条目不会重复调用 Judge 模型
- **配置变更（hash 不同）**：提示用户手动删除旧结果后重试，或使用 `--force` 强制覆盖
- **无历史记录**：正常开始全新评测

```bash
# 配置变更后强制重新评测
python -m src.cli eval --force

# 输出示例：
[force] 评测配置已变更 (hash: a1b2c3d4 → e5f6g7h8)，清空旧结果
```

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
- **Responses**：逐条响应数据 + 评分数据合并在一起，按 `row_index` 对齐（query、response、reasoning、token 用量、延迟等在前，各维度分数与分析追加在右），方便在同一张表里筛选排序

### 人工评估校准

如果对 Judge 评分有疑虑，可以进行人工评估并与 Judge 评分对比：

1. 在 `results/<run_name>/` 下创建 `analysis_by_human.xlsx`，包含三列：
   - `row_index`：与 `responses.jsonl` 中的行号对应
   - `query`：用户问题原文
   - `analysis_by_human`：人工评估意见（自由文本）

2. 运行校准命令，自动追加 Calibration 页到 report.xlsx：

```bash
python -m src.cli calibrate
python -m src.cli calibrate <run_name>
```

Calibration 页并排展示每条数据的 Judge 各维度评分与人工评估意见，方便逐条对比差异。

## Run 名称生成

Run 名称由配置自动生成，格式为 `{dataset_stem}@{provider}@{model}@[{profile}@]{prompt}@{hash}`，使用 `@` 分隔各部分。稳定的部分在前（数据集 > 服务商 > 模型 > profile > prompt），方便目录浏览时同类实验自然聚集。例如：

```
example@ali@deepseek-v4-flash@default@candidate-prompt.md@0c3361be
```

相同配置产生相同名称，因此相同命令天然支持断点续跑。
