# 常用命令

> 在 PyCharm 终端中直接复制粘贴运行。

---

## 运行实验

```bash
# 使用默认 profile 运行
python -m src.cli run

# 指定 profile 运行
python -m src.cli run --profile think

# 自定义名称
python -m src.cli run --name my-experiment

# 流式调用
python -m src.cli run --profile default-stream
```

## 评测实验结果

```bash
# 评测最新实验
python -m src.cli eval

# 评测指定实验
python -m src.cli eval <run_name>

# 并发评测（加速）
python -m src.cli eval --concurrency 4

# 指定实验 + 并发
python -m src.cli eval <run_name> --concurrency 4

# 强制重新评测（覆盖已有结果）
python -m src.cli eval <run_name> --force
```

## 导入现网数据

```bash
# 基本导入
python -m src.cli import data/production.xlsx --name prod-baseline

# 指定列名
python -m src.cli import data/production.xlsx --name prod-baseline --query-col query --response-col response
```

### 从 Summarybox 日志导入

```bash
# 将 summarybox 日志转为标准数据集
python scripts/import_summarybox.py

# 指定输出名
python scripts/import_summarybox.py my_dataset

# 生成后导入评测
python -m src.cli import my_dataset
```

## 查看结果

```bash
# 查看实验摘要
python -m src.cli show <run_name>
```

## 生成报告

```bash
# 最新实验的 HTML 报告（不自动打开浏览器）
python -m src.cli report --no-open

# 指定实验的 HTML 报告
python -m src.cli report <run_name>

# 启动交互报告，在评分分析旁直接向大模型流式追问
python -m src.cli report <run_name> --serve

# 指定本地服务端口
python -m src.cli report <run_name> --serve --port 8766

# 导出最新实验为 Excel
python -m src.cli export

# 导出指定实验为 Excel
python -m src.cli export <run_name>
```

## 人工评估校准

```bash
# 对比人工评分与 Judge 评分（需先在 run 目录下创建 analysis_by_human.xlsx）
python -m src.cli calibrate

# 指定实验
python -m src.cli calibrate <run_name>
```

`analysis_by_human.xlsx` 格式：包含 `row_index`、`query`、`analysis_by_human` 三列。运行 calibrate 后自动追加 Calibration 页到 report.xlsx。

## 优化建议

```bash
# 读取最新已评测 run 的结果，由大模型给出 System Prompt 优化建议
python -m src.cli advise

# 指定 run
python -m src.cli advise <run_name>
```

读取 run 的评测结果，抽取低分样本喂给大模型，输出 System Prompt 的优化建议。

- 依赖评分结果：未评测的 run 会报错，请先运行 `eval`。
- 配置文件：`config/advise.yaml`（模型、低分样本选取策略），修改后重新运行即生效，无需改 `experiment.yaml`。
- API Key：通过环境变量 `ADVISE_OPENAI_API_KEY` 读取。
- Prompt：`config/prompts/advise-prompt.md`。

## 典型工作流

```bash
# 完整流程：运行 → 评测 → 报告 → 人工校准 → 优化建议
python -m src.cli run --profile default
python -m src.cli eval --concurrency 4
python -m src.cli report --no-open
# 人工评估后：在 run 目录下创建 analysis_by_human.xlsx，然后运行
python -m src.cli calibrate
# 让大模型基于评分给出 System Prompt 优化建议
python -m src.cli advise

# 对比两个 profile
python -m src.cli run --profile default --name exp-default
python -m src.cli run --profile think --name exp-think
python -m src.cli eval exp-default --concurrency 4
python -m src.cli eval exp-think --concurrency 4
python -m src.cli export exp-default
python -m src.cli export exp-think
```
