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
```

## 导入现网数据

```bash
# 基本导入
python -m src.cli import data/production.xlsx --name prod-baseline

# 指定列名
python -m src.cli import data/production.xlsx --name prod-baseline --query-col query --response-col response

# 导入并指定 profile
python -m src.cli import data/production.xlsx --name prod-baseline --profile think
```

## 查看实验

```bash
# 列出所有实验
python -m src.cli list

# 列出指定 profile 的实验
python -m src.cli list --profile think

# 查看实验摘要
python -m src.cli show <run_name>
```

## 生成报告

```bash
# 最新实验的 HTML 报告（不自动打开浏览器）
python -m src.cli report --no-open

# 指定实验的 HTML 报告
python -m src.cli report <run_name>

# 导出最新实验为 Excel
python -m src.cli export

# 导出指定实验为 Excel
python -m src.cli export <run_name>
```

## 典型工作流

```bash
# 完整流程：运行 → 评测 → 报告
python -m src.cli run --profile default
python -m src.cli eval --concurrency 4
python -m src.cli report --no-open

# 对比两个 profile
python -m src.cli run --profile default --name exp-default
python -m src.cli run --profile think --name exp-think
python -m src.cli eval exp-default --concurrency 4
python -m src.cli eval exp-think --concurrency 4
python -m src.cli export exp-default
python -m src.cli export exp-think
```
