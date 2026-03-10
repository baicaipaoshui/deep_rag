# Deep RAG

Deep RAG 是一个面向企业知识库的检索增强问答系统，支持混合文档格式
（`.md`、`.pdf`、`.xlsx/.xlsm/.xls`），采用 Python 编排 + MCP 工具层 + 离线索引构建。

## 核心能力

- 五阶段状态机编排：
  `FILTERING -> LOCATING -> EXTRACTING -> VERIFYING -> GENERATING`
- 三层索引架构：
  导航索引（Navigation）+ 发现索引（Discovery）+ 证据索引（Evidence）
- 混合检索：
  Whoosh 关键词检索 + ChromaDB 向量检索 + RRF 融合
- 预算与稳定性控制：
  token 预算守护 + 证据不足自动补检轮次
- Web 交互：
  Streamlit 问答 + 文件上传 + 自动分目录 + 重建索引
- 评测工具：
  基于 `eval/sample_qa.jsonl` 计算 Recall@1/3/5 和 MRR

## 项目结构

```text
app/                    # CLI / Web 入口与编排逻辑
mcp_server/             # MCP 工具服务与检索工具实现
preprocessor/           # 离线解析与索引构建
knowledge_base/         # 原始知识库文件
  mock_lab/             # 检索能力验证用的模拟数据
index_store/            # 生成后的索引（navigation/discovery/evidence）
eval/                   # 检索评测脚本与问答数据集
config/                 # 运行配置（YAML）
logs/                   # 查询日志
```

## 快速开始

1. 安装依赖

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. 初始化运行环境

```bash
cp .env.example .env
```

编辑 `config/deep_rag.yaml`：

- `llm.provider: ollama`：本地 / Docker Ollama
- `llm.provider: openai_compatible`：OpenAI 兼容接口（如 SiliconFlow）

说明：

- 环境变量会覆盖 YAML 配置。
- 个人密钥建议放到 `config/deep_rag.local.yaml`（已 gitignore），并设置：
  `DEEP_RAG_CONFIG=config/deep_rag.local.yaml`
- 若代理导致 TLS 证书问题，可临时设置
  `openai_compatible.verify_ssl: false`，或配置 `ca_bundle`。

3. 构建索引

```bash
python -m preprocessor.run_preprocess --kb-dir knowledge_base --index-dir index_store
```

4. 启动 CLI

```bash
python -m app.main "最近三年的销售趋势如何？" --pretty
python -m app.main --interactive
```

5. 启动 Web

```bash
streamlit run app/web.py
```

## 检索评测

在问答数据集上执行检索指标评测：

```bash
python -m eval.run_eval --qa eval/sample_qa.jsonl --k 5
```

输出指标包括：

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`

`eval/sample_qa.jsonl` 为 JSONL（一行一个样本）：

```json
{"id":"q01","question":"...","expected_file":"..."}
```

## 调参项

可在 `config/deep_rag.yaml` 中调整 chunk 和检索约束：

```yaml
retrieval:
  max_supplements: 2
  max_total_tokens: 50000
  chunk_size: 1200
  chunk_overlap: 150
```

## 常用命令

知识库变更后重建索引：

```bash
python -m preprocessor.run_preprocess --kb-dir knowledge_base --index-dir index_store
```

运行测试：

```bash
PYTHONPATH=. pytest -q
```
