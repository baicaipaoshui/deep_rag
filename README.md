# Deep RAG

Deep RAG 是一个面向企业知识库的智能检索问答系统，支持混合文档（`.md`、`.pdf`、`.xlsx/.xlsm/.xls`），采用 Python 编排层 + MCP 工具层 + 离线索引构建的架构，支持本地 LLM（Ollama）与 OpenAI 兼容云端接口（如 SiliconFlow）。

## 核心特性

- 五阶段状态机编排：`FILTERING -> LOCATING -> EXTRACTING -> VERIFYING -> GENERATING`
- 三层索引体系：导航索引（L1）+ 发现索引（L2）+ 证据索引（L3）
- 混合检索：Whoosh 关键词检索 + ChromaDB 向量检索 + RRF 融合
- 稳定性控制：全链路 token 预算守护 + 自动补检轮次
- 充分性门控：支持证据不足的迭代检索与有界重试
- 双评测链路：
  - `eval/`：检索指标（Recall@1/3/5、MRR）
  - `evaluation/`：RAGAS 四维评测与结果归档

## 当前项目结构

```text
.
├── app/                         # CLI/Web入口、编排状态机、查询分析与答案生成
│   └── orchestrator/
│       ├── prompts/             # 编排与生成提示词模板
│       ├── query_analyzer.py
│       ├── state_machine.py
│       └── answer_generator.py
├── mcp_server/                  # MCP 服务层
│   ├── tools/                   # 6个标准工具（导航/发现/读取/证据/校验/浏览）
│   └── core/                    # 检索引擎、文件读取、索引管理
├── preprocessor/                # 离线预处理与索引构建
│   ├── parsers/                 # markdown/pdf/excel 解析器
│   └── index_builder.py
├── knowledge_base/              # 原始知识库
│   ├── general/
│   ├── sales_reports/
│   ├── tech_docs/
│   └── mock_lab/
├── index_store/                 # 索引产物（navigation/discovery/evidence）
├── eval/                        # 检索评测（Recall/MRR）
├── evaluation/                  # 生成与RAGAS评测（含结果落盘）
├── tests/                       # 单元测试与集成测试
├── config/
│   └── deep_rag.yaml            # 统一配置入口
├── project_config.py            # 配置模型与默认值回退逻辑
└── README.md
```

## 快速开始

### 1) 安装依赖

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2) 初始化配置

```bash
cp .env.example .env
```

编辑 `config/deep_rag.yaml`，常见模式：

- `llm.provider: ollama`：本地 / Docker Ollama
- `llm.provider: openai_compatible`：OpenAI 兼容 API（如 SiliconFlow）

推荐做法：

- 环境变量覆盖 YAML。
- 个人密钥放入 `config/deep_rag.local.yaml`（已忽略），并设置：
  `DEEP_RAG_CONFIG=config/deep_rag.local.yaml`
- 如代理导致 TLS 问题，可临时设置 `openai_compatible.verify_ssl: false` 或配置 `ca_bundle`。

### 3) 构建索引

```bash
python -m preprocessor.run_preprocess --kb-dir knowledge_base --index-dir index_store
```

从 SQLite 抽取后构建索引：

```bash
python -m preprocessor.run_preprocess \
  --source-type sqlite \
  --sqlite-path ./data/kb.db \
  --sqlite-query "SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents" \
  --kb-dir knowledge_base \
  --index-dir index_store
```

`sqlite-query` 默认要求返回字段：`id`、`title`、`content`、`updated_at`，并可额外返回 `source_table`。

从 MySQL 抽取后构建索引：

```bash
python -m preprocessor.run_preprocess \
  --source-type mysql \
  --mysql-host 127.0.0.1 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password your_password \
  --mysql-database your_db \
  --mysql-query "SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents" \
  --kb-dir knowledge_base \
  --index-dir index_store
```

从 PostgreSQL 抽取后构建索引：

```bash
python -m preprocessor.run_preprocess \
  --source-type postgres \
  --postgres-dsn "postgresql://user:password@127.0.0.1:5432/your_db" \
  --postgres-query "SELECT id, title, content, updated_at, 'documents' AS source_table FROM documents" \
  --kb-dir knowledge_base \
  --index-dir index_store
```

### 4) 启动 CLI

```bash
python -m app.main "最近三年的销售趋势如何？" --pretty
python -m app.main --interactive
```

### 5) 启动 Web

```bash
streamlit run app/web.py
```

## 评测与验证

### A. 检索评测（Recall/MRR）

```bash
python -m eval.run_eval --qa eval/sample_qa.jsonl --k 5
```

输出指标：

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`

### B. RAGAS 评测（四维）

```bash
python -m evaluation.run_eval
```

结果默认写入 `evaluation/results/`。

RAGAS 依赖预检：

```bash
python -m evaluation.check_ragas_prereq
```

## 常用命令

重建索引：

```bash
python -m preprocessor.run_preprocess --kb-dir knowledge_base --index-dir index_store
```

运行测试：

```bash
PYTHONPATH=. pytest -q
```

## 当前不足与改进方向

- 不足点：数据源接入能力不足（Data Source Connectors 不足）
- 现状说明：当前已支持本地文件夹、SQLite、MySQL、PostgreSQL 作为知识源入口，仍缺少对数据仓库、业务 API 的直连能力
- 影响：实时性不足、同步链路长、企业落地通常需要额外 ETL
- 改进方向：新增统一 Connector 层，支持 DB/API 增量抽取与权限审计对接
