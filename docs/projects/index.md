# 实战项目

本教程包含 4 个递进式实战项目，从轻量级单库召回到多库联合 RAG，覆盖向量数据库在不同场景下的典型应用。

建议按照从易到难的顺序学习。

## 项目列表

| 难度 | 项目 | 技术栈 | 场景 |
|------|------|--------|------|
| ⭐ 入门 | [基于 Annoy 的推荐系统召回](./project1/README) | Annoy + DSSM + torch-rechub | 推荐系统向量召回 |
| ⭐⭐ 基础 | [基于 FAISS 框架 RAG 实战](./project2/README) | FAISS + LLM + RAG | 检索增强生成 |
| ⭐⭐⭐ 进阶 | [基于 Milvus 框架的 Agent 项目](./project3/README) | Milvus + LangGraph + Agent | AI Agent 工具调用 |
| ⭐⭐⭐⭐ 综合 | [基于 Milvus 和 ArangoDB 的 RAG 系统](./project4/README) | Milvus + ArangoDB + GraphRAG | 多库联合 RAG |

## 项目简介

### 实战项目 1 · 入门：基于 Annoy 的推荐系统召回

使用 torch-rechub 训练 DSSM 双塔召回模型，结合 Annoy 向量检索实现推荐系统中的召回环节。适合刚学完 Annoy 教程的同学，理解向量数据库在推荐系统中的核心作用。

**你将学到**：双塔模型训练、Embedding 推理、Annoy 索引构建与 Top-K 召回、召回效果评估

### 实战项目 2 · 基础：基于 FAISS 框架 RAG 实战

基于 FAISS 构建检索增强生成（RAG）系统，实现"开卷考试"式的智能问答。从文档切分、向量化、索引构建到 LLM 生成，完整走通 RAG 流程。

**你将学到**：文档处理与切分、FAISS 索引构建与检索、Prompt 工程、RAG 端到端流程

### 实战项目 3 · 进阶：基于 Milvus 框架的 Agent 项目

通过 LangGraph 和 Milvus 构建 AI Agent，让大模型具备工具调用和向量检索能力。理解 Agent 的规划、记忆、工具使用等核心组件。

**你将学到**：Agent 架构设计、LangGraph 工作流、Milvus 向量检索集成、工具调用

### 实战项目 4 · 综合：基于 Milvus 和 ArangoDB 的 RAG 系统

结合 Milvus 向量检索和 ArangoDB 图数据库，构建多模态存储的 GraphRAG 系统。在传统 RAG 基础上引入知识图谱，提升检索的准确性和关联性。

**你将学到**：多数据库架构设计、ArangoDB 图查询、GraphRAG 设计思想、向量+图谱混合检索

## 学习路径

```
Annoy 教程 ──→ 实战项目1（Annoy 召回）
                    │
FAISS 教程 ──→ 实战项目2（FAISS RAG）
                    │
Milvus 教程 ─→ 实战项目3（Milvus Agent）
                    │
               实战项目4（Milvus + ArangoDB）
```

> 每个项目都提供了 Jupyter Notebook（`.ipynb`），可以逐步执行代码并观察输出。建议先跑通 sample 数据，再尝试全量数据。

## 更多实践项目

除了上述 4 个教程配套项目外，仓库 `src/` 目录下还提供了以下独立实践项目，可按兴趣选学：

| 项目 | 简介 | 技术栈 | 源码 |
|------|------|--------|------|
| URL 处理实践 | 基于 RAG 的问答系统，支持视频链接提取和智能回答 | ZhipuAI + Milvus + Gradio | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/url_process) |
| Cre_milvus | 通用向量化处理器，支持多种文件格式的向量化存储 | Milvus + Streamlit + FastAPI | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/Cre_milvus) |
| Graph RAG | 文搜图应用 | Milvus + LangGraph | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/graph_rag) |
| HDBSCAN 聚类可视化 | 聚类数据可视化 | Milvus + HDBSCAN + UMAP | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/HDBSCAN) |
| K8s + Loki 监控 | 基于 Kubernetes 部署的 Milvus 日志监控系统 | Kubernetes + Grafana + Loki | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/k8s+loki) |
| Meta-chunking | Meta-chunking 论文的代码实现 demo | — | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/Meta_chunking) |
| Faiss 问答系统 | 基于 Faiss 的问答系统实战 | FAISS | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/faissSear) |
| Locust 性能测试 | Milvus 性能测试工具和基准测试 | Locust + Milvus | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/locustProj) |
| ANN 算法实现 | 近似最近邻搜索算法实现 | Python | [源码](https://github.com/datawhalechina/easy-vecdb/tree/main/src/ANN_alorithms) |
