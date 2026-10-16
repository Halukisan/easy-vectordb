# Chapter 7.5 Milvus 3.0 端到端实战：从湖上文件到可回滚的检索服务

<!-- EDITOR:BEGIN 本子章节定位
写什么：把 7.2–7.4 的能力串成一条完整链路，让读者一次跑通并看到"3.0 与 2.6 的体验差"。
不写：任何单点特性的重复教学（外部集合语法见 7.2.3、StructArray 见 7.3.2、稀疏索引见 7.4.2），本章只组装。
EDITOR:END -->

## 7.5.1 学习目标

<!-- EDITOR:BEGIN 7.5.1
🎯 要求：4 条：独立搭起一条 3.0 检索链路 / 会用函数在库内派生检索字段 / 会在一次请求里完成检索+排序+聚合+重排 / 会用快照做发布前保护与回滚。
EDITOR:END -->

## 7.5.2 场景与数据准备

<!-- EDITOR:BEGIN 7.5.2
🎯 要求：给一个贴教程气质的场景（建议：本地文档库→导出为 Parquet 放对象存储/本地目录→做混合检索+去重+版本可回滚）。列清目录结构与样例字段。
🔗 参考：https://milvus.io/docs/prepare-source-data.md ｜ https://milvus.io/docs/import-data.md
⚠️ 数据放 `data/` 下新建子目录（如 `data/milvus30_demo/`），生成脚本放 `src/` 独立目录，沿用本项目 README + requirements.txt 惯例。
EDITOR:END -->

## 7.5.3 搭建步骤

### 7.5.3.1 环境与开关自检

<!-- EDITOR:BEGIN 7.5.3.1
🎯 要求：三步确认（服务端版本 / Loon 开关 / 索引版本开关），给一段打印版本的代码。回引 7.1.4，不重抄配置细节。
🔗 参考：https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/dynamic_config.md
EDITOR:END -->

### 7.5.3.2 挂载外部集合

<!-- EDITOR:BEGIN 7.5.3.2
🎯 要求：8–15 行代码，跑通"湖上文件→外部集合→可检索"。语法回引 7.2.3，此处只给最终形态配置。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md
EDITOR:END -->

### 7.5.3.3 库内派生检索字段

<!-- EDITOR:BEGIN 7.5.3.3
🎯 要求：一次配好两类函数：BM25 稀疏向量（全文）与 MinHash 签名（去重），说明"原文留外部、检索字段在库里算"。
🔗 参考：https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/embedding-function-overview.md
EDITOR:END -->

### 7.5.3.4 用 StructArray 承载切片与多向量

<!-- EDITOR:BEGIN 7.5.3.4
🎯 要求：给"一文档多切片、每切片一个向量 + 若干标量"的字段设计与写入代码，体现父子同集合。
🔗 参考：https://milvus.io/docs/create-structarray-field.md ｜ https://milvus.io/docs/search-with-embedding-lists.md
🚫 去重边界：语法细节见 7.3.2，此处不重讲。
EDITOR:END -->

### 7.5.3.5 索引配置

<!-- EDITOR:BEGIN 7.5.3.5
🎯 要求：一段代码同时给：稠密向量索引 / 稀疏向量用新算法 / struct 子字段索引 / 标量索引。并给"确认新路径生效"的验证行。
🔗 参考：https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index-structarray-fields.md
EDITOR:END -->

### 7.5.3.6 一次请求完成检索 + 排序 + 分面 + 重排

<!-- EDITOR:BEGIN 7.5.3.6
🎯 要求：本章高潮。给一份完整调用（含 filter、排序、聚合分面、重排阶段），并贴真实返回片段，指出"过去要 3 次请求 + 客户端拼装"。
🔗 参考：https://milvus.io/docs/search-aggregation.md ｜ https://milvus.io/docs/single-vector-search.md （找页内小节 "Sort Search Results by Scalar Fields"） ｜ https://milvus.io/docs/model-ranker-overview.md ｜ https://milvus.io/docs/boost-ranker.md ｜ https://milvus.io/docs/multi-vector-search.md
EDITOR:END -->

### 7.5.3.7 发布前打快照与回滚演练

<!-- EDITOR:BEGIN 7.5.3.7
🎯 要求：演练一次"故意写坏→从快照恢复→结果一致"，让读者亲眼看到边际存储≈0 与恢复速度。
🔗 参考：https://milvus.io/docs/manage-snapshots.md ｜ https://milvus.io/docs/snapshot-use-cases.md
EDITOR:END -->

## 7.5.4 日常运行：增量与演进

<!-- EDITOR:BEGIN 7.5.4
🎯 要求：两小节合并写：外部数据刷新（refresh）+ 上线后换嵌入模型时在线加列回填；各 5–10 行代码。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/alter-collection-field.md
⚠️ 明确"外部集合只读"，更新靠刷新而非写入。
EDITOR:END -->

## 7.5.5 结果观察与代价

<!-- EDITOR:BEGIN 7.5.5
🎯 要求：给一张"本次链路用了哪些 3.0 新能力 / 2.6 下要怎么做 / 实测差异"的总结表；再诚实列出踩到的限制（引用各节限制清单，不复述）。
🔗 参考：https://milvus.io/docs/structarray-limits.md ｜ https://milvus.io/docs/limitations.md ｜ https://milvus.io/docs/tune_consistency.md
⚠️ 所有数字必须是你自己跑出来的，未跑的写"待补测"。
EDITOR:END -->

## 7.5.6 本章小结

<!-- EDITOR:BEGIN 7.5.6
🎯 要求：`本章核心掌握：` + 4 条 + 一句全大章收束（"至此 Milvus 3.0x 的新增能力全部覆盖，未列出的均为与 2.6 相同的行为，见前六章"）。
EDITOR:END -->
