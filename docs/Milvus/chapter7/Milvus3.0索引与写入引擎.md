# Chapter 7.4 Milvus 3.0 索引与写入引擎

<!-- EDITOR:BEGIN 本子章节定位
写什么：稀疏索引换代、索引引擎直通、磁盘多向量检索、GPU/标量索引增量、Woodpecker 写入引擎、段合并治理。
不写：索引类型原理与选型总表（Ch2 2.3.3/2.3.7）、AUTOINDEX 入门（Ch3 3.7.5）、mmap 基础（Ch2 2.3.9 / Ch6 存储优化）。
EDITOR:END -->

## 7.4.1 学习目标

<!-- EDITOR:BEGIN 7.4.1
🎯 要求：4–5 条：为新稀疏索引配参数并验证生效 / 判断何时该用引擎直通 / 说清写入引擎换了什么、代价是什么 / 会用强制合并治理碎段。
EDITOR:END -->

## 7.4.2 稀疏向量索引全面换代

### 7.4.2.1 新算法与新能力

<!-- EDITOR:BEGIN 7.4.2.1
🎯 要求：先列"3.0 新增了什么"清单（新搜索算法族 / 倒排列表压缩 / 量化可配 / 按负载选算法 / 位图缓存与序列化优化 / 暴力检索路径优化），再各配一句"读者什么时候该关心"。
🔗 参考：https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/index-explained.md ｜ SINDI 论文 https://arxiv.org/abs/2509.08395
⭐ 重点：稀疏索引从"单一倒排"变成"可按 BM25 / 学习式稀疏 / 延迟预算 三选一"的能力。
🚫 去重边界：Ch2（2.3.3 常见索引类型）已列 SPARSE_INVERTED_INDEX 老参数表，此处**只写增量算法与增量参数**。
EDITOR:END -->

### 7.4.2.2 生效前提与自查

<!-- EDITOR:BEGIN 7.4.2.2
🎯 要求：把 7.1.4.4 的两个开关在这里再落地一次：给"改配置 → 重建索引 → 验证走新路径"的三步操作，并明确"没开时不报错、只是没用到新算法"。
🔗 参考：https://milvus.io/docs/release_notes.md ｜ https://milvus.io/docs/dynamic_config.md
⚠️ 用 > 📌 **前置条件** 标注，这是本章最高频踩坑点。
EDITOR:END -->

### 7.4.2.3 选型与调优

<!-- EDITOR:BEGIN 7.4.2.3
🎯 要求：一张决策表（场景：全文 BM25 / SPLADE 类学习稀疏 / 高并发低延迟 → 建议算法 + 关键参数 + 预期代价），加一组自测脚本（同一份语料三种配置对比召回与延迟）。
🔗 参考：https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/full-text-search.md ｜ https://milvus.io/docs/index_selection.md
🚫 去重边界：不做跨索引大横评（Ch2 2.3.5 已有性能权衡），只比新算法之间。
EDITOR:END -->

## 7.4.3 索引引擎直通模式

### 7.4.3.1 参数与最小可用样例

<!-- EDITOR:BEGIN 7.4.3.1
🎯 要求：一段建索引代码 + 一段检索代码，参数名照官方页抄。
🔗 参考：https://milvus.io/docs/faiss.md ｜ https://milvus.io/docs/knowhere.md
⭐ 重点：把它讲成"Milvus 的一种索引类型"，读者只需给一条引擎配方串即可复用经典组合；检索参数透传。
🚫 口径红线：**只讲 Milvus 侧用法，不与任何其他检索库做对比、不提其他教程轨**。
EDITOR:END -->

### 7.4.3.2 适用场景与代价

<!-- EDITOR:BEGIN 7.4.3.2
🎯 要求：3–6 行。适合：复现既有配方、需要经典组合参数化实验。代价：失去 Milvus 上层自动优化/部分特性兼容需自行验证。
🔗 参考：https://milvus.io/docs/faiss.md ｜ https://milvus.io/docs/limitations.md
⚠️ 需在 3.0.x + 对应 pymilvus 版本实测后再写结论，未验证的一律标"待验证"。
EDITOR:END -->

## 7.4.4 磁盘上的多向量检索

### 7.4.4.1 多向量字段的新数据类型

<!-- EDITOR:BEGIN 7.4.4.1
🎯 要求：一句话定义 + 一段"写入一条含向量数组的记录"代码。
🔗 参考：https://milvus.io/docs/search-with-embedding-lists.md ｜ https://milvus.io/docs/array-of-structs.md
🚫 去重边界：StructArray 定义在 7.3.2，此处只讲"多向量成员 + 磁盘索引"这一组合，勿重复语法教学。
EDITOR:END -->

### 7.4.4.2 搜索策略选型

<!-- EDITOR:BEGIN 7.4.4.2
🎯 要求：一张表（策略 / 精度-延迟-内存三角定位 / 数据规模适用 / 典型配置片段），这是本节主体。
🔗 参考：https://milvus.io/docs/choose-an-embeddinglist-search-strategy.md ｜ https://milvus.io/docs/diskann.md
⚠️ 表里数值口径（如召回差异、内存占用量级）需自己跑一遍再填，别抄宣传数字。
EDITOR:END -->

## 7.4.5 索引能力其余增量

### 7.4.5.1 GPU 索引与镜像变化

<!-- EDITOR:BEGIN 7.4.5.1
🎯 要求：5 行内。要点=GPU 镜像迁至 CUDA 12.9、不再保留 Ubuntu 20.04 兼容；GPU 索引用法本身不变。
🔗 参考：https://milvus.io/docs/gpu-index-overview.md ｜ https://milvus.io/docs/release_notes.md
🚫 去重边界：GPU 索引原理不重讲。
EDITOR:END -->

### 7.4.5.2 标量索引版本提升

<!-- EDITOR:BEGIN 7.4.5.2
🎯 要求：两三行 + 一个开关提醒（标量新索引版本需 targetScalarIndexVersion=4）。
🔗 参考：https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/set-collection-ttl.md
EDITOR:END -->

## 7.4.6 写入引擎 Woodpecker

### 7.4.6.1 它替换了什么、为什么

<!-- EDITOR:BEGIN 7.4.6.1
🎯 要求：一张表（旧：外部消息队列角色 / 新：Milvus 自研写前日志承担的角色），再配一段"读者视角的收益"：少一个外部依赖、运维面缩小。
🔗 参考：https://milvus.io/docs/woodpecker.md ｜ https://milvus.io/docs/woodpecker_architecture.md ｜ https://milvus.io/docs/mqtype-overview.md
🚫 去重边界：整体架构分层在 Ch6《Milvus底层架构详解》已讲，此处**只讲写入链路这一处变化**，用 `> ⛳ **架构演进提示**` 回引即可。
EDITOR:END -->

### 7.4.6.2 从消息队列切换到 Woodpecker

<!-- EDITOR:BEGIN 7.4.6.2
🎯 要求：分 Standalone 与集群两种给出切换步骤 + 回退说明 + 数据一致性注意点；每步一句"预期现象"。
🔗 参考：https://milvus.io/docs/switch-mq-type.md
⚠️ 切换前提与兼容性以官方页现文为准；生产切换务必先在测试集群演练，正文要写这句。
EDITOR:END -->

### 7.4.6.3 独立部署与可观测

<!-- EDITOR:BEGIN 7.4.6.3
🎯 要求：讲"作为独立服务跑起来"意味着什么（资源隔离、单独扩缩、故障定位）；给关键指标/日志位点看法。
🔗 参考：https://milvus.io/docs/woodpecker_architecture.md ｜ https://milvus.io/docs/tune_consistency.md ｜ https://milvus.io/docs/configure-querynode-localdisk.md
🚫 边界：写路径原理只在 7.4.6.1 讲一次，此处不重述。
EDITOR:END -->

## 7.4.7 段合并与治理

### 7.4.7.1 合并策略与碎段问题

<!-- EDITOR:BEGIN 7.4.7.1
🎯 要求：先讲碎段带来的两个后果（查询延迟抖动、存储膨胀），再给合并的观测方法。
🔗 参考：https://milvus.io/docs/clustering-compaction.md ｜ https://milvus.io/docs/limitations.md
🚫 去重边界：Ch6 的 mmap/存储优化不重述，只写"合并"这一条新链路。
EDITOR:END -->

### 7.4.7.2 强制合并（同步 / 异步）

<!-- EDITOR:BEGIN 7.4.7.2
🎯 要求：一段代码演示强制合并（同步与异步各一次）+ 何时该手动触发（批量导入后、索引换代后）。
🔗 参考：https://milvus.io/docs/force-merge.md ｜ https://milvus.io/docs/import-data.md
⚠️ 大集合强制合并耗时与资源占用需实测，写清前置条件。
EDITOR:END -->

## 7.4.8 本章小结

<!-- EDITOR:BEGIN 7.4.8
🎯 要求：`本章核心掌握：` + 4 条 + 衔接句（"7.5 把四章能力串成一条链路"）。
⚠️ 自检：是否误重述 Ch2 的索引类型总表、Ch6 的 mmap 与架构分层。
EDITOR:END -->
