# Chapter 7.4 Milvus 3.0 索引与写入引擎

<!-- EDITOR:BEGIN 本子章节定位
写什么：稀疏索引换代、索引引擎直通、磁盘多向量检索、GPU/标量索引增量、Woodpecker 写入引擎、段合并治理。
不写：索引类型原理与选型总表（Ch2 2.3.3/2.3.7）、AUTOINDEX 入门（Ch3 3.7.5）、mmap 基础（Ch2 2.3.9 / Ch6 存储优化）。
   共通规矩（每节都适用）：
     1) 开头一句回答"本节解决什么"，第二句说"不解决什么"；
     2) 每段代码前说清前提、后说清预期输出，贴出的代码必须能跑；
     3) 数字只写自己跑出来的，未跑写"待补测"；只有官方原文明确写的才能称"官方口径"；
     4) 限制单独成块，不混进正文；已知限制不准用"暂不支持"一句带过；
     5) 只做 2.6 与 3.0 的对比，不与其他检索库比较；
     6) 术语按本文件头统一，英文原词首次出现加括号。
   注释怎么用：每节标题下一条 EDITOR 注释，含【本节要写什么】【参考资料】【必须注意】【写完自检】【删除】；
     动笔前先读【参考资料】，写完对着【写完自检】验收，定稿后删掉整块注释（或跑 .temp/strip_editor_notes.py）。
EDITOR:END -->

## 7.4.1 学习目标

<!-- EDITOR:BEGIN 7.4.1
【本节要写什么】4–5 条：为新稀疏索引配参数并验证生效 / 判断何时该用引擎直通 / 说清写入引擎换了什么、代价是什么 / 会用强制合并治理碎段。
【必须注意】"代价"要落到具体资源（内存/磁盘/延迟），不要写"有一定开销"。
【写完自检】4–6 条目标都能在本页找到对应节；读者能据此判断要不要读本章。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.2 稀疏向量索引全面换代

### 7.4.2.1 新算法与新能力

<!-- EDITOR:BEGIN 7.4.2.1
【本节要写什么】先列"3.0 新增了什么"清单（新搜索算法族 / 倒排列表压缩 / 量化可配 / 按负载选算法 / 位图缓存与序列化优化 / 暴力检索路径优化），再各配一句"读者什么时候该关心"。
【先读这些】https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/index-explained.md ｜ SINDI 论文 https://arxiv.org/abs/2509.08395
【必须注意】
・稀疏索引从"单一倒排"变成"可按 BM25 / 学习式稀疏 / 延迟预算 三选一"的能力。
・Ch2（2.3.3 常见索引类型）已列 SPARSE_INVERTED_INDEX 老参数表，此处**只写增量算法与增量参数**。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.2.2 生效前提与自查

<!-- EDITOR:BEGIN 7.4.2.2
【本节要写什么】把 7.1.4.4 的两个开关在这里再落地一次：给"改配置 → 重建索引 → 验证走新路径"的三步操作，并明确"没开时不报错、只是没用到新算法"。
【先读这些】https://milvus.io/docs/release_notes.md ｜ https://milvus.io/docs/dynamic_config.md
【必须注意】
・️ 用 > 📌 **前置条件** 标注，这是本章最高频踩坑点。
・不开不报错，仍走老路径。本节必须给"看到什么才算生效"的判据，与 7.1.4.4 形成闭环。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.2.3 选型与调优

<!-- EDITOR:BEGIN 7.4.2.3
【本节要写什么】一张决策表（场景：全文 BM25 / SPLADE 类学习稀疏 / 高并发低延迟 → 建议算法 + 关键参数 + 预期代价），加一组自测脚本（同一份语料三种配置对比召回与延迟）。
【先读这些】https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/full-text-search.md ｜ https://milvus.io/docs/index_selection.md
【必须注意】
・不做跨索引大横评（Ch2 2.3.5 已有性能权衡），只比新算法之间。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.3 索引引擎直通模式

### 7.4.3.1 参数与最小可用样例

<!-- EDITOR:BEGIN 7.4.3.1
【本节要写什么】一段建索引代码 + 一段检索代码，参数名照官方页抄。
【先读这些】https://milvus.io/docs/faiss.md ｜ https://milvus.io/docs/knowhere.md
【必须注意】
・把它讲成"Milvus 的一种索引类型"，读者只需给一条引擎配方串即可复用经典组合；检索参数透传。
・口径红线：**只讲 Milvus 侧用法，不与任何其他检索库做对比、不提其他教程轨**。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.3.2 适用场景与代价

<!-- EDITOR:BEGIN 7.4.3.2
【本节要写什么】3–6 行。适合：复现既有配方、需要经典组合参数化实验。代价：失去 Milvus 上层自动优化/部分特性兼容需自行验证。
【先读这些】https://milvus.io/docs/faiss.md ｜ https://milvus.io/docs/limitations.md
【必须注意】
・️ 需在 3.0.x + 对应 pymilvus 版本实测后再写结论，未验证的一律标"待验证"。
・结论必须 3.0.x + 对应 pymilvus 实测过；未验证写"待验证"。口径红线：不与任何其他检索库对比。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.4 磁盘上的多向量检索

### 7.4.4.1 多向量字段的新数据类型

<!-- EDITOR:BEGIN 7.4.4.1
【本节要写什么】一句话定义 + 一段"写入一条含向量数组的记录"代码。
【先读这些】https://milvus.io/docs/search-with-embedding-lists.md ｜ https://milvus.io/docs/array-of-structs.md
【必须注意】
・StructArray 定义在 7.3.2，此处只讲"多向量成员 + 磁盘索引"这一组合，勿重复语法教学。
・术语统一用"向量列表（embedding list）"；StructArray 语法已在 7.3.2 教过，这里只讲与磁盘索引的组合。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.4.2 搜索策略选型

<!-- EDITOR:BEGIN 7.4.4.2
【本节要写什么】一张表（策略 / 精度-延迟-内存三角定位 / 数据规模适用 / 典型配置片段），这是本节主体。
【先读这些】https://milvus.io/docs/choose-an-embeddinglist-search-strategy.md ｜ https://milvus.io/docs/diskann.md
【必须注意】
・️ 表里数值口径（如召回差异、内存占用量级）需自己跑一遍再填，别抄宣传数字。
・表里的召回差异、内存量级自己跑，别抄宣传数字。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.5 索引能力其余增量

### 7.4.5.1 GPU 索引与镜像变化

<!-- EDITOR:BEGIN 7.4.5.1
【本节要写什么】5 行内。要点=GPU 镜像迁至 CUDA 12.9、不再保留 Ubuntu 20.04 兼容；GPU 索引用法本身不变。
【先读这些】https://milvus.io/docs/gpu-index-overview.md ｜ https://milvus.io/docs/release_notes.md
【必须注意】
・GPU 索引原理不重讲。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.5.2 标量索引版本提升

<!-- EDITOR:BEGIN 7.4.5.2
【本节要写什么】两三行 + 一个开关提醒（标量新索引版本需 targetScalarIndexVersion=4）。
【先读这些】https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/release_notes.md#v300（索引版本开关）
【必须注意】写清 targetScalarIndexVersion 在哪一层配置、改完是否要重启或重建索引；不要暗示默认已开。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.6 写入引擎 Woodpecker

### 7.4.6.1 它替换了什么、为什么

<!-- EDITOR:BEGIN 7.4.6.1
【本节要写什么】一张表（旧：外部消息队列角色 / 新：Milvus 自研写前日志承担的角色），再配一段"读者视角的收益"：少一个外部依赖、运维面缩小。
【先读这些】https://milvus.io/docs/woodpecker.md ｜ https://milvus.io/docs/woodpecker_architecture.md ｜ https://milvus.io/docs/mqtype-overview.md
【必须注意】
・整体架构分层在 Ch6《Milvus底层架构详解》已讲，此处**只讲写入链路这一处变化**，用 `> ⛳ **架构演进提示**` 回引即可。
・Woodpecker 是 3.x 默认 WAL，默认嵌入模式；独立服务仅集群可用。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.6.2 从消息队列切换到 Woodpecker

<!-- EDITOR:BEGIN 7.4.6.2
【本节要写什么】分 Standalone 与集群两种给出切换步骤 + 回退说明 + 数据一致性注意点；每步一句"预期现象"。
【先读这些】https://milvus.io/docs/switch-mq-type.md（总览） ｜ https://milvus.io/docs/switch-rocksmq-woodpecker.md（Standalone） ｜ https://milvus.io/docs/switch-pulsar-woodpecker.md（集群）
【必须注意】
・️ 切换前提与兼容性以官方页现文为准；生产切换务必先在测试集群演练，正文要写这句。
・切换前要停写；旧 MQ 数据是否回放若文档未写明，标"待验证"。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.6.3 独立部署与可观测

<!-- EDITOR:BEGIN 7.4.6.3
【本节要写什么】讲"作为独立服务跑起来"意味着什么（资源隔离、单独扩缩、故障定位）；给关键指标/日志位点看法。
【先读这些】https://milvus.io/docs/woodpecker_architecture.md ｜ https://milvus.io/docs/woodpecker.md（配置与指标）
【必须注意】
・边界：写路径原理只在 7.4.6.1 讲一次，此处不重述。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.7 段合并与治理

### 7.4.7.1 合并策略与碎段问题

<!-- EDITOR:BEGIN 7.4.7.1
【本节要写什么】先讲碎段带来的两个后果（查询延迟抖动、存储膨胀），再给合并的观测方法。
【先读这些】https://milvus.io/docs/clustering-compaction.md ｜ https://milvus.io/docs/limitations.md
【必须注意】
・Ch6 的 mmap/存储优化不重述，只写"合并"这一条新链路。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.4.7.2 强制合并（异步）

<!-- EDITOR:BEGIN 7.4.7.2
【本节要写什么】一段代码演示强制合并（官方操作页只提供异步：compact(target_size=...) 后用 job id 轮询；release notes 虽提到 sync/async，但 API 无 sync 参数，不要编造）+ 何时该手动触发（批量导入后、索引换代后）。
【先读这些】https://milvus.io/docs/force-merge.md
【必须注意】
・️ 官方明确：Do not use force merge compaction in production environments。正文必须原样告知，并给“低峰期 / 测试环境”前提。
・️ 大集合强制合并耗时与资源占用需实测，写清前置条件。
・只有异步接口，没有同步版；官方原文 Do not use force merge compaction in production environments，且为 public preview——必须用 > ⚠️ 注意 单独收口。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.4.8 本章小结

<!-- EDITOR:BEGIN 7.4.8
【本节要写什么】`本章核心掌握：` + 4 条 + 衔接句（"7.5 把四章能力串成一条链路"）。
【必须注意】
・️ 自检：是否误重述 Ch2 的索引类型总表、Ch6 的 mmap 与架构分层。
【写完自检】以「本章核心掌握：」起头；只看小结也能复述本章三条主线。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->
