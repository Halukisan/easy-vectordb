# Chapter 7.2 Milvus 3.0 湖仓原生：外部集合与快照

<!-- EDITOR:BEGIN 本子章节定位
写什么：Storage V3 怎么开怎么验（原理一笔带过）、外部集合从零建到检索到追平、快照四个用例、在线加列/回填/删列。
边界：只讲 3.0 新增能力；Milvus 基础概念（集合/字段/索引/搜索）见 Chapter 2、Chapter 3，不重述。
EDITOR:END -->

## 7.2.1 学习目标

<!-- EDITOR:BEGIN 7.2.1
🎯 要求：4–5 条，动词起头。建议：解释 Storage V3 与上两层新范式的关系 / 独立完成一个外部集合的挂载·建索引·检索·追平 / 会用快照做误操作回滚与版本对照 / 在不停机前提下给已有集合加列并回填。
EDITOR:END -->

## 7.2.2 存储引擎 V3：新底座

### 7.2.2.1 为什么先讲底座

<!-- EDITOR:BEGIN 7.2.2.1
🎯 要求：5–8 行 + 一张"2.x 存储与元数据组织 vs Storage V3"两列简表。
🔗 参考：https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/object-storage.md ｜ https://milvus.io/docs/release_notes.md （beta "Storage V3（Loon）"一节）
⭐ 重点：manifest 是唯一真相 + 原子更新；列式对象存储；Vortex 将成默认内部格式（自适应编码：字典/RLE/bit-packing/浮点压缩、零拷贝解压）；LOB 改为分区级 Vortex 文件跨段共享，compaction 只搬 (file_id, offset) 引用不搬正文。
🚫 去重边界：Chapter 2（2.3.8 内存与存储优化 / 2.3.9 mmap）与 Chapter 6《milvus 存储优化》讲的是加载与 mmap，属另一层，勿混讲、勿重复。
⚠️ 开启方法在 7.1.4.3，此处只讲"是什么/为什么"，不重复给配置。
EDITOR:END -->

### 7.2.2.2 验证是否真用上了 V3

<!-- EDITOR:BEGIN 7.2.2.2
🎯 要求：给一段可跑的"建集合→插入→查询对象存储目录结构"的观察流程，让读者肉眼看到新格式产物。
🔗 参考：https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/import-data.md ｜ https://milvus.io/docs/prepare-source-data.md
⭐ 重点：Loon 未开时功能会静默降级或报错，正文要写出"确认已生效"的判据（对象存储里出现新的清单/段文件格式）。
EDITOR:END -->

## 7.2.3 外部集合（External Collection）

### 7.2.3.1 三个关键属性与适用场景

<!-- EDITOR:BEGIN 7.2.3.1
🎯 要求：三条属性各一句话 + 一张"该用外部集合 / 该用普通集合"的判据表。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/data-infra-integration-overview.md
⭐ 重点：原地读（zero-copy，不搬数据）/ 只读（写入、upsert、删除一律拒绝）/ 靠增量 refresh 追平。
⚠️ 只读这点必须前置强调，否则读者第一行代码就想 insert。
EDITOR:END -->

### 7.2.3.2 支持的存储格式与前置准备

<!-- EDITOR:BEGIN 7.2.3.2
🎯 要求：一张表（格式 / 典型来源 / 需要的凭据与路径形态），加一条 > 📌 **前置条件**。
🔗 参考：https://milvus.io/docs/data-infra-integration-overview.md ｜ https://milvus.io/docs/object-storage.md ｜ https://milvus.io/docs/prepare-source-data.md
⭐ 重点：Parquet / Lance / Iceberg / Vortex；外部字段类型映射在 create 时定；建集合前要有对象存储访问凭据。
🚫 术语口径：这些一律表述为"Milvus 外部集合支持的存储格式"，不介绍对应生态产品本身。
EDITOR:END -->

### 7.2.3.3 实操：挂载一个外部集合

<!-- EDITOR:BEGIN 7.2.3.3
🎯 要求：最小可跑闭环，用 MilvusClient 风格，中文注释，每段代码前后各一句"做什么/预期输出"。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md
⭐ 重点：参数按官方页原样抄，不要凭记忆拼字段名（尤其外部路径、字段映射、主键/向量列的对应关系）。
🚫 去重边界：普通集合的创建写法见 Chapter 3（3.3.7/3.3.8），此处只给外部集合的差异参数。
EDITOR:END -->

### 7.2.3.4 在外部集合上建索引与检索

<!-- EDITOR:BEGIN 7.2.3.4
🎯 要求：两段代码（建索引、search），并点明"索引在库内、数据在库外"这一核心心智。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/filtered-search.md
⭐ 重点：外部列上可用全文索引/BM25、标量索引、JSON、动态字段、分区键、clustering key；检索语义与普通集合一致。
🚫 去重边界：索引类型科普在 Chapter 2（2.3.3），此处不重述，只讲外部集合上的可用性差异。
EDITOR:END -->

### 7.2.3.5 外部列直连函数输出字段

<!-- EDITOR:BEGIN 7.2.3.5
🎯 要求：本子章节最有卖点的一节。给"文本留在湖里、检索字段在库里生成"的完整例子（BM25 稀疏向量为首选）。
🔗 参考：https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/embedding-function-overview.md ｜ https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/create-an-external-collection.md
⭐ 重点：3.0.0 起外部列可直接作为函数输入，产出 BM25 稀疏向量、MinHash 签名、text embedding；这解决了"湖上原文 + 库内检索表示"的双写难题。
⚠️ 外部 embedding 函数对远端模型的批量调用开销不小，正文给一句性能提示。
EDITOR:END -->

### 7.2.3.6 增量 refresh：让外部数据追平

<!-- EDITOR:BEGIN 7.2.3.6
🎯 要求：给 refresh 的调用 + "湖上改一行→refresh→检索结果变化"的对照验证。
🔗 参考：https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/alter-external-collection-schema.md ｜ https://milvus.io/docs/import-data.md
⭐ 重点：元数据同步走增量流程；refresh 的粒度、耗时与何时该刷（正文给决策建议）。
⚠️ 注意：官方 v3.0.x 文档中 refresh 相关内容并入 create-an-external-collection.md 页（无独立 refresh 页），落笔前以该页现文为准。
EDITOR:END -->

### 7.2.3.7 外部集合的 schema 调整

<!-- EDITOR:BEGIN 7.2.3.7
🎯 要求：短小节（10 行内）。讲湖上加了列怎么办、字段类型不兼容怎么办。
🔗 参考：https://milvus.io/docs/alter-external-collection-schema.md
🚫 去重边界：与 7.2.5 的普通集合在线加列区分开——这里专指外部集合。
EDITOR:END -->

### 7.2.3.8 常见报错与排查

<!-- EDITOR:BEGIN 7.2.3.8
🎯 要求：3–5 条 `#### 1. xxx` 式排错条目，每条=现象/原因/处理。建议覆盖：凭据或路径错、格式不匹配、在只读集合上写入、外部字段映射不到、refresh 后无变化。
🔗 参考：https://milvus.io/docs/troubleshooting.md
⭐ 重点：把你实跑时遇到的真实报错原文贴进来（比编造更有价值）。
EDITOR:END -->

## 7.2.4 快照（Snapshot）

### 7.2.4.1 快照是什么

<!-- EDITOR:BEGIN 7.2.4.1
🎯 要求：3–6 行 + 一张"快照 / 备份 / 副本"三者区别表（易混，必须辨析）。
🔗 参考：https://milvus.io/docs/snapshots.md ｜ https://milvus.io/docs/snapshot-use-cases.md
⭐ 重点：时点只读视图；按段引用而非复制，边际存储近零；依赖 Loon 已开启。
🚫 去重边界：与 Chapter 3（3.9 分区）无关，不要为省事拿分区当快照讲。
EDITOR:END -->

### 7.2.4.2 实操：创建·列举·恢复·删除

<!-- EDITOR:BEGIN 7.2.4.2
🎯 要求：四段短代码，每段一句说明；参数名以官方页为准。
🔗 参考：https://milvus.io/docs/snapshots.md ｜ https://milvus.io/docs/manage-snapshots.md
⭐ 重点：create / list / restore / drop 四个动作；恢复的目标是"新集合"还是"原集合"要写清楚。
🚫 术语提醒：Lite 的 snapshot 是嵌入式引擎机制，与服务端快照不同（见 7.1.5.1），此处加 > 💡 **版本说明** 一句话即可，别展开。
EDITOR:END -->

### 7.2.4.3 四个典型用例

<!-- EDITOR:BEGIN 7.2.4.3
🎯 要求：每个用例=场景一句话 + 3–6 行代码，不贴整页 API 表。
🔗 参考：https://milvus.io/docs/snapshot-use-cases.md ｜ https://milvus.io/docs/manage-snapshots.md
⭐ 重点：误删回滚 / A-B 版本对照检索 / 灌数据中途失败回退到干净基线（对应官方 2.6→3.0 灌数据失败案例）/ 配合备份做时间点还原。
EDITOR:END -->

### 7.2.4.4 快照与备份恢复的分工

<!-- EDITOR:BEGIN 7.2.4.4
🎯 要求：一张对照表（解决什么问题/是否跨实例/是否依赖对象存储/可回滚粒度），正文不超过 12 行。
🔗 参考：https://milvus.io/docs/multi-storage-backup-and-restore.md ｜ https://milvus.io/docs/cross-bucket-backup-and-restore.md ｜ https://milvus.io/docs/shared-bucket-backup-and-restore.md ｜ https://milvus.io/docs/single-instance-backup-and-restore.md
🚫 去重边界：Chapter 6《Milvus底层架构详解》讲的是架构容错，此处只讲数据面备份恢复，勿重述架构。
EDITOR:END -->

## 7.2.5 Schema 在线演进：加列·回填·删列

### 7.2.5.1 在线加列与删列

<!-- EDITOR:BEGIN 7.2.5.1
🎯 要求：一段代码贯穿"加列→服务不中断可查→删列"，并明确写出"取代过去的重建集合或双写"。
🔗 参考：https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/nullable-and-default.md
⭐ 重点：百万行级也是无停机热路径；默认值与可空语义。
🚫 去重边界：Chapter 3（3.3.3 创建 Schema / 3.3.6 动态字段）讲的是建表时定 schema；此处只讲**建完之后改**。
EDITOR:END -->

### 7.2.5.2 双向回填

<!-- EDITOR:BEGIN 7.2.5.2
🎯 要求：分两小节讲清"外部回填"与"内部回填"，各给一段代码。
🔗 参考：https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/snapshots.md
⭐ 重点：外部回填=值在库外算好后灌入（可借快照做时间点灌数）；内部回填=给已有集合挂 BM25/MinHash 函数，由内核派生新字段值。
EDITOR:END -->

### 7.2.5.3 改造前后对照

<!-- EDITOR:BEGIN 7.2.5.3
🎯 要求：一张两列表（2.6 做法 / 3.0 做法），加一段 > ⛳ **架构演进提示**。这是本章最好教学点，务必用表格不用长段落。
🔗 参考：https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/alter-collection-field.md
EDITOR:END -->

## 7.2.6 本章小结

<!-- EDITOR:BEGIN 7.2.6
🎯 要求：`本章核心掌握：` + 3–5 条 + 衔接句（"7.3 在此数据底座上看数据模型与检索能力"）。
⚠️ 自检：全章是否只写增量？外部集合/快照/在线演进三块是否各只出现一处，未交叉重复？
EDITOR:END -->
