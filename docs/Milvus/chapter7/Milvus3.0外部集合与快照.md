# Chapter 7.2 Milvus 3.0 湖仓原生：外部集合与快照

<!-- EDITOR:BEGIN 本子章节定位
写什么：Storage V3 怎么开怎么验（原理一笔带过）、外部集合从零建到检索到追平、快照四个用例、在线加列/回填/删列。
边界：只讲 3.0 新增能力；Milvus 基础概念（集合/字段/索引/搜索）见 Chapter 2、Chapter 3，不重述。
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

## 7.2.1 学习目标

<!-- EDITOR:BEGIN 7.2.1
【本节要写什么】4–5 条，动词起头。建议：解释 Storage V3 与上两层新范式的关系 / 独立完成一个外部集合的挂载·建索引·检索·追平 / 会用快照做误操作回滚与版本对照 / 在不停机前提下给已有集合加列并回填。
【必须注意】目标不超过 5 条；每条都要能在本页找到落地节，找不到对应节的目标就删掉。
【写完自检】4–6 条目标都能在本页找到对应节；读者能据此判断要不要读本章。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.2.2 存储引擎 V3：新底座

### 7.2.2.1 为什么先讲底座

<!-- EDITOR:BEGIN 7.2.2.1
【本节要写什么】5–8 行 + 一张"2.x 存储与元数据组织 vs Storage V3"两列简表。
【先读这些】https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/object-storage.md ｜ https://milvus.io/docs/release_notes.md （beta "Storage V3（Loon）"一节）
【必须注意】
・manifest 是唯一真相 + 原子更新；列式对象存储；Vortex 是可选列式格式（默认仍是 parquet，官方建议先做基准测试再切；自适应编码：字典/RLE/bit-packing/浮点压缩、零拷贝解压）；LOB 改为分区级 Vortex 文件跨段共享，compaction 只搬 (file_id, offset) 引用不搬正文。
・Chapter 2（2.3.8 内存与存储优化 / 2.3.9 mmap）与 Chapter 6《milvus 存储优化》讲的是加载与 mmap，属另一层，勿混讲、勿重复。
・️ 开启方法在 7.1.4.3，此处只讲"是什么/为什么"，不重复给配置。
・Vortex 是可选格式，默认仍是 parquet，官方建议先基准测试再切——不要写"Vortex 将成默认"。manifest 是唯一真相＋原子更新；compaction 只搬 (file_id, offset) 引用。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.2.2 验证是否真用上了 V3

<!-- EDITOR:BEGIN 7.2.2.2
【本节要写什么】给一段可跑的"建集合→插入→查询对象存储目录结构"的观察流程，让读者肉眼看到新格式产物。
【先读这些】https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/configure_common.md#commonstorageuseLoonFFI ｜ https://milvus.io/docs/configure_datacoord.md
【必须注意】
・Loon 未开时功能会静默降级或报错，正文要写出"确认已生效"的判据（对象存储里出现新的清单/段文件格式）。
・必须给出"看到什么才算生效"的判据，不能只写"检查日志"。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.2.3 外部集合（External Collection）

### 7.2.3.1 三个关键属性与适用场景

<!-- EDITOR:BEGIN 7.2.3.1
【本节要写什么】三条属性各一句话 + 一张"该用外部集合 / 该用普通集合"的判据表。
【先读这些】https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/data-infra-integration-overview.md
【必须注意】
・原地读（zero-copy，不搬数据）/ 只读（release notes 明确 read-only；官方操作页未提供写入接口）/ 靠增量 refresh 追平。
・️ 只读这点必须前置强调，否则读者第一行代码就想 insert。
・只读要前置强调（官方写明 read-only，无写入接口），否则读者第一行就想 insert。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.2 支持的存储格式与前置准备

<!-- EDITOR:BEGIN 7.2.3.2
【本节要写什么】一张表（格式 / 典型来源 / 需要的凭据与路径形态），加一条 > 📌 **前置条件**。
【先读这些】https://milvus.io/docs/data-infra-integration-overview.md ｜ https://milvus.io/docs/object-storage.md ｜ https://milvus.io/docs/prepare-source-data.md
【必须注意】
・Parquet / Lance / Iceberg / Vortex / milvus-table（GA 新增，可把 Milvus 集合/快照当外部源）；外部字段类型映射在 create 时定；建集合前要有对象存储访问凭据。
・术语口径：这些一律表述为"Milvus 外部集合支持的存储格式"，不介绍对应生态产品本身。
・格式清单：Parquet / Lance / Iceberg / Vortex / milvus-table；milvus-table 可把 Milvus 集合或快照当外部源。字段类型映射在建集合时定死。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.3 实操：挂载一个外部集合

<!-- EDITOR:BEGIN 7.2.3.3
【本节要写什么】最小可跑闭环，用 MilvusClient 风格，中文注释，每段代码前后各一句"做什么/预期输出"。
【先读这些】https://milvus.io/docs/create-an-external-collection.md
【必须注意】
・参数按官方页原样抄，不要凭记忆拼字段名（尤其外部路径、字段映射、主键/向量列的对应关系）。
・普通集合的创建写法见 Chapter 3（3.3.7/3.3.8），此处只给外部集合的差异参数。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.4 在外部集合上建索引与检索

<!-- EDITOR:BEGIN 7.2.3.4
【本节要写什么】两段代码（建索引、search），并点明"索引在库内、数据在库外"这一核心心智。
【先读这些】https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/filtered-search.md
【必须注意】
・外部列上可用全文索引/BM25、标量索引、JSON、动态字段、分区键、clustering key；检索语义与普通集合一致。
・索引类型科普在 Chapter 2（2.3.3），此处不重述，只讲外部集合上的可用性差异。
・写清"索引在库内、数据在库外"；外部列可用全文/BM25、标量、JSON、动态字段、分区键、clustering key。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.5 外部列直连函数输出字段

<!-- EDITOR:BEGIN 7.2.3.5
【本节要写什么】本子章节最有卖点的一节。给"文本留在湖里、检索字段在库里生成"的完整例子（BM25 稀疏向量为首选）。
【先读这些】https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/embedding-function-overview.md ｜ https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/create-an-external-collection.md
【必须注意】
・3.0.0 起外部列可直接作为函数输入，产出 BM25 稀疏向量、MinHash 签名、text embedding；这解决了"湖上原文 + 库内检索表示"的双写难题。
・️ 外部 embedding 函数对远端模型的批量调用开销不小，正文给一句性能提示。
・函数语法不要在这里展开，回引 7.3.4.4；远端 embedding 有调用成本，给一句性能提示。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.6 增量 refresh：让外部数据追平

<!-- EDITOR:BEGIN 7.2.3.6
【本节要写什么】给 refresh 的调用 + "湖上改一行→refresh→检索结果变化"的对照验证。
【先读这些】https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/alter-external-collection-schema.md
【必须注意】
・元数据同步走增量流程；refresh 的粒度、耗时与何时该刷（正文给决策建议）。
・️ 注意：官方 v3.0.x 文档中 refresh 相关内容并入 create-an-external-collection.md 页（无独立 refresh 页），落笔前以该页现文为准。
・说明 refresh 不是实时，给出粒度与耗时量级（实测）以及"什么时候该刷"。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.7 外部集合的 schema 调整

<!-- EDITOR:BEGIN 7.2.3.7
【本节要写什么】短小节（10 行内）。讲湖上加了列怎么办、字段类型不兼容怎么办。
【先读这些】https://milvus.io/docs/alter-external-collection-schema.md
【必须注意】
・与 7.2.5 的普通集合在线加列区分开——这里专指外部集合。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.3.8 常见报错与排查

<!-- EDITOR:BEGIN 7.2.3.8
【本节要写什么】3–5 条 `#### 1. xxx` 式排错条目，每条=现象/原因/处理。建议覆盖：凭据或路径错、格式不匹配、在只读集合上写入、外部字段映射不到、refresh 后无变化。
【先读这些】https://milvus.io/docs/create-an-external-collection.md（限制与报错节） ｜ https://milvus.io/docs/limitations.md
【必须注意】
・把你实跑时遇到的真实报错原文贴进来（比编造更有价值）。
・贴真实报错原文；每条按"现象/原因/处理"三行写，超过 6 行就删。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.2.4 快照（Snapshot）

### 7.2.4.1 快照是什么

<!-- EDITOR:BEGIN 7.2.4.1
【本节要写什么】3–6 行 + 一张"快照 / 备份 / 副本"三者区别表（易混，必须辨析）。
【先读这些】https://milvus.io/docs/manage-snapshots.md ｜ https://milvus.io/docs/snapshot-use-cases.md
【必须注意】
・时点只读视图；按段引用而非复制，创建只写元数据/manifest、不复制向量数据（近零）；但快照会钉住被引用文件不被 GC，保留成本≈集合大小，极端可达 2×，需定期清理；依赖 Storage V3 已开启。
・与 Chapter 3（3.9 分区）无关，不要为省事拿分区当快照讲。
・快照只写元数据、不复制向量，但会钉住被引用文件不被 GC；保留成本约等于集合大小，极端可翻倍。不要写"边际存储近零"。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.4.2 实操：创建·列举·恢复·删除

<!-- EDITOR:BEGIN 7.2.4.2
【本节要写什么】四段短代码，每段一句说明；参数名以官方页为准。
【先读这些】https://milvus.io/docs/manage-snapshots.md
【必须注意】
・create / list / restore / drop 四个动作；恢复的目标是"新集合"还是"原集合"要写清楚。
・术语提醒：Lite 的 snapshot 是嵌入式引擎机制，与服务端快照不同（见 7.1.5.1），此处加 > 💡 **版本说明** 一句话即可，别展开。
・参数名以 manage-snapshots.md 为准；写清恢复到新集合还是原集合。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.4.3 四个典型用例

<!-- EDITOR:BEGIN 7.2.4.3
【本节要写什么】每个用例=场景一句话 + 3–6 行代码，不贴整页 API 表。
【先读这些】https://milvus.io/docs/snapshot-use-cases.md ｜ https://milvus.io/docs/manage-snapshots.md
【必须注意】
・误删回滚 / A-B 版本对照检索 / 灌数据中途失败回退到干净基线（对应官方 2.6→3.0 灌数据失败案例）/ 配合备份做时间点还原。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.4.4 快照与备份恢复的分工

<!-- EDITOR:BEGIN 7.2.4.4
【本节要写什么】一张对照表（解决什么问题/是否跨实例/是否依赖对象存储/可回滚粒度），正文不超过 12 行。
【先读这些】https://milvus.io/docs/multi-storage-backup-and-restore.md ｜ https://milvus.io/docs/cross-bucket-backup-and-restore.md ｜ https://milvus.io/docs/shared-bucket-backup-and-restore.md ｜ https://milvus.io/docs/single-instance-backup-and-restore.md
【必须注意】
・Chapter 6《Milvus底层架构详解》讲的是架构容错，此处只讲数据面备份恢复，勿重述架构。
・只做分工表，不重述 Ch6 的架构容错。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.2.5 Schema 在线演进：加列·回填·删列

### 7.2.5.1 在线加列与删列

<!-- EDITOR:BEGIN 7.2.5.1
【本节要写什么】一段代码贯穿"加列→服务不中断可查→删列"，并明确写出"取代过去的重建集合或双写"。
【先读这些】https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/nullable-and-default.md
【必须注意】
・百万行级也是无停机热路径；默认值与可空语义。
・Chapter 3（3.3.3 创建 Schema / 3.3.6 动态字段）讲的是建表时定 schema；此处只讲**建完之后改**。
・新增的用户自定义向量字段必须 nullable，已有实体该字段为 NULL，检索会跳过；要可检索须 upsert 或回填。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.5.2 双向回填

<!-- EDITOR:BEGIN 7.2.5.2
【本节要写什么】分两小节讲清"外部回填"与"内部回填"，各给一段代码。
【先读这些】https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/minhash-function.md
【必须注意】
・外部回填=值在库外算好后灌入（可借快照做时间点灌数）；内部回填=给已有集合挂 BM25/MinHash 函数，由内核派生新字段值。
・三个硬事实：①内部回填由后台 compaction 异步完成，describe_collection() 看到新字段≠回填完成；②给已有集合加 Function 需 schema version compaction 与 storage version compaction 同时开启，任一关闭请求被拒；③BM25/MinHash 输入必须是 VARCHAR，TEXT 不支持该回填路径。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.2.5.3 改造前后对照

<!-- EDITOR:BEGIN 7.2.5.3
【本节要写什么】一张两列表（2.6 做法 / 3.0 做法），加一段 > ⛳ **架构演进提示**。这是本章最好教学点，务必用表格不用长段落。
【先读这些】https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/alter-collection-field.md
【必须注意】左列必须是 2.6 的真实做法（不是简化版），右列每一步都要能指到 7.2.3/7.2.4 的节号；不要把 2.6 写成"做不到"。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.2.6 本章小结

<!-- EDITOR:BEGIN 7.2.6
【本节要写什么】`本章核心掌握：` + 3–5 条 + 衔接句（"7.3 在此数据底座上看数据模型与检索能力"）。
【必须注意】
・️ 自检：全章是否只写增量？外部集合/快照/在线演进三块是否各只出现一处，未交叉重复？
【写完自检】以「本章核心掌握：」起头；只看小结也能复述本章三条主线。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->
