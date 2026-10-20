# Chapter 7.3 Milvus 3.0 数据模型与检索增强

<!-- EDITOR:BEGIN 本子章节定位
写什么：StructArray、可空向量/默认值/TIMESTAMPTZ/行级 TTL、TEXT 与四类匹配算子、MinHash DIDO、服务端排序/聚合/分面、服务端重排函数链、中文检索增强。
不写：向量/索引/Schema 基础概念（Ch2、Ch3），基于 BM25 的混合检索入门（Ch4），客户端 Reranker 用法（Ch6）。
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

## 7.3.1 学习目标

<!-- EDITOR:BEGIN 7.3.1
【本节要写什么】4–6 条，动词起头。建议：用 StructArray 建模"一文档多切片" / 判断该用可空向量还是默认值 / 用一次请求拿到排序+统计+分面结果 / 把重排从客户端挪到服务端 / 给中文语料配一套分析器与词典。
【必须注意】目标里的"一次请求"由 7.3.5 兑现；兑现不了就改目标，不要硬凑。
【写完自检】4–6 条目标都能在本页找到对应节；读者能据此判断要不要读本章。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.2 数组结构体（StructArray）

多向量这条线，7.0 只开了一个头。这里把 7.0 那一整段原文摆在最前面——它讲清了"是什么、有几种检索粒度"，读完它，后面几节的定义、写入、索引才有着落点：

最后一件是**多向量**——它最容易被当成“3.0 的全新能力”，而它在 2.6 线就已经存在；不先把“是什么、有几种检索粒度”说明白，后面 7.3 的示例就只能照着代码抄，抄完也不知道自己用的是哪一种。

**先看它的两种检索粒度。** 一个实体带多个向量，并不等于"3.0 才有能力存多个向量"——**结构数组（StructArray）与相关检索在 2.6 线已经出现**。v2.6.18 写着：

> You can now run vector search on Struct Array fields at the granularity of individual elements instead of the whole row, with each result reporting the matched element's offset within the array.

v2.6.16 还有一条：Bumped Go SDK to v2.6.4 with full struct-array support (vector sub-fields, EmbeddingList search, schema validation)。所以 3.0 做的是**在这套结构上继续加能力**，官方那句动机是对旧假设的批评，而不是对 2.x 能力的描述：

> The "one entity = one vector" assumption no longer fits modern retrieval. Long documents get split into many chunks, late-interaction models such as ColBERT emit one vector per token, and multimodal entities can carry several views.

> 一个实体对应一个向量的假设已经不再适用现代检索需求，长文档可能拆成多个片段；ColBERT这类late-interaction模型会为每个token生成一个向量；多模态实体也可能包含多个视角或表示。

要真正理解这条主线，得先分清**两种检索粒度**——它们共用一份嵌套数据，但查询形态和返回结果完全不同。

> 这里的嵌套数据的含义是：数据并不是所有的字段都平铺在实体上，而是实体里面还有一组结构化子项。例如文章实体里面有多个chunks，每个chunks有文本、页码、向量等字段。

> 下面提到的StructArry就是用来存储这组子项的字段类型

**其一，按实体出结果（EmbeddingList 搜索）。** 把 StructArray 某个向量子字段里的向量整体看作"这条实体的向量列表"，查询本身也是一个向量列表，Milvus 用 `MAX_SIM*` 指标族比较两个列表，返回**命中的实体**。官方定义：

> EmbeddingList search treats the vectors inside a StructArray vector subfield as one embedding list for the parent entity. The query is also an embedding list. Milvus compares the query embedding list with the stored embedding list by using a `MAX_SIM*` metric and returns matching entities.

> EmbeddingList搜索会把StructArray向量子字段中的多个向量视为所属父实体的一份向量列表。查询本身也以向量列表的形式提供。Milvus使用MAX_SIM*指标比较查询向量列表与已存储的向量列表，并返回匹配的实体。
> > 这里举个例子，比如一篇文章被拆分成为多个片段，每个片段都有一个向量表示；这些向量合起来就是这篇文章的向量列表。文档示例中，StructArray可以把多个chunk及其向量存放在同一个父实体下。

> >  查询的时候，用的是一组向量，ColBERT会把查询拆成token并为其生成向量列表，再与文档的向量列表比较。

> > MAX_SIM* 是embeddingList搜索的时候使用的一类指标，它负责比较查询向量和实体向量列表，搜索结果依然按照父实体返回，*号表示一类指标，比如MAX_SIM_COSINE、MAX_SIM_IP 和 MAX_SIM_L2。具体分数怎么算，本篇不展开公式。这里只记两条结论：**查询是向量列表；返回的是父实体，不是其中某个元素。** 它对应的典型用法是 late interaction——ColBERT 给每个 token、ColPali 给每个 patch 各生成一个向量，匹配的细粒度发生在列表内部，但结果仍然落到文档或页面这一级。

**其二，按元素出结果（element-level 搜索）。** StructArray 里的每个 Struct 元素各自作为独立候选，命中就返回那个元素，未分组时还能给出它在数组中的下标。官方定义：

> Element-level search treats each Struct element as an independent vector-search candidate. Each hit represents a matched element inside the StructArray field, and ungrouped results can expose the element offset.

此时查询是**普通向量**，指标也是普通向量指标（`COSINE` / `IP` / `L2`），返回的是"最相关的那一段、那一页、那个 patch"。

所以两种方式不是新旧替代，而是**结果粒度不同**。选哪个，取决于你要的结果长什么样：**目标是实体级结果，就用 EmbeddingList 搜索；要返回匹配的 Struct 元素（带 offset），就用 element-level 搜索。**

> **注意**：一条向量字段或向量子字段**只能创建一个索引**，而两种搜索用的指标不同。官方因此明确要求：同一集合既要实体级又要元素级，就**拆成两个向量子字段**。原文：If your collection needs both EmbeddingList search and element-level search, use two separate vector subfields. A vector field or vector subfield accepts only one index, and the two search modes require different metric families.

### 7.3.2.1 它解决什么建模问题

<!-- EDITOR:BEGIN 7.3.2.1
【本节要写什么】3–6 行 + 分点写清，不要做成表格（"一文档多切片"在 2.6 的三种凑合写法 vs 3.0 的 StructArray），点明"父子同集合、一次写入、检索时按子元素返回"。
【先读这些】https://milvus.io/docs/array-of-structs.md ｜ https://milvus.io/docs/best-practices-for-array-of-structs.md
【必须注意】
・Ch3（3.3.6 动态字段 / JSON 字段）讲过"弱结构扩展"，此处只对比"嵌套数组对象"这一新类型，不重讲 JSON。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.2.2 定义与写入

<!-- EDITOR:BEGIN 7.3.2.2
【本节要写什么】两段可跑代码（建集合含 struct 字段 → 插入嵌套数据），注释写清嵌套层级；给一段"打印返回结构"的观察代码。
【先读这些】https://milvus.io/docs/create-structarray-field.md ｜ https://milvus.io/docs/insert-data-into-structarray-fields.md
【必须注意】
・️ 嵌套层数、子字段类型上限等以官方页现值为准，不要凭印象写数字。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.2.3 给子字段建索引

<!-- EDITOR:BEGIN 7.3.2.3
【本节要写什么】一段代码 + 一句"哪些子字段值得建索引、哪些不需要"。
【先读这些】https://milvus.io/docs/index-structarray-fields.md ｜ https://milvus.io/docs/index_selection.md
【必须注意】
・索引类型选择原理在 Ch2（2.3.3），此处只讲 struct 子字段的索引写法差异。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.2.4 四种检索姿势

<!-- EDITOR:BEGIN 7.3.2.4
【本节要写什么】小节内四个 `#### 1. xxx`，每个 5–10 行代码，统一用同一份样例语料，避免每次换数据。
【先读这些】https://milvus.io/docs/basic-vector-search-with-structarray.md ｜ https://milvus.io/docs/filtered-search-with-structarray.md ｜ https://milvus.io/docs/grouping-search-with-structarray.md ｜ https://milvus.io/docs/hybrid-search-with-structarray.md ｜ https://milvus.io/docs/range-search-with-structarray.md
【必须注意】
・按子元素命中返回、带过滤的 struct 检索、按父文档分组去重、多向量混合。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.2.5 操作符

<!-- EDITOR:BEGIN 7.3.2.5
【本节要写什么】逐个算子各写一条：算子名、作用、示例表达式，正文不超过 15 行。
【先读这些】https://milvus.io/docs/struct-array-operators.md
【必须注意】算子只列官方 text-match 页面里给出的那几类，别自造函数名；示例表达式必须是可执行语法。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.2.6 限制清单

<!-- EDITOR:BEGIN 7.3.2.6
【本节要写什么】把官方限制逐条抄成中文清单（数量上限、不支持的操作、与动态字段/分区键的交互限制）。这一节是本子章节**最该存在**的内容，别省略。
【先读这些】https://milvus.io/docs/structarray-limits.md ｜ https://milvus.io/docs/best-practices-for-array-of-structs.md
【必须注意】
・️ 用 > ⚠️ **注意** 收口，明确"先查限制再设计 schema"的顺序。
【写完自检】读者只看这一节就能做出选择；这段内容能反复翻看，不必回正文对照。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

7.0 说的第二条限制，属于这里：

**二是"改集合字段"与"改 StructArray 内部结构"不是一回事。** 集合级字段可以整体增删，但子字段不行。官方原文：You can drop a whole `ARRAY<STRUCT>` field, but you cannot drop an individual sub-field inside an `ARRAY<STRUCT>` field. 加子字段同样不行：After a StructArray field is created, you cannot add new subfields to that existing StructArray field. 需要调整子字段结构时，官方给的做法是先 `drop_collection_field()` 删掉整个 StructArray，再按新的 Struct schema 重新添加。

所以准确的说法是：**3.0.0 让集合级字段的变更可以在服务运行中完成；它并没有改变"StructArray 的内部结构只能整体重建"这条约束。**


## 7.3.3 字段属性：可空 · 默认值 · 时区 · 行级 TTL

> 本节只改字段属性，不改列结构；加列 / 删列 / 回填见 7.2.5。

### 7.3.3.1 可空向量与默认值

<!-- EDITOR:BEGIN 7.3.3.1
【本节要写什么】先讲场景（部分行暂无向量、嵌入模型上线前占位、稀疏向量可选），再给代码。
【先读这些】https://milvus.io/docs/nullable-and-default.md
【必须注意】
・边界：在线加列本身在 7.2.5 讲过，此处只讲 NULL/default 语义，勿重复 API 流程。
・节首必须写划界语：这里改字段属性，不改列结构；加列/删列见 7.2.5。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.3.2 TIMESTAMPTZ 字段

<!-- EDITOR:BEGIN 7.3.3.2
【本节要写什么】讲清与 Int64 存时间戳的区别（时区语义、按时间过滤/过期的可用性），给一段建字段+过滤查询。
【先读这些】https://milvus.io/docs/timestamptz-field.md
【必须注意】TIMESTAMPTZ 是 3.0 新类型，别写成"带时区的字符串"；时间字面量写法照官方示例，并写清按 UTC 还是会话时区解释。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.3.3 行级 TTL

<!-- EDITOR:BEGIN 7.3.3.3
【本节要写什么】分两段对照写：集合级 TTL 与行级 TTL 各自的作用范围、配置位置、依赖字段、清理粒度——这是最容易写混的地方。
【先读这些】https://milvus.io/docs/set-collection-ttl.md （找页内小节 "Set entity-level TTL"） ｜ https://milvus.io/docs/timestamptz-field.md
【必须注意】
・Ch3（3.10 数据过期管理 = 集合级 TTL）已写旧机制，**只写差异，不重讲集合级用法**。
・️ 必须提醒：过期不等于合规删除，敏感数据另议。
・只写与集合级 TTL 的差异（Ch3 已讲集合级）；提醒"过期 ≠ 合规删除"。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.4 长文本字段 TEXT 与匹配算子

TEXT 这一项，7.0 把官方的定位讲得最透，原文搬来当引子，字段分界与算子在下面两节展开：

**三是 TEXT 长文本字段。** 官方这一项的小标题就叫 TEXT long-text fields，原话是：

> TEXT fields make long text first-class, with storage-side length limits removed: they support `text_match`, `phrase_match`, and BM25.

紧接着是存储细节：`Values under 64 KB stay inline; larger values go to partition-level LOB files in Vortex format, with the column storing only (file_id, offset) references.`，而且 LOB 文件跨段共享，压缩整理时搬的是引用、不是重写正文。对写教程的人来说，这项能力真正省事的地方在官方最后那句：`For RAG this means retrieving vectors and source text from the same store in one IO — no external blob store to operate.`——**向量与原文在同一个库里一次取回**，不必再自己维护一个对象存储专门放原文。它为什么算“数据模型扩展”：字段类型本身装得下长文本了，匹配算子也随之进到这一类字段上。字段分界与算子用法在 7.3.4 展开，这里不重复。

### 7.3.4.1 TEXT 与 VarChar 的分界

<!-- EDITOR:BEGIN 7.3.4.1
【本节要写什么】分点写清，不要做成表格（类型/长度上限/可建什么索引/可用于哪些算子/存储位置）。
【先读这些】https://milvus.io/docs/text.md ｜ https://milvus.io/docs/string.md
【必须注意】
・TEXT 去掉短文本长度限制，短的内联、超长的走大对象存储且 compaction 只搬引用；这是 RAG 长文场景的关键变化。
・VarChar 基础在 Ch2（2.2.4 标量字段）/Ch3（3.3.4），此处只讲新增类型差异。
・TEXT 内联阈值默认 65,536 B（dataNode.text.inlineThreshold），超出写 LOB；写前再核一次现值。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.4.2 四类匹配算子

<!-- EDITOR:BEGIN 7.3.4.2
【本节要写什么】四段各 5–8 行代码，统一同一份中文+英文混合语料。
【先读这些】https://milvus.io/docs/keyword-match.md ｜ https://milvus.io/docs/phrase-match.md ｜ https://milvus.io/docs/pattern-matching.md ｜ https://milvus.io/docs/membership-match.md
【必须注意】
・关键词 / 短语（带邻近语义）/ 模式匹配 / 集合成员判定（配合 Bloom·Roaring 位图构建器）。
・Ch3（3.6.2 字符串匹配与数组字段查询操作符）只讲过 like/==/in 基本式，**算子语义升级部分才写**。
・只写语义升级部分，基础过滤在 Ch3 已讲；membership 要提位图构建器（Bloom/Roaring）。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.4.3 MinHash DIDO：文档进·文档出查重

<!-- EDITOR:BEGIN 7.3.4.3
【本节要写什么】给一个"原始语料 → 自动签名 → 近重复检索 → 去重后入库"的完整小闭环（15–25 行）。
【先读这些】https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/minhash-lsh.md
【必须注意】
・DIDO=输入文档、输出签名向量，函数由内核驱动；典型场景=语料清洗、爬取内容去重。
・Ch4 的 BM25 混合检索（含稀疏向量用法）不重述，只写 MinHash 这条新链路。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.5 服务端排序 · 聚合 · 分面

### 7.3.5.1 检索结果按标量字段排序

<!-- EDITOR:BEGIN 7.3.5.1
【本节要写什么】先给"2.6 只能客户端排"的痛点，再给 3.0 写法；给多字段升降序 + 与 topk 交互的说明。
【先读这些】https://milvus.io/docs/single-vector-search.md （找页内小节 "Sort Search Results by Scalar Fields"）
【必须注意】
・Ch3（3.5.1 普通向量检索）不重讲，只写排序参数。
・必须写清排序发生在 topk 之内还是之外——这是读者最容易误解的点。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

下面几段从 7.0 原样迁来，从排序说起：

分面之外，另外四项各自补的是另一个缺口。

**服务端 ORDER BY** 的官方原话是 ：

`Search and Query now accept multi-field ordering, with the sort pushed down into the Milvus kernel and ASC / DESC settable per field.`，

它同时点出了要补的坑：`Top-K by distance alone often does not match the business need when the most similar item is not the cheapest, the most recent, or the most popular.`。

说人话就是：和你查询最像的东西不一定是最便宜、最新或最热的；以前“按距离取 top 再按价格重排”只能多取一批回然后在本地计算，现在把排序条件直接写进请求。


### 7.3.5.2 查询侧排序与聚合

<!-- EDITOR:BEGIN 7.3.5.2
【本节要写什么】一段代码给出 COUNT/SUM/AVG/MIN/MAX + group-by 的返回样例（贴真实输出 JSON）。
【先读这些】https://milvus.io/docs/get-and-scalar-query.md （找页内小节 "Aggregate Query Results"） ｜ https://milvus.io/docs/search-aggregation.md
【必须注意】
・Ch3（3.6.4 统计查询）是客户端聚合、过取的老写法——**必须写清"旧写法 → 新写法"的两步对照**，这是本节最大价值。
・️ 提醒读者：统计口径是"命中集合"而非"全表"时，要写清 filter 的作用范围。
・必须画"旧写法（客户端聚合/过取）vs 新写法（一次请求）"两步对照；写清 filter 决定统计口径。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

第二件是聚合，7.0 的原文：

**查询聚合** 官方给的是 `COUNT / SUM / AVG / MIN / MAX with group-by, evaluated server-side`，并强调 `Aggregation is evaluated server-side after filtering.（聚合是在过滤之后由Milvus服务端执行的）`。这里最关键的是“**过滤之后**”这个顺序——统计对象是符合条件的数据，不是整个集合。


### 7.3.5.3 分面检索

<!-- EDITOR:BEGIN 7.3.5.3
【本节要写什么】给"商品搜索页左侧可点筛选"式的最小例子：一次请求同时拿向量 topK + 各分面 Top 值与计数。
【先读这些】https://milvus.io/docs/release_notes.md#v300（"Search Aggregation and faceted search"） ｜ https://milvus.io/docs/get-and-scalar-query.md
【必须注意】
・️ facet 目前只在 release notes 有定义，官方无独立页；正文不要伪造 API，按 SDK 实测写。
・这是 3.0.0 GA 新增（beta 无），每个分面值由其最优匹配成员代表。
・官方没有独立 facet 页，不要伪造 API 形状；以 SDK 实测返回结构为准，并注明这一点。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.5.4 分组返回

<!-- EDITOR:BEGIN 7.3.5.4
【本节要写什么】与 7.3.2.4 的 struct 分组区分开：这里是普通字段 group_by 透传；5 行代码即可。
【先读这些】https://milvus.io/docs/grouping-search.md ｜ https://milvus.io/docs/multi-vector-search.md ｜ https://milvus.io/docs/rrf-ranker.md
【必须注意】
・Ch3 无 group_by 内容，无需回引；不要和 struct 分组混为一谈。
・与 7.3.2.4 的 struct 分组明确区分（那是按父文档聚合子元素）。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.6 服务端重排函数链

### 7.3.6.1 与客户端 Reranker 的分工

<!-- EDITOR:BEGIN 7.3.6.1
【本节要写什么】分点写清，不要做成表格（重排位置/是否多一跳网络/延迟影响/可组合性/适用场景），用 `> ⛳ **架构演进提示**` 收口。
【先读这些】https://milvus.io/docs/model-ranker-overview.md ｜ https://milvus.io/docs/rerankers-overview.md
【必须注意】
・Ch6《Milvus Reranker 重排》讲的是**外部模型/客户端 ranker**，此处只讲服务端函数链，明确"两条路都在，怎么选"。
・Ch6 讲的是外部模型/客户端 ranker，这里只讲服务端函数链，并回答两条路怎么选。
【写完自检】读者只看这一节就能做出选择；这段内容能反复翻看，不必回正文对照。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

函数链这件事，7.0 只点到为止。原文连同官方讲解视频，一起搬到这里：

> 这里推荐 Zilliz 官方的讲解视频：
【商品推荐中，为什么不能只看相似度？Milvus FunctionChain检索、打分、重排精讲】 https://www.bilibili.com/video/BV16tJC6rENw/?share_source=copy_web&vd_source=2ee6373221bc82cfd994fdd5d96888f5

**Function Chain 重排** 的官方原话是 `Reranking is now composable through the Function Chain API, which executes an ordered, typed pipeline as part of a single search request（可以通过Function Chain API 将多个重排操作按照顺序组合成一个流程，并在一次搜索请求中执行）.`，这一版还加入了原生 XGBoost 打分用于 L0 重排。它替你省掉的是“检索调一次、重排再调一次、再自己把结果拼回来”这条客户端流水线：重排逻辑变成库里一段可声明、可组合的链。细节见 7.3.6。


### 7.3.6.2 声明式重排实操

<!-- EDITOR:BEGIN 7.3.6.2
【本节要写什么】三个小例子递进：融合排名 → 提升/降权某表达式命中 → 外部模型打分融合；每个 8–12 行。
【先读这些】https://milvus.io/docs/rrf-ranker.md ｜ https://milvus.io/docs/weighted-ranker.md ｜ https://milvus.io/docs/boost-ranker.md ｜ https://milvus.io/docs/decay-ranker-overview.md
【必须注意】
・️ 客户端函数链 API 名与参数请对照 pymilvus 3.x 与官方页现值核对后再写，勿凭记忆；模型打分需外部服务在跑，正文注明前置条件。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.7 中文检索体验增强

### 7.3.7.1 文件级词典资源

<!-- EDITOR:BEGIN 7.3.7.1
【本节要写什么】讲"为什么要在集群级注册词典文件"（多库共用、热更新、免重启），给上传→引用→生效三步。
【先读这些】https://milvus.io/docs/manage-file-resources.md ｜ https://milvus.io/docs/analyzer-overview.md
【必须注意】词典是集群级资源，删除前先确认无引用；没生效先查路径与加载状态，不要随口归因于"缓存"。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.3.7.2 中文分析器与过滤链

<!-- EDITOR:BEGIN 7.3.7.2
【本节要写什么】给一套可直接复用的中文 BM25 配置（分析器 + 拼音 + 同义词 + 数字规范化），并给"改前/改后检索效果对照"两行示例结果。
【先读这些】https://milvus.io/docs/chinese-analyzer.md ｜ https://milvus.io/docs/pinyin-filter.md ｜ https://milvus.io/docs/synonym-filter.md ｜ https://milvus.io/docs/decimaldigit-filter.md ｜ https://milvus.io/docs/multi-language-analyzers.md ｜ https://milvus.io/docs/choose-the-right-analyzer-for-your-use-case.md
【必须注意】
・Ch4 已讲 BM25 基本式，此处只写分析器/词典侧增量。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.3.8 本章小结

<!-- EDITOR:BEGIN 7.3.8
【本节要写什么】`本章核心掌握：` + 4–5 条 + 衔接句（"7.4 看索引与写入引擎"）。
【必须注意】
・️ 自检：StructArray/TEXT/聚合/重排四块是否各只出现一次；是否误重述了 Ch3 的统计查询、Ch3 的集合级 TTL、Ch6 的客户端重排。
【写完自检】以「本章核心掌握：」起头；只看小结也能复述本章三条主线。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->
