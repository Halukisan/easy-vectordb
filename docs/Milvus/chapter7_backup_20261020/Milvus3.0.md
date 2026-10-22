# Chapter 7.0 Milvus 3.0：从 2.x 走过来

<!-- EDITOR:BEGIN 
   状态：草稿。未写入 docs/.vitepress/config.js 的 sidebar / nav。
   
   { text: 'Chapter 7 · Milvus 3.0x 新特性', items: [
     { text: '从 2.x 走过来', link: '/Milvus/chapter7/Milvus3.0' },
     { text: '版本演进与环境升级', link: '/Milvus/chapter7/Milvus3.0版本演进与环境升级' },
     { text: '外部集合与快照', link: '/Milvus/chapter7/Milvus3.0外部集合与快照' },
     { text: '数据模型与检索增强', link: '/Milvus/chapter7/Milvus3.0数据模型与检索增强' },
     { text: '索引与写入引擎', link: '/Milvus/chapter7/Milvus3.0索引与写入引擎' },
     { text: '端到端实战', link: '/Milvus/chapter7/Milvus3.0端到端实战' } ] },

EDITOR:END -->

本章将介绍Milvus 3.0 新版本，AI时代，是否还有必要手动编写教程一直都是值得我们DW思考的事情，大可以告诉自己的 AI 助手：“帮我查询Milvus官方文档，告诉我3.0版本都更新了什么”。然后AI会给你输出一大串的内容，详细的告诉你新版本都更新了什么，并且还会附带一张和旧版本对比的表格。下意识的指令和流水似的答案，无论是输入还是输出都没有神经元的信号跳动，答案宛如回家吃饭一样顺其自然的出现，“妈妈会做好饭然后喊我吃饭的”，日复一日的习惯，养成了衣来伸手饭来张口的自己，自己的能力慢慢退化，品尝这些饭菜，思考是如何做的，哪怕和“妈妈”对话，问做这个菜的时候放了些什么，盐放了多少......


本章节会采用分总的形式编写，不会直接像AI一样陈列新版本的变动，静下心来阅读这些内容。



Milvus 3.0 于 2026 年 7 月 29 日发布首个正式版本（v3.0.0），截至本文撰写(2026.10.19)，3.0 线已经走到 v3.0.3。作为一个 major 版本，它带来的不是"多几个参数、多几个索引类型"，而是把几件在 2.x 里**必须由应用在库外完成的事**，变成了数据库内部的一等能力。

本篇不重复后面五篇的操作细节，只解决三个问题：

- 3.0 在整个版本演进里处于什么位置，和 2.6 是什么关系；
- 哪些看起来"新"的能力其实 2.5 / 2.6 就已经有；
- 要让这些新能力真正生效，你必须做些什么事情，哪些是默认就有的。

## 7.0.1 学习目标

- 能说出 Milvus 3.0 三条主线分别解决什么问题：数据在哪里处理、检索逻辑在哪里执行、数据与检索模式怎么扩展
- 能区分"3.0 新增"与"2.x 已有"
- 记住官方明确写出的四条兼容性与默认行为边界
- 能判断自己该马上升级、该观望、还是只读文档不动环境

## 7.0.2：Milvus 3.0 的三条主线

3.0 的新特性看着多，本文把它们归成三条主线，分别回答三个问题：**数据在哪里处理、检索逻辑在哪里执行、系统怎样适应新的数据与检索需求**。

**第一条：Lake-native，让计算靠近数据。** 这里的"湖原生"指直接面向数据湖里的数据，不必先搬进 Milvus。在 **外部集合（External Collection）** 出现之前，湖上的数据要先进入 Milvus 才能查；3.0 的外部集合让 Milvus 直接读 Parquet / Lance / Iceberg / Vortex，**不搬数据**。官方对这条能力的定义只有一句：

> **External Collection** — query lakehouse data (Parquet, Lance, Iceberg, Vortex) in place: zero-copy, read-only, synced through incremental refresh.

零拷贝、**只读**、增量刷新，三个限定词都是官方原话。**只读**本身就说明这类集合不用于写入；**增量刷新**说明数据更新靠刷新动作带进来。至于写入走哪条路径、刷新后多久可见，发布说明没有写，**要看外部集合的专门文档**，本篇不在这里下结论。

这条主线上还有两个配套。**Storage V3（Loon）** 是新的存储格式，官方称它是快照与外部集合的"基础"：

> **Snapshot** — point-in-time read-only collection views by segment reference, with near-zero marginal storage.

合起来的效果是：数据不必先搬进来（外部集合），同时还能拿到一份**时点视图**——官方强调的两个词是 point-in-time 与 near-zero marginal storage，也就是"按某个时间点的状态看数据"和"额外存储几乎为零"。

官方在这一节开头交代了它要解决的场景：在线服务和批处理任务需要同时看同一份集合——A/B 评测、大规模查重、回填校验、版本回滚，都要求“写入照常进行，但视图不许动”。快照就是这个不动的参照点：它引用已有的段而不是复制数据，所以边际存储几乎为零，批任务在快照里读、活集合照常写。细节见 7.2。

> ⚠️ **注意**：**Storage V3（Loon）默认关闭**，要用依赖它的能力，得在配置里把 `common.storage.useLoonFFI` 打开。

**第二条：检索引擎增强，把更多检索与结果处理放进库内。** 这类需求过去很容易落到应用侧：先多取一批结果，再在本地排序、统计或重排。3.0 把这几件事搬进库内：**服务端 ORDER BY**、**查询聚合**、**搜索时分面返回**、**稀疏索引全面换代**（SINDI、Block-Max WAND、Block-Max MaxScore），以及声明式的 **Function Chain 重排**（含原生 XGBoost 打分），细节见 7.3 与 7.4。官方对分面的描述顺带把"为什么要搬进库内"说清了：

> Specify a facet field at search time and Milvus returns the top facet values, each represented by its best-matching member in ANN ranking and annotated with aggregates such as COUNT and AVG — the faceted-search sidebar (brand, price range, attributes) in one request, instead of over-fetching and counting client-side.

> 可以在搜索请求中指定一个分面字段，Milvus 会返回该字段的热门取值；每个取值由 ANN 排名中最匹配的实体代表，并附带 COUNT、AVG 等统计值。这样应用可以在一次请求中生成类似按品牌、价格区间或属性筛选的分面栏，而不必先多取一批结果，再由客户端计算统计值。

分面之外，另外四项各自补的是另一个缺口。
> 这里推荐 Zilliz 官方的讲解视频：
【商品推荐中，为什么不能只看相似度？Milvus FunctionChain检索、打分、重排精讲】 https://www.bilibili.com/video/BV16tJC6rENw/?share_source=copy_web&vd_source=2ee6373221bc82cfd994fdd5d96888f5

**服务端 ORDER BY** 的官方原话是 ：

`Search and Query now accept multi-field ordering, with the sort pushed down into the Milvus kernel and ASC / DESC settable per field.`，

它同时点出了要补的坑：`Top-K by distance alone often does not match the business need when the most similar item is not the cheapest, the most recent, or the most popular.`。

说人话就是：和你查询最像的东西不一定是最便宜、最新或最热的；以前“按距离取 top 再按价格重排”只能多取一批回然后在本地计算，现在把排序条件直接写进请求。

**查询聚合** 官方给的是 `COUNT / SUM / AVG / MIN / MAX with group-by, evaluated server-side`，并强调 `Aggregation is evaluated server-side after filtering.（聚合是在过滤之后由Milvus服务端执行的）`。这里最关键的是“**过滤之后**”这个顺序——统计对象是符合条件的数据，不是整个集合。

**稀疏索引改进** Milvus 3.0.0 为稀疏向量索引引入 SINDI、Block-Max WAND 和 Block-Max MaxScore，并加入倒排列表压缩、可配置量化和按负载选择搜索算法。官方内部基准显示：在召回率相近时，压缩后的 BM25 索引约为 2.6 稀疏索引大小的三分之一；在学习稀疏向量上，SINDI 的 QPS 最高约为 MaxScore 的 10 倍。这些是内部基准结果，并非本教程复测。新引入的索引算法需要手动提高目标索引版本后才会生效；具体配置和用法请核对相应章节7.4.2。

**Function Chain 重排** 的官方原话是 `Reranking is now composable through the Function Chain API, which executes an ordered, typed pipeline as part of a single search request（可以通过Function Chain API 将多个重排操作按照顺序组合成一个流程，并在一次搜索请求中执行）.`，这一版还加入了原生 XGBoost 打分用于 L0 重排。它替你省掉的是“检索调一次、重排再调一次、再自己把结果拼回来”这条客户端流水线：重排逻辑变成库里一段可声明、可组合的链。细节见 7.3.6。

**第三条：数据与检索模式扩展，适应持续演进的数据与多向量场景。** 这条主线是三件事：**在线改 schema**（加字段 / 删字段 / 回填）、**TEXT 长文本字段**、**多向量（StructArray）**。官方对第一件（在线改 schema）的原话是：

> 3.0.0 closes the loop: columns can be added, filled, and dropped while serving continues.

> 在 Milvus 提供服务期间，可以在线添加字段、为字段回填数据，并删除字段。

这句要说得准确：它支持的是**在服务持续期间做字段级的增删与回填**，不能理解成"任何 schema 变更都不用停服、不用重建 collection"。具体到操作，有两条限制值得记住。

**一是新增字段必须可空，回填也不是自动的。**

官方原文：Added user-defined fields must be nullable. Set `nullable=True` when calling `add_collection_field()`. For existing entities, the added field is `NULL` unless you add a scalar field with a `default_value`. 

也就是说，给已经有数据的 collection 加字段，老数据不会自动"长"出新值。把它想成给一张已经有 100 万行的表加一列：

- **这一列对老数据是空的（`NULL`）。** 注意 `NULL` 不是 0，也不是空字符串，而是"这里没有值"。数据库不会猜你想填什么，所以它就空着。
- **唯一会自动有值的情况**：新加的是标量字段（int、float、varchar 这类），而且你在添加时给了默认值。这样 100 万条老数据会统一读到这个默认值。
- **如果要给每条老数据填不同的值，就只能自己补。** 官方给的两条路是 upsert（按主键把新值写回对应实体）或者回填流程。加完字段就放着不管，老数据会一直是 `NULL`。
- **如果新加的是向量字段**，老数据在补上向量之前，按这个字段检索时会被直接跳过——不是排在末尾，而是根本不出现在结果里。

一句话：**加字段只是开了一个空槽；填不填、填什么，取决于有没有给默认值、有没有补写数据。**

**二是"改集合字段"与"改 StructArray 内部结构"不是一回事。** 集合级字段可以整体增删，但子字段不行。官方原文：You can drop a whole `ARRAY<STRUCT>` field, but you cannot drop an individual sub-field inside an `ARRAY<STRUCT>` field. 加子字段同样不行：After a StructArray field is created, you cannot add new subfields to that existing StructArray field. 需要调整子字段结构时，官方给的做法是先 `drop_collection_field()` 删掉整个 StructArray，再按新的 Struct schema 重新添加。

所以准确的说法是：**3.0.0 让集合级字段的变更可以在服务运行中完成；它并没有改变"StructArray 的内部结构只能整体重建"这条约束。**

**三是 TEXT 长文本字段。** 官方这一项的小标题就叫 TEXT long-text fields，原话是：

> TEXT fields make long text first-class, with storage-side length limits removed: they support `text_match`, `phrase_match`, and BM25.

紧接着是存储细节：`Values under 64 KB stay inline; larger values go to partition-level LOB files in Vortex format, with the column storing only (file_id, offset) references.`，而且 LOB 文件跨段共享，压缩整理时搬的是引用、不是重写正文。对写教程的人来说，这项能力真正省事的地方在官方最后那句：`For RAG this means retrieving vectors and source text from the same store in one IO — no external blob store to operate.`——**向量与原文在同一个库里一次取回**，不必再自己维护一个对象存储专门放原文。它为什么算“数据模型扩展”：字段类型本身装得下长文本了，匹配算子也随之进到这一类字段上。字段分界与算子用法在 7.3.4 展开，这里不重复。

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


---
三条主线到此讲完，落到动手层面只有一句话：**第一条改的是数据放在哪里、怎么进库；第二条改的是查询请求里能写什么；第三条改的是集合本身长什么样。** 第二、三条要动应用代码，第一条主要动部署与数据布局。至于哪些能力必须先开开关才真正存在，见 7.0.4。

3.0 的能力**并不是一升级就全部生效**：官方在 v3.0.0 的 *Compatibility and behavior notes* 里只明确写出了四条默认行为，逐条看原文与含义。

**一、存储引擎 V3 默认关闭。** 原文：

> **Storage V3 (Loon) is disabled by default.** Features that depend on it — such as Snapshot and TEXT fields — require enabling it manually via `common.storage.useLoonFFI`. Storage V3 will be enabled by default in a later release.

含义是：装好 3.0 之后，**快照和 TEXT 长文本字段这些功能实际上并未启用**，必须手动打开 `common.storage.useLoonFFI` 才会出现。

**二、2.6 → 3.0 保证兼容，也能回滚——但有个一次性的开关。** 原文：

> **2.6 → 3.0 compatibility and rollback are guaranteed** — a 3.0 deployment can be rolled back to 2.6. However, once you enable or use features that change the serialized data format (for example Storage V3), rollback is no longer possible.

也就是说升级本身留了后路，可**一旦启用了改变序列化数据格式的特性（例如 Storage V3），回滚窗口就永久关闭**。

**三、新引入的索引算法默认不生效。** 原文：

> **New index versions are opt-in for now.** Newly introduced index algorithms require manually raising the target index version (`dataCoord.targetVecIndexVersion` to 10, `dataCoord.targetScalarIndexVersion` to 4) before they take effect; a later release will enable them by default.

含义是：新引入的索引算法需要手动提高对应目标索引版本后才生效，也就是把 `dataCoord.targetVecIndexVersion` 抬到 **10**、`dataCoord.targetScalarIndexVersion` 抬到 **4**。

**四、GPU 镜像换到 CUDA 12.9。** 原文只有一句：**GPU images move to CUDA 12.9** and no longer preserve Ubuntu 20.04 GPU compatibility。
