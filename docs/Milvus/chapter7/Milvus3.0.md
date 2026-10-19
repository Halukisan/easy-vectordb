# Chapter 7.0 Milvus 3.0：从 2.x 走过来

<!-- EDITOR:BEGIN 本块在审核通过后整块删除（读者不可见）
   状态：草稿。本篇未写入 docs/.vitepress/config.js 的 sidebar / nav，站点上不展示。
   审核通过后，把本篇排在 '/Milvus/' 数组首位（侧栏顺序由数组顺序决定，与编号无关）：
   { text: 'Chapter 7 · Milvus 3.0x 新特性', items: [
     { text: '从 2.x 走过来', link: '/Milvus/chapter7/Milvus3.0' },
     { text: '版本演进与环境升级', link: '/Milvus/chapter7/Milvus3.0版本演进与环境升级' },
     { text: '外部集合与快照', link: '/Milvus/chapter7/Milvus3.0外部集合与快照' },
     { text: '数据模型与检索增强', link: '/Milvus/chapter7/Milvus3.0数据模型与检索增强' },
     { text: '索引与写入引擎', link: '/Milvus/chapter7/Milvus3.0索引与写入引擎' },
     { text: '端到端实战', link: '/Milvus/chapter7/Milvus3.0端到端实战' } ] },

   排版约定（按你的要求：不用表格）：全篇用段落、无序列表、**加粗**、`> 💡 / ⚠️ / ⛳` 注释块；
   需要精确口径处直接引用官方英文原文（release notes），引用块内保留英文、块外给中文解释。

   请你用浏览器复核两处（本机访问不到 milvus.io，无法核对页面级细节）：
   1) 7.0.4.2 列出的配置项键名与默认值，是否与 v3.0.2 文档页一致；
   2) 各处指向的小节号（7.1.4 / 7.2.2 / 7.3.5 等）与你最终定稿的章节号是否一致。

   已核实的事实来源：milvus-docs 仓库 v3.0.x 与 v2.6.x 的 site/en/release_notes.md 原文、
   两分支 menuStructure/en.json 的页面差集、github.com/milvus-io/milvus 的 tag 页与 releases.atom 日期。
   本轮新核到：**3.0-beta 的发布日期是 2026-05-09**（release notes 原文 "## v3.0-beta / Release date: May 9, 2026"），
   上一版正文只写"2026 年"，现已补上确切日期。

   原文与解释的边界：引用块里的英文一律是官方原文，可追溯到 milvus-docs 的 release_notes.md 或
   site/en/userGuide/schema/array-of-structs.md；凡出现"可以这样理解""本篇不展开公式"的，是教程解释，不是官方结论。
   已据此删掉的作者推断：外部集合的写入路径与可见性时机、快照"可复现"、2.x"服务端只按 top-k"、
   "不能给已有 StructArray 加子字段"（这四处原文都没有）。
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

这条主线上还有两个配套。**Storage V3（Loon）** 是新的存储格式，官方称它是快照与外部库 的"基础"：

> **Snapshot** — point-in-time read-only collection views by segment reference, with near-zero marginal storage.

合起来的效果是：数据不必先搬进来（外部集合），同时还能拿到一份**时点视图**——官方强调的两个词是 point-in-time 与 near-zero marginal storage，也就是"按某个时间点的状态看数据"和"额外存储几乎为零"。

> ⚠️ **注意**：**Storage V3（Loon）默认关闭**，要用依赖它的能力，得在配置里把 `common.storage.useLoonFFI` 打开。

**第二条：检索引擎增强，把更多检索与结果处理放进库内。** 这类需求过去很容易落到应用侧：先多取一批结果，再在本地排序、统计或重排。3.0 把这几件事搬进库内：**服务端 ORDER BY**、**查询聚合**、**搜索时分页返回**、**稀疏索引全面换代**（SINDI、Block-Max WAND、Block-Max MaxScore），以及声明式的 **Function Chain 重排**（含原生 XGBoost 打分~），细节见 7.3 与 7.4。官方对分面的描述顺带把"为什么要搬进库内"说清了：

> Specify a facet field at search time and Milvus returns the top facet values, each represented by its best-matching member in ANN ranking and annotated with aggregates such as COUNT and AVG — the faceted-search sidebar (brand, price range, attributes) in one request, instead of over-fetching and counting client-side.

> 可以在搜索请求中指定一个分面字段，Milvus 会返回该字段的热门取值；每个取值由 ANN 排名中最匹配的实体代表，并附带 COUNT、AVG 等统计值。这样应用可以在一次请求中生成类似按品牌、价格区间或属性筛选的分面栏，而不必先多取一批结果，再由客户端计算统计值。

**第三条：数据与检索模式扩展，适应持续演进的数据与多向量场景。** 这条主线是三件事：**在线改 schema**（加字段 / 删字段 / 回填）、**TEXT 长文本字段**、**多向量（StructArray）**。官方对这一件的原话是：

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

这条主线的另外两件是 **TEXT 长文本字段**和**多向量**。TEXT 与 VarChar 的分界、匹配算子怎么配，放到 7.3.4 展开，这里不重复。需要在这里先讲清的是多向量——原因很实际：它是 3.0 里最容易被当成"全新能力"的部分，而它其实在 2.6 版本就已经存在；不先把它"是什么、有几种检索方式"说明白，后面 7.3 的示例就只能照着代码抄，抄完也不知道自己用的是哪一种粒度。

**先说多向量这件事。** 一个实体带多个向量，并不等于"3.0 才有能力存多个向量"——**结构数组（StructArray）与相关检索在 2.6 线已经出现**。v2.6.18 写着：

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

> > MAX_SIM* 是embeddingList搜索的时候使用的一类指标，它负责比较查询向量和实体向量列表，搜索结果依然按照父实体返回，*号表示一类指标，比如MAX_SIM_COSINE、MAX_SIM_IP 和 MAX_SIM_L2。


`MAX_SIM*` 里的星号，是官方用来表示"这一类指标"的写法，具体包含哪些指标、分数怎么算，本篇不展开公式。这里只记两条结论：**查询是向量列表；返回的是父实体，不是其中某个元素。** 它对应的典型用法是 late interaction——ColBERT 给每个 token、ColPali 给每个 patch 各生成一个向量，匹配的细粒度发生在列表内部，但结果仍然落到文档或页面这一级。

**其二，按元素出结果（element-level 搜索）。** StructArray 里的每个 Struct 元素各自作为独立候选，命中就返回那个元素，未分组时还能给出它在数组中的下标。官方定义：

> Element-level search treats each Struct element as an independent vector-search candidate. Each hit represents a matched element inside the StructArray field, and ungrouped results can expose the element offset.

此时查询是**普通向量**，指标也是普通向量指标（`COSINE` / `IP` / `L2`），返回的是"最相关的那一段、那一页、那个 patch"。

所以两种方式不是新旧替代，而是**结果粒度不同**。选哪个，取决于你要的结果长什么样：**目标是实体级结果，就用 EmbeddingList 搜索；要返回匹配的 Struct 元素（带 offset），就用 element-level 搜索。**

> ⚠️ **注意**：一条向量字段或向量子字段**只能建一个索引**，而两种搜索用的指标族不同。官方因此明确要求：同一集合既要实体级又要元素级，就**拆成两个向量子字段**。原文：If your collection needs both EmbeddingList search and element-level search, use two separate vector subfields. A vector field or vector subfield accepts only one index, and the two search modes require different metric families.


## 7.0.4 变化全景：按"要不要动手"分层

上一节讲完三条主线，这一节回答"要不要动手"。3.0 的能力**并不是一升级就全部生效**：官方在 v3.0.0 的 *Compatibility and behavior notes* 里只明确写出了四条默认行为，其余都要你改配置或改代码。

### 7.0.4.1 不用改代码就可能感知到的变化

官方在这一小节里写明的默认行为只有四条，逐条看原文与含义。

**一、存储引擎 V3 默认关闭。** 官方原文：

> **Storage V3 (Loon) is disabled by default.** Features that depend on it — such as Snapshot and TEXT fields — require enabling it manually via `common.storage.useLoonFFI`. Storage V3 will be enabled by default in a later release.

含义是：装好 3.0 之后，**快照和 TEXT 长文本字段这些功能实际上并不存在**，必须手动打开 `common.storage.useLoonFFI` 才会出现。

**二、2.6 → 3.0 保证兼容，也能回滚——但有个一次性的开关。** 官方原文：

> **2.6 → 3.0 compatibility and rollback are guaranteed** — a 3.0 deployment can be rolled back to 2.6. However, once you enable or use features that change the serialized data format (for example Storage V3), rollback is no longer possible.

也就是说升级本身留了后路，可**一旦启用了改变序列化数据格式的特性（例如 Storage V3），回滚窗口就永久关闭**。这一条必须写进你的升级变更方案，而不是事后才发现。

**三、新引入的索引算法默认不生效。** 官方原文：

> **New index versions are opt-in for now.** Newly introduced index algorithms require manually raising the target index version (`dataCoord.targetVecIndexVersion` to 10, `dataCoord.targetScalarIndexVersion` to 4) before they take effect; a later release will enable them by default.

含义是：光升版本拿不到新稀疏 / 新标量索引，必须把 `dataCoord.targetVecIndexVersion` 抬到 **10**、`dataCoord.targetScalarIndexVersion` 抬到 **4**。

**四、GPU 镜像换到 CUDA 12.9。** 官方原文只有一句：**GPU images move to CUDA 12.9** and no longer preserve Ubuntu 20.04 GPU compatibility。沿用旧 GPU 环境的读者会直接撞上不兼容。

> ⚠️ **注意**：除这四条之外，官方**没有**写过"某项新能力已默认开启"。因此任何"升级到 3.0 就自动更快 / 自动用上 Woodpecker / 自动启用 StructArray"的说法，都必须先在 release notes 或对应文档页找到原文，找不到就不要写。尤其注意：Woodpecker 作为默认 WAL 是 **2.6** 就完成的事，3.0 新增的是"可以作为独立服务部署"。

### 7.0.4.2 必须动手才有用的部分

把上面四条反过来看，就是七件要做的事，每件后面注明它解锁什么、在哪一节讲。

- **打开 `common.storage.useLoonFFI`**：解锁存储引擎 V3、快照、TEXT 长文本字段。见 **7.2.2**，官方页 https://milvus.io/docs/storage-v3.md 。
- **把 `dataCoord.targetVecIndexVersion` 抬到 10**：解锁换代后的稀疏向量索引（SINDI、Block-Max WAND / MaxScore）。见 **7.4.2**，官方页 https://milvus.io/docs/sparse-inverted-index.md 。
- **把 `dataCoord.targetScalarIndexVersion` 抬到 4**：解锁新标量索引路线（含位图类索引增强）。见 **7.4.5**，官方页 https://milvus.io/docs/bitmap.md 与 https://milvus.io/docs/index_selection.md 。
- **配置 MQ 切换与 Woodpecker 的部署形态**：把 WAL 作为独立服务运行，换取独立扩缩与故障隔离。见 **7.4.6**，官方页 https://milvus.io/docs/woodpecker.md 与 https://milvus.io/docs/switch-mq-type.md 。
- **把 pymilvus 升到 3.0.x**：拿到 TEXT 字段类型、函数链、搜索分面、compaction 生命周期观测这些客户端能力。见 **7.1.3.1**，官方页 https://milvus.io/docs/install-pymilvus.md 与 https://milvus.io/docs/using_milvusclient.md 。
- **准备对象存储与湖表的访问权限**：外部集合原地查询、快照的跨桶备份恢复都依赖它。见 **7.2.3** 与 **7.2.4**，官方页 https://milvus.io/docs/create-an-external-collection.md 与 https://milvus.io/docs/multi-storage-backup-and-restore.md 。
- **在集群侧登记文件资源（词典）**：自定义词典、同义词词典，用来改善中文检索。见 **7.3.7**，官方页 https://milvus.io/docs/manage-file-resources.md 与 https://milvus.io/docs/analyzer-overview.md 。

具体怎么改配置文件、怎么滚动重启、改完如何验证生效，全部在 **7.1.4**，本篇只回答"哪些开关值得动"。

### 7.0.4.3 新增能力归属清单

下面按阶段把 3.0 线的新增项列全，依据是官方 **3.0-beta 的 Key Features** 与 **v3.0.0 的 What's new since 3.0-beta** 两份清单，每项后面注明它影响的是运维、数据布局、代码还是索引参数，以及讲它的小节。

**3.0-beta 提出、GA 沿用或扩大的：**

- 存储引擎 V3（Loon）——运维，默认关，见 7.2.2。
- 外部集合原地查询——数据布局，见 7.2.3。
- 快照 Snapshot——运维，见 7.2.4。
- Entity TTL（由 TIMESTAMPTZ 驱动的行级过期）——代码，见 7.3.3。
- MinHash DIDO（写文档进库即得签名）——代码，见 7.3.4。
- 自定义词典与同义词词典（集群级 FileResource）——运维 + 代码，见 7.3.7。
- Query / Search 服务端 ORDER BY——代码，见 7.3.5。
- 查询聚合（COUNT / SUM / AVG / MIN / MAX 与 group-by）——代码，见 7.3.5。
- EmbList + DiskANN（把多向量检索放到磁盘上）——索引参数，见 7.4.4。
- 强制合并 Force Merge（同步 / 异步）——运维，见 7.4.7。
- 向量字段可为 NULL——代码，见 7.3.3。官方在 beta 清单里的表述是 **Nullable vectors — NULL on all six vector types; search skips NULL rows**；3.0 的说明页补了三句关键限定：检索自动跳过 NULL 向量、检索质量不受影响、NULL 向量几乎不占存储，并且 `AddField` 在 `nullable=True` 时能给已有集合在线加向量字段。**这是 3.0 的能力，不要写成 2.6 已有。**

**v3.0.0 才补齐的：**

- 外部集合的 Iceberg 支持、可改 Schema 与完整湖仓工作流——数据布局，见 7.2.3。
- Vortex 与 Lance 两种开放列式格式支持——数据布局，见 7.2.3。
- 在线 Schema 演进全流程（加列 / 回填 / 删列）——代码，见 7.2.5。加列本身 v2.6.0 已有（Add Field），3.0 补的是回填与删列。
- StructArray 覆盖面补齐——代码 + 索引，见 7.3.2。官方原文：**StructArray now supports null values, bitmap indexes, dynamic field addition on live collections, and partial update of struct fields through upsert, with REST and bulk-import coverage to match.**
- 搜索侧分面返回——代码，见 7.3.5。
- Function Chain 重排（含原生 XGBoost 打分）——代码，见 7.3.6。
- 稀疏索引全面换代——索引参数，见 7.4.2。官方原文：**It introduces new search algorithms — SINDI, Block-Max WAND, and Block-Max MaxScore — along with inverted-list compression, configurable quantization, and per-workload search-algorithm selection.**
- `index_type=FAISS` 直通任意 index-factory 串——索引参数，见 7.4.3。官方原文：**A new `FAISS` index type accepts arbitrary Faiss index-factory strings through the `faiss_index_name` parameter — `IVF64,Flat`, `HNSW16,Flat`, `OPQ16,IVF64,PQ16x4` — with search parameters passed through.**
- Woodpecker 可作为独立服务部署——运维，见 7.4.6。
- compaction 生命周期可观测——运维，见 7.4.7。

> ⛳ **架构演进提示**：EmbList + DiskANN 是官方所称 StructList 家族的**第一个落地变体**，家族其余部分（StructList 过滤、Muvera / Lemur 多向量加速）官方定位为后续目标，撰写时不要写成"已经可用"，需按读者实际使用的版本核对。
