# Chapter 7.3 Milvus 3.0 数据模型与检索增强

## 7.3.1 学习目标

本章解决的是 Milvus 3.0 在“数据模型”与“检索表达”两个方向上的升级：让一行实体装得下更复杂的数据（数组结构体、长文本、带时区与行级过期的字段），也让本来要搬回客户端算的重活（排序、聚合、分面、重排）能在服务端一次请求里完成。它不解决向量、索引、Schema 的基础概念——那些在 Ch2、Ch3 已经讲过，这里默认你已会写单向量检索、建过索引、跑过基础过滤。

读完本章，你应当能：

- **用 StructArray 把“一文档多切片”装进同一行**：区分实体级（EmbeddingList）与元素级（element-level）两种检索粒度，写出定义、写入、建索引、检索的完整链路（7.3.2）。
- **判断该给字段用可空向量还是默认值**：说清 NULL 与 default 的语义边界，以及行级 TTL 和集合级 TTL 各自管什么（7.3.3）。
- **用 TEXT 字段承载长文本，并用四种匹配算子做长文/短语/模式/集合成员过滤**，理解 MinHash DIDO 的文档查重闭环（7.3.4）。
- **用一次请求拿到“排序 + 聚合 + 分面”结果**：把 2.6 里“多取一批回客户端再 sort/统计”的旧写法，换成服务端 ORDER BY、COUNT/SUM/AVG 与分面返回（7.3.5）。
- **把重排从客户端挪到服务端**：理解 Function Chain 与客户端 Reranker 的分工，能写出一条可组合的声明式重排链（7.3.6）。
- **给中文语料配一套可复用的分析器与词典资源**：拼音、同义词、数字规范化，让中文商品名、人名也能被准确召回（7.3.7）。

如果你只关心其中某一项，可以直接跳到对应小节；但 7.3.2 的两种检索粒度是后文 StructArray 检索与分组返回的公共前提，建议不要跳过。

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

这一节回答 StructArray 到底解决哪一类建模难题；它不展开 Schema 定义与检索语法——那些留到 7.3.2.2 到 7.3.2.6 逐段给代码。

在 StructArray 之前，要把“一个文档拆成多个切片、每个切片带自己的向量和元数据”这种嵌套关系塞进 Milvus，只有三条绕路：

- **每个切片单独存一行**：文档的每个 chunk 都变成一条独立实体，靠一个 `doc_id` 标量串起来。回文档时要自己按 `doc_id` 聚合去重，一次写入被拆成 N 次，查询要额外 group，父子关系完全落在客户端。
- **建两张集合**：文档放主表、切片放子表，靠外键关联。关系表达出来了，但向量检索和过滤跨了两张表，没法在同一个集合里同时按“切片相似度”和“文档元数据”过滤，一致性也要自己维护。
- **JSON 字段塞一个 blob**：把整篇文档的切片序列化成一个 JSON 字段。子元素没有 schema 约束、没法对嵌套字段建向量索引，查询只能取回后自己解析——Ch3 讲过的“弱结构扩展”指的就是这条路。

3.0 的 StructArray（`ARRAY<STRUCT>`）把这三种凑合写法合并成一个字段：每个 Struct 元素都能带标量子字段和向量子字段，父实体和子元素在同一条记录里**一次写入**；查询时既能把子元素当独立候选返回（element-level，带 offset），也能把整条实体当候选返回（EmbeddingList），父子关系不再需要外键或客户端拼装。

一句话记住它的定位：StructArray 替代的是“JSON blob 的无约束嵌套”和“双表/一表多行的父子分离”，而不是简单的“一行多值”。这条边界在 7.3.2.6 的限制清单里会反过来约束你怎么设计。

### 7.3.2.2 定义与写入

这一节给出一条从零到能查的 StructArray 链路：建集合（含 struct 子字段）→ 插入嵌套数据 → 打印返回结构确认落库。它不展开索引与检索写法——那些在 7.3.2.3、7.3.2.4 展开，这里只把数据模型立起来。

先定一份贯穿后文示例的语料：`tech_articles` 集合，每行是一篇文章，`chunks` 字段是 StructArray，存文章切片。每个切片元素带 5 个标量子字段（`text` / `section` / `page` / `quality_score` / `has_code`）和 2 个向量子字段（`emb_list_vector` 给 EmbeddingList 检索、`emb` 给元素级检索）。同一份向量在两个子字段各写一遍，是因为两种检索需要的指标族不同、而一个向量子字段只能建一个索引——这条规则在 7.3.2 开头已经讲清。

**第一步：定义 Schema 并建集合。** StructArray 不是独立字段类型，它是 `element_type=STRUCT` 的 Array 字段，元素自己的 schema 用 `create_struct_field_schema()` 单独建：

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

# 集合级字段：主键 + 文章级字段
schema.add_field(field_name="doc_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="title_vector", datatype=DataType.FLOAT_VECTOR, dim=4)

# 每个 Struct 元素遵循的 schema（嵌套层级：集合 -> chunks 数组 -> 单个元素）
chunk_schema = client.create_struct_field_schema()
chunk_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
chunk_schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=128)
chunk_schema.add_field(field_name="page", datatype=DataType.INT64)
chunk_schema.add_field(field_name="quality_score", datatype=DataType.FLOAT)
chunk_schema.add_field(field_name="has_code", datatype=DataType.BOOL)
chunk_schema.add_field(field_name="emb_list_vector", datatype=DataType.FLOAT_VECTOR, dim=4)  # EmbeddingList 检索
chunk_schema.add_field(field_name="emb", datatype=DataType.FLOAT_VECTOR, dim=4)              # 元素级检索

# 把 Struct schema 挂到 Array 字段上
schema.add_field(
    field_name="chunks",
    datatype=DataType.ARRAY,
    element_type=DataType.STRUCT,
    struct_schema=chunk_schema,
    max_capacity=1000,   # 必填：限制单行最多存多少个元素
)

client.create_collection(collection_name="tech_articles", schema=schema)
```

三个要点：`max_capacity` 必填，限制每个实体最多存多少个 Struct 元素；子字段类型只能用标量（BOOL / 各宽整数 / FLOAT / DOUBLE / VARCHAR）和普通向量（FLOAT_VECTOR、FLOAT16_VECTOR、BFLOAT16_VECTOR、INT8_VECTOR、BINARY_VECTOR）；稀疏向量、JSON、Geometry、TEXT、TIMESTAMPTZ 以及再嵌套一层 Array/Struct 都不支持，具体清单见 7.3.2.6。

**第二步：插入嵌套数据。** 写入时 `chunks` 是「对象数组」——每个对象是一个元素、键名就是子字段名，**不要**用 `chunks[text]` 这种路径语法：

```python
data = [
    {
        "doc_id": 1,
        "title": "StructArray indexing patterns",
        "category": "index",
        "title_vector": [0.12, 0.08, 0.32, 0.48],
        "chunks": [
            {"text": "Create one index for each vector subfield.",
             "section": "index", "page": 1, "quality_score": 0.96,
             "has_code": False,
             "emb_list_vector": [0.10, 0.20, 0.30, 0.40],
             "emb": [0.10, 0.20, 0.30, 0.40]},
            {"text": "Use MAX_SIM metrics for EmbeddingList search.",
             "section": "index", "page": 2, "quality_score": 0.91,
             "has_code": True,
             "emb_list_vector": [0.16, 0.24, 0.35, 0.45],
             "emb": [0.16, 0.24, 0.35, 0.45]},
        ],
    },
    # 第二、三行结构相同，这里省略
]

result = client.insert(collection_name="tech_articles", data=data)
print(result)
```

预期输出类似 `{'insert_count': 3, 'ids': [1, 2, 3], 'cost': 0}`——插入条数与主键 id 列表（`auto_id=False` 手动指定主键）。注意 `chunks` 里向量维度要与 `dim=4` 一致、元素个数不能超过 `max_capacity`。

**第三步：确认落库结构。** 用 `query` 把嵌套子字段按路径投影出来；路径语法只在查询/检索/过滤/建索引时用，插入时不用：

```python
rows = client.query(
    collection_name="tech_articles",
    filter="doc_id in [1, 2, 3]",
    output_fields=["doc_id", "title", "chunks[text]", "chunks[section]", "chunks[quality_score]"],
)
for row in rows:
    print(row)
```

预期输出：每行一个 dict，`chunks[text]`、`chunks[section]` 等键对应的是该文档所有切片在这些子字段上的值列表——父子关系在这一次查询里就带出来了，不用客户端按 `doc_id` 聚合。


### 7.3.2.3 给子字段建索引

这一节解决「StructArray 的子字段怎么建索引、哪些值得建」。它不展开 HNSW/IVF 等索引选型原理——那是 Ch2（2.3.3）的内容，这里只讲子字段与普通字段在写法上的差异。

和普通字段不同，StructArray 的索引目标是**子字段路径**而不是字段名：写 `chunks[emb]`，不写 `emb`，也不写 `chunks.emb`。选指标族看检索模式——`emb_list_vector` 走 EmbeddingList 检索必须配 `MAX_SIM*` 指标族，`emb` 走元素级检索配普通指标（`COSINE` / `IP` / `L2`）：

```python
index_params = client.prepare_index_params()

# EmbeddingList 检索：MAX_SIM* 指标族
index_params.add_index(
    field_name="chunks[emb_list_vector]",
    index_name="chunks_emb_list_max_sim",
    index_type="HNSW",
    metric_type="MAX_SIM_COSINE",
    params={"M": 16, "efConstruction": 200},
)

# 元素级检索：普通向量指标
index_params.add_index(
    field_name="chunks[emb]",
    index_name="chunks_emb_cosine",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
)

# 标量子字段：只给频繁出现在过滤条件里的建
index_params.add_index(field_name="chunks[section]", index_name="chunks_section_inverted", index_type="INVERTED")
index_params.add_index(field_name="chunks[has_code]", index_name="chunks_has_code_inverted", index_type="INVERTED")
index_params.add_index(field_name="chunks[quality_score]", index_name="chunks_quality_score_sort", index_type="STL_SORT")

client.create_index(collection_name="tech_articles", index_params=index_params)
```

建完用 `client.list_indexes(collection_name="tech_articles")` 打印，确认子字段路径都建上了。

哪些子字段值得建索引，一句话判断：**向量子字段按你要跑的检索模式各建一个（两套都要就两个子字段分开建）；标量子字段只在它真的频繁出现在过滤条件里时才建**。例如后文示例会用 `element_filter(chunks, $[quality_score] > 0.9)`、`MATCH_ANY(chunks, $[section] == "index")` 这类过滤，那 `quality_score`、`section` 就值得建标量索引；反过来只用于输出、从不进过滤器的 `text` 就不必建。

> ⚠️ **注意**：同一个向量子字段**只能建一个索引**，`MAX_SIM*` 指标与普通向量指标不能同时建在 `chunks[emb]` 一个字段上。两套检索模式都要，就拆成两个向量子字段（本例的 `emb_list_vector` 与 `emb`）分别建索引。官方原文：Do not create a `MAX_SIM*` index and a regular vector-metric index on the same vector subfield.


### 7.3.2.4 四种检索姿势

这一节把 StructArray 的四种检索姿势各给一个最小可跑例子，统一复用 7.3.2.2 建好的 `tech_articles` 语料，不再换数据。它不重讲建集合、建索引，默认前两节已经跑通；四种姿势对应的完整边界在 7.3.2.6 的限制清单里。

#### 1. 按元素命中返回（element-level）

查询是普通向量，`anns_field` 指向 `chunks[emb]`，命中返回切片元素，`offset` 给出它在数组里的下标，同一文档命中多个切片会返回多行。要实体级结果则改用 `chunks[emb_list_vector]` + EmbeddingList 查询（7.3.2 开头已讲）。

```python
query_vector = [0.19, 0.24, 0.30, 0.37]

results = client.search(
    collection_name="tech_articles",
    data=[query_vector],
    anns_field="chunks[emb]",
    limit=5,
    output_fields=["doc_id", "title", "chunks[text]", "chunks[section]"],
)
for hits in results:
    for hit in hits:
        print(hit["id"], hit["distance"], hit.get("offset"), hit["entity"])
```

预期输出：每行一个命中的切片元素，`offset` 是它在 `chunks` 数组里的下标（从 0 起）；`limit` 限制的是元素命中数，不是去重后的父实体数。

#### 2. 带过滤的 struct 检索（element_filter + MATCH_*）

顶层谓词筛实体，`element_filter` 把元素级向量检索限制到满足条件的切片；只做「文档有没有合格切片」这种实体级判定，就改用 `MATCH_*`。

```python
filter_expr = (
    'category == "search" && '
    'element_filter(chunks, $[section] == "index" && $[quality_score] > 0.9)'
)
results = client.search(
    collection_name="tech_articles",
    data=[query_vector],
    anns_field="chunks[emb]",
    filter=filter_expr,
    limit=5,
    output_fields=["doc_id", "title", "chunks[text]", "chunks[section]"],
)
```

要点：`element_filter` 要放在整个表达式最后、一条过滤里只能出现一次；`$[subfield]` 只能用在 `element_filter`/`MATCH_*` 谓词内。实体级判定用 `MATCH_ANY(chunks, $[section] == "index")` 这类行级算子，不返回 offset。

#### 3. 按父文档分组去重（group_by_field）

element-level 检索下，同一文档多个切片命中会重复出现，用 `group_by_field` 按主键去重，每个父实体最多返回一次。

```python
results = client.search(
    collection_name="tech_articles",
    data=[query_vector],
    anns_field="chunks[emb]",
    limit=5,
    group_by_field="doc_id",
    output_fields=["doc_id", "title", "chunks[text]", "chunks[section]"],
)
```

预期输出：每个 `doc_id` 至多一行。分组只支持按主键、只作用于 element-level 检索（EmbeddingList 检索不支持 group-by），且不能与 range 参数同用。

#### 4. 多向量混合（hybrid_search + RRF）

把文章级向量 `title_vector` 与切片级向量 `chunks[emb]` 各发一个请求，用 RRF 融合排名：

```python
from pymilvus import AnnSearchRequest, RRFRanker

title_req = AnnSearchRequest(data=[query_vector], anns_field="title_vector", limit=10)
chunk_req = AnnSearchRequest(
    data=[query_vector], anns_field="chunks[emb]", limit=10,
    expr='element_filter(chunks, $[section] == "index")',
)
results = client.hybrid_search(
    collection_name="tech_articles",
    reqs=[title_req, chunk_req],
    ranker=RRFRanker(),
    limit=5,
    output_fields=["doc_id", "title", "chunks[text]", "chunks[section]"],
)
```

注意结果粒度：只有「同一 StructArray 下的多个 element-level 请求」混合时，最终结果才保持元素级；一旦混入文章级向量或 EmbeddingList 请求，最终候选回落到实体级、元素 offset 不再返回。

> ⚠️ **注意**：范围检索（`radius` / `range_filter`）只支持 element-level 向量字段，EmbeddingList 检索不支持 range search；range 与 group-by 不能同用。


### 7.3.2.5 操作符

StructArray 过滤有两族算子：`element_filter` 约束「哪些切片参与元素级检索」，`MATCH_*` 判断「实体是否合格」。逐个列示如下（谓词里的 `$[subfield]` 指当前切片元素的标量子字段，只能出现在这两族算子里）：

- `element_filter(chunks, $[section] == "index")` —— 匹配满足标量条件的切片元素；多条件 `&&` 时都作用于同一个元素，如 `element_filter(chunks, $[section] == "index" && $[quality_score] > 0.9)`。
- `MATCH_ANY(chunks, $[section] == "index")` —— 至少一个切片满足，返回实体；空数组返回 `false`。
- `MATCH_ALL(chunks, $[has_code] == true)` —— 所有切片都满足；空数组返回 `true`。
- `MATCH_LEAST(chunks, $[quality_score] > 0.9, threshold=2)` —— 至少 N 个切片满足，`threshold` 须为正整数。
- `MATCH_MOST(chunks, $[has_code] == true, threshold=1)` —— 至多 N 个切片满足，`threshold` 可为 0 或正整数。
- `MATCH_EXACT(chunks, $[section] == "filter", threshold=1)` —— 恰好 N 个切片满足，`threshold` 可为 0 或正整数。

> ⚠️ **注意**：`MATCH_ANY` 与 `element_filter` 都能表达「至少一个切片满足」，但前者只做行级过滤、后者做元素级约束——需要约束哪些切片进向量检索时用 `element_filter`。`element_filter` 与 `MATCH_*` 不能嵌套；`$[...]` 不接受 text_match/phrase_match、JSON 路径、`is null`、Geometry/Timestamptz 表达式，向量子字段也不是 `$[...]` 的标量输入。


### 7.3.2.6 限制清单

这一节把 StructArray 的限制一次性列全，按「schema 形状 → 子字段类型 → 路径与写入 → 索引/检索 → 过滤 → 可空与动态字段/分区键」分组。读完这张清单再设计 schema，比写完再返工省事得多。

**Schema 形状**

- Struct 只能做 Array 字段的元素类型（`element_type=STRUCT`），不能当顶层字段。
- 同一 StructArray 的所有元素共用一份 Struct schema，不能逐元素异构。
- `max_capacity` 必填，限制单个实体在 StructArray 里最多存多少个元素。
- 子字段结构建好后固定：不能加子字段、不能删单个子字段，要改只能删掉整个 StructArray 字段再按新 schema 重建。
- 不支持嵌套：子字段里不能再放 `Array` / `ArrayOfVector` / `Struct` / `ArrayOfStruct`。

**子字段类型**

- 支持：`BOOL`、`INT8/16/32/64`、`FLOAT`、`DOUBLE`、`VARCHAR`（须 `max_length`），以及 `FLOAT_VECTOR` / `FLOAT16_VECTOR` / `BFLOAT16_VECTOR` / `INT8_VECTOR` / `BINARY_VECTOR`（须 `dim`）。
- 不支持：稀疏向量、`String`（用 `VARCHAR` 替代）、`JSON`、Geometry/GIS、`TEXT`、Timestamptz 及时间表达式。
- 字段函数（function）不能作用在 StructArray 字段或其子字段上。

**路径与写入形状**

- 建索引/检索/输出/过滤用子字段路径 `chunks[emb]`，不写 `chunks.emb`。
- 插入写对象数组（`chunks: [{"text": "...", "emb": [...]}, ...]`），键名是子字段名，不用路径语法；每个元素必须对齐 Struct schema、向量维度对齐 `dim`、元素个数不超 `max_capacity`。

**索引与检索**

- 每个向量字段/向量子字段只能建一个索引；EmbeddingList 检索要 `MAX_SIM*` 指标族、元素级检索要普通指标（`L2` / `IP` / `COSINE` / `HAMMING` / `JACCARD`），两者必须拆成两个向量子字段。
- 元素级检索才返回 `offset`；EmbeddingList 检索是实体级结果，不返回 offset。
- range search 只支持元素级向量字段，EmbeddingList 检索不支持 range；range 与 group-by 不能同用。
- group-by 只认主键、只作用于元素级检索。

**过滤算子**

- `$[subfield]` 只能出现在 `element_filter` / `MATCH_*` 内；一条过滤里 `element_filter` 只能出现一次、不能嵌套。
- 向量子字段不能当 `$[...]` 的标量输入；JSON 路径/函数、数组函数、`text_match` / `phrase_match`、Geometry、Timestamptz 表达式都不支持。

**可空与动态字段/分区键**

- Milvus 3.0.0 起 StructArray 字段可空（`nullable=True`），但 null 作用于整个字段，不能对单个子字段独立设 nullable。
- 向已有集合动态加 StructArray 字段须 `nullable=True`，旧实体会对该字段返回 null。
- nullable 字段不能做分区键（分区键必须非空）。

> ⚠️ **注意**：先查限制再设计 schema——`max_capacity`、子字段类型白名单、「EmbeddingList 与元素级检索必须拆两个向量子字段」这三条一旦写定就很难改，宁可先读这张清单再动手。


7.0 说的第二条限制，属于这里：

**二是"改集合字段"与"改 StructArray 内部结构"不是一回事。** 集合级字段可以整体增删，但子字段不行。官方原文：You can drop a whole `ARRAY<STRUCT>` field, but you cannot drop an individual sub-field inside an `ARRAY<STRUCT>` field. 加子字段同样不行：After a StructArray field is created, you cannot add new subfields to that existing StructArray field. 需要调整子字段结构时，官方给的做法是先 `drop_collection_field()` 删掉整个 StructArray，再按新的 Struct schema 重新添加。

所以准确的说法是：**3.0.0 让集合级字段的变更可以在服务运行中完成；它并没有改变"StructArray 的内部结构只能整体重建"这条约束。**


## 7.3.3 字段属性：可空 · 默认值 · 时区 · 行级 TTL

> 本节只改字段属性，不改列结构；加列 / 删列 / 回填见 7.2.5。

### 7.3.3.1 可空向量与默认值

这一节只改字段属性（可空、默认值），不改列结构；加列 / 删列 / 回填见 7.2.5。

什么时候需要可空字段？三类典型场景：**部分行暂时没有向量**（嵌入模型还没跑完，先占位）、**嵌入模型上线前的空槽**（向量后补）、**稀疏向量可选**（有的行有、有的行没有）。它的核心语义只有一条：`nullable=True` 时，缺省该字段和显式写 `None` 等价，都存成 NULL；`nullable=False`（默认）时两者都直接报错。

**定义 + 插入 + 检索的最小闭环：**

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
# 向量字段可空：占位行暂不填向量
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=4, nullable=True)

client.create_collection(collection_name="docs_nullable", schema=schema)

# 一行正常、一行向量为 None（等价于缺省该字段）
data = [
    {"id": 1, "title": "ready",   "embedding": [0.1, 0.2, 0.3, 0.4]},
    {"id": 2, "title": "pending", "embedding": None},
]
client.insert(collection_name="docs_nullable", data=data)

# 检索：NULL 向量自动跳过，只对非空向量排名
res = client.search(
    collection_name="docs_nullable",
    data=[[0.1, 0.2, 0.3, 0.4]],
    anns_field="embedding",
    limit=3,
    search_params={"metric_type": "COSINE"},
    output_fields=["title", "embedding"],
)
print(res)
```

预期输出：只返回 `id=1` 一行；`id=2` 因向量为 NULL 被自动跳过、不报错。当有效向量数小于 `limit` 时，返回条数也会比 `limit` 少。

**nullable 与默认值的交互**（决定 NULL/缺省输入最终存成什么）：

| nullable | default_value | 输入 NULL 或缺省 | 实际存的值 |
| --- | --- | --- | --- |
| 是 | 有（非 NULL） | NULL / 缺省 | 用默认值 |
| 是 | 无 | NULL / 缺省 | 存 NULL |
| 否 | 有（非 NULL） | NULL / 缺省 | 用默认值 |
| 否 | 无 | NULL / 缺省 | 报错 |
| 否 | 有（NULL 默认） | NULL / 缺省 | 报错 |

三条口诀：**有非 NULL 默认值就永远用默认值，跟 nullable 无关**；**nullable=True 且无默认值才存 NULL**；**非可空字段设 NULL 默认值本身就是非法 schema**。`default_value` 的完整设置 API（含向量字段给零向量等写法）见官方 default-values 页，本节只讲它和 nullable 的交互语义。

两个容易踩的坑，单独拎出来：

- **可空向量字段不支持 `IS NULL` / `IS NOT NULL` 过滤**——你没法靠「向量是否为 NULL」来筛实体，NULL 向量只是在检索时被静默跳过。
- **nullable 字段不能做分区键**——分区键字段必须始终持有有效非空值。


### 7.3.3.2 TIMESTAMPTZ 字段

这一节回答「为什么需要一个带时区的专用时间类型，而不是拿 Int64 存 epoch」。它不重讲标量字段与过滤基础（Ch3），只讲 `TIMESTAMPTZ` 相对 Int64 的增量。

**和 Int64 存时间戳的区别，一句话说清：** Int64 存 epoch 秒/毫秒只是「一个数」，时区换算、跨时区比较、按绝对时间过期都得客户端自己维护，稍不留神就错位；`TIMESTAMPTZ` 是 schema 里的专用类型（`DataType.TIMESTAMPTZ`），输入 ISO 8601 带时区偏移的字符串，Milvus 内部统一归一化成 **UTC 绝对时间**存储，过滤、排序、比较全在 UTC 上做，跨时区结果一致。

**建字段 + 过滤查询：**

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("tsz", DataType.TIMESTAMPTZ, nullable=True)  # 时区感知时间
schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)

client.create_collection("timestamptz_demo", schema=schema)

# 插入：ISO 8601 带时区偏移（第一条用上海时区，第二条用 UTC，第三条 NULL）
client.insert("timestamptz_demo", [
    {"id": 1, "tsz": "2025-01-03T00:00:00+08:00", "vec": [0.1, 0.2, 0.3, 0.4]},
    {"id": 2, "tsz": "2025-01-03T00:00:00Z",    "vec": [0.2, 0.3, 0.4, 0.5]},
    {"id": 3, "tsz": None,                       "vec": [0.3, 0.4, 0.5, 0.6]},
])

# 过滤：时间字面量写 ISO '...'，按它自带的 offset 解释成绝对时刻再与字段（已存 UTC）比较
rows = client.query(
    collection_name="timestamptz_demo",
    filter="tsz < ISO '2025-01-03T00:00:00Z'",
    output_fields=["id", "tsz"],
)
print(rows)
```

预期输出：`id=1` 被选中——上海 `2025-01-03T00:00:00+08:00` 等于 UTC `2025-01-02T16:00:00Z`，确实早于 `2025-01-03T00:00:00Z`；`id=2` 恰好相等、不满足 `<`；`id=3` 是 NULL，任何比较都为 false（NULL 恒不满足过滤条件）。

**时间字面量与解释规则：**

- 字面量写 `ISO '2025-01-03T00:00:00+08:00'`，按它自带的 offset 解释成绝对时刻再比较；比较/排序始终在 UTC 上做。
- 支持 `==` / `!=` / `<` / `<=` / `>` / `>=`，以及 `INTERVAL` 运算：`tsz + INTERVAL 'P1D'`（加 1 天）、`tsz - INTERVAL 'PT2H'`（减 2 小时）、`P2DT6H`（2 天 6 小时）。
- 会话时区只影响展示：数据库 / 集合 / 单次查询三级 `timezone` 参数（优先级从低到高）决定把内部 UTC 值格式化成哪个时区字符串，不改变内部存储的 UTC 绝对值。
- 默认无索引时对 `TIMESTAMPTZ` 过滤是全表扫描，给字段建 `STL_SORT` 索引（`index_type="STL_SORT"`）可加速。

> ⚠️ **注意**：官方 timestamptz-field.md 前署标注 `beta: Milvus 2.6.6+`，本文按官方口径写，不称其为「3.0 新类型」；也勿把它当成「带时区的字符串」字段——它是一个归一化到 UTC 的类型，字符串只是输入/输出形态。


### 7.3.3.3 行级 TTL

Ch3（3.10）已讲集合级 TTL，这一节只补它没有的东西：**行级 TTL（entity-level TTL）**，以及两者最容易写混的差异。不重讲集合级用法。

两种模式先摆在一张表里对照：

| | 集合级 TTL | 行级 TTL |
| --- | --- | --- |
| 配置位置 | `properties={"collection.ttl.seconds": N}` | 加 TIMESTAMPTZ 字段 + `properties={"ttl_field": "expire_at"}` |
| 作用范围 | 全表同一保留窗口 | 每行自己的绝对过期时间 |
| 依赖字段 | 不需要额外字段，用系统插入时间 | 必须有一个 `TIMESTAMPTZ` 字段 |
| 过期时刻 | 相对（插入后 N 秒） | 绝对（墙钟时间） |
| 永不过期 | 做不到（全表统一） | 字段值为 NULL 即永不过期 |
| 清理粒度 | 过期先不可见，物理删除在下轮 compaction | 同左 |

**三条关键差异：**

1. **相对窗口 vs 绝对时刻。** 集合级是「插入后保留 N 秒」，行级是「每条自己带一个过期墙钟时间」——后者天然支持多租户不同保留期、按业务时间过期、以及「有的行永不删、有的行会删」的混搭。
2. **行级 TTL 依赖专用字段。** 通过 `ttl_field` 属性指定一个 `TIMESTAMPTZ` 字段；该字段必须是 `TIMESTAMPTZ`（其他类型会被拒绝），一个集合只能指定一个 `ttl_field`。
3. **两种模式互斥。** 一个集合不能同时设 `collection.ttl.seconds` 和 `ttl_field`，切换要先删旧属性再加新属性。

**建集合 + 三行数据的最小闭环：**

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
schema.add_field("expire_at", DataType.TIMESTAMPTZ, nullable=True)  # 行级 TTL 字段

client.create_collection(
    collection_name="events",
    schema=schema,
    properties={"ttl_field": "expire_at"},  # 指定哪个字段决定每行过期
)

# 三行：一行已过期、一行永不过期(NULL)、一行远期
client.insert("events", [
    {"id": 1, "vec": [0.1, 0.2, 0.3, 0.4], "expire_at": "2020-01-01T00:00:00Z"},
    {"id": 2, "vec": [0.2, 0.3, 0.4, 0.5], "expire_at": None},
    {"id": 3, "vec": [0.3, 0.4, 0.5, 0.6], "expire_at": "2099-01-01T00:00:00Z"},
])

res = client.search(
    collection_name="events",
    data=[[0.1, 0.2, 0.3, 0.4]],
    anns_field="vec",
    limit=5,
    search_params={"metric_type": "COSINE"},
    output_fields=["id", "expire_at"],
)
print(res)
```

预期输出：只返回 `id=2` 与 `id=3`；`id=1` 过期后立即从检索/查询结果里消失。给已存在集合补行级 TTL：先 `client.add_collection_field(..., data_type=DataType.TIMESTAMPTZ, nullable=True)` 加字段，再 `client.alter_collection_properties(properties={"ttl_field": "expire_at"})`；历史行没回填 `expire_at` 的为 NULL=永不过期，需要哪行过期就 `upsert` 那行补上过期时间。

> ⚠️ **注意**：过期 ≠ 合规删除。过期实体只是立刻从查询/检索结果里消失，物理删除要等下一轮 compaction（通常 24 小时内）才落盘，compaction 之前旧数据仍可能留在存储里。涉及敏感数据的合规（GDPR 类）不能只靠 TTL，需另行确认物理删除时点。


## 7.3.4 长文本字段 TEXT 与匹配算子

TEXT 这一项，7.0 把官方的定位讲得最透，原文搬来当引子，字段分界与算子在下面两节展开：

**三是 TEXT 长文本字段。** 官方这一项的小标题就叫 TEXT long-text fields，原话是：

> TEXT fields make long text first-class, with storage-side length limits removed: they support `text_match`, `phrase_match`, and BM25.

紧接着是存储细节：`Values under 64 KB stay inline; larger values go to partition-level LOB files in Vortex format, with the column storing only (file_id, offset) references.`，而且 LOB 文件跨段共享，压缩整理时搬的是引用、不是重写正文。对写教程的人来说，这项能力真正省事的地方在官方最后那句：`For RAG this means retrieving vectors and source text from the same store in one IO — no external blob store to operate.`——**向量与原文在同一个库里一次取回**，不必再自己维护一个对象存储专门放原文。它为什么算“数据模型扩展”：字段类型本身装得下长文本了，匹配算子也随之进到这一类字段上。字段分界与算子用法在 7.3.4 展开，这里不重复。

### 7.3.4.1 TEXT 与 VarChar 的分界

这一节讲清楚 TEXT 和 VarChar 的分界——什么时候用哪个。它不重讲 VarChar 基础（Ch2 标量字段、Ch3 3.3.4），只讲 3.0 新增 TEXT 类型带来的差异。

- **长度约束是分界线。** VarChar 必须设 `max_length`，上限 65,535 字节，超了要么截断要么报错；TEXT 不设 `max_length`，schema 里没有固定字节上限，长文直接塞进去。

- **定位不同。** VarChar 是「短、有界、拿来标识/分类/过滤」的元数据（`title` / `tag` / `category` / `external_id`）；TEXT 是「长、给 LLM/agent 当上下文」的原文（`content` / `passage` / `article_body` / `log_message`）。

- **存储两段式。** TEXT 按大小自动选存储路径：小于 `dataNode.text.inlineThreshold`（默认 **65,536 字节**）就内联存进字段数据；大于等于阈值就当大对象（LOB）写到对象存储（如 MinIO），字段里只存一个内部引用，查询/检索返回时按引用取回原文。这个选择对你是透明的——插入、查询、检索的写法完全一样。对 RAG 长文场景的意义在于：compaction 只搬引用不搬全文，避免大文本反复搬运。

  ![TEXT 大文本存储路径：小于内联阈值内联存储，大于等于阈值写大对象存储](/images/milvus30_text_large_storage_flow.png)

  > 图源：Milvus 官方文档（v3.0.x）

- **过滤能力差异。** VarChar 能进过滤表达式（`==` / `in` / `like` / regex 等）；TEXT 不支持一般标量过滤算子（`==` / `!=` / `>` / `<` / `IN` / `LIKE` / `=~` / `!~` / `IS NULL` 都不行），要按词过滤得给字段开 `enable_analyzer=True` + `enable_match=True`，用 `TEXT_MATCH` / `TEXT_MATCH_FUZZY`（见 7.3.4.2），要相关性排序就用 BM25（Ch4）。

- **TEXT 的硬限制，单独列出来：** 不能做主键 / 分区键 / 聚类键；不能当 Array 元素类型（含 StructArray 子字段，见 7.3.2.6）；Milvus 3.0.0 里不支持默认值、不支持外部集合；不能在 TEXT 上建用户标量索引（`enable_match=True` 时 Milvus 会建一个系统管理的文本索引，不是用户建的）；以 TEXT 为输入的 BM25 / MinHash 函数必须在建集合时就定义好，事后加不了。

- **前置开关。** TEXT 依赖 Storage V3，需把 `common.storage.useLoonFFI` 设为 `true`（默认 `false`），否则建含 TEXT 字段的集合会被拒。


### 7.3.4.2 四类匹配算子

这一节讲四类匹配算子的语义升级——基础过滤（`==` / `like` / `in`）Ch3（3.6.2）已经讲过，这里只写升级部分。四段统一复用同一份中英混合语料：字段 `text` 在建集合时开 `enable_analyzer=True` + `enable_match=True`（TEXT 或 VARCHAR 都行）。

**1. 关键词匹配（`TEXT_MATCH` / `TEXT_MATCH_FUZZY`）——精确分词命中，布尔过滤不评分：**

```python
filter_exact = "TEXT_MATCH(text, 'machine learning')"          # 精确分词命中
filter_fuzzy = "TEXT_MATCH_FUZZY(text, 'machne', max_edit_distance=1)"  # 容忍 1 个编辑距离
res = client.query(collection_name="docs", filter=filter_exact, output_fields=["text"])
```

`TEXT_MATCH` 找精确的分析词，`TEXT_MATCH_FUZZY` 容忍查询词与索引词之间的小编辑距离；两者都是布尔过滤，不给文档打分。要相关性排序走 BM25（Ch4）。

**2. 短语匹配（`PHRASE_MATCH` + `slop`）——顺序 + 邻近语义：**

```python
filter_0 = "PHRASE_MATCH(text, 'machine learning')"     # slop=0：紧邻且同序
filter_2 = "PHRASE_MATCH(text, 'machine learning', 2)"  # 允许最多 2 个词位的位移/倒序
res = client.query(collection_name="docs", filter=filter_2, output_fields=["text"])
```

`slop` 控制允许的词位偏移：0 表示必须紧邻同序；2 允许中间隔词或倒序（如 "learning machine" 也能命中）。

**3. 模式匹配（`LIKE` / `=~` / `!~`）——3.0 升级点在 regex：**

```python
filter_like = 'name LIKE "Prod%"'        # LIKE 通配符（Ch3 已讲）
filter_re   = 'message =~ "E[0-9]{4}"'   # 3.0 新增：RE2 正则，子串语义
filter_not  = 'message !~ "^DEBUG"'      # 排除满足正则的值
res = client.query(collection_name="docs", filter=filter_re, output_fields=["message"])
```

`=~` / `!~` 是 Milvus 3.0.x 新增，右侧必须是字符串字面量，走 RE2 语法，子串语义（不必整串匹配，要整串加 `^` `$` 锚点）；`LIKE` 只支持 `%` / `_` 通配符，不支持字符类、重复、锚点。

**4. 集合成员判定（`membership_match` + Bloom / Roaring 位图构建器）：**

```python
from pymilvus import MilvusClient, build_bloom_filter, build_roaring_bitmap

bloom_blob = build_bloom_filter(["creator-a", "creator-b"], fpr=0.005)   # 近似，省内存
roaring_blob = build_roaring_bitmap([1001, 1002, 1003])                  # 精确
res = client.query(collection_name="docs",
                   filter="membership_match(user_id, {members}, type=roaring)",
                   filter_params={"members": roaring_blob}, output_fields=["user_id"])
```

位图在客户端构建、以二进制 blob 传入（`filter_params` 占位符 `{members}`）。`type=bloom` 近似（可调 `fpr`，默认 0.005）、`type=roaring` 精确；**删除只能配 Roaring**——Bloom 的近似成员判断不是安全的删除谓词，解析期就会被拒。


### 7.3.4.3 MinHash DIDO：文档进·文档出查重

这一节给一个「原始语料 → 自动签名 → 近重复检索 → 去重后入库」的完整小闭环，典型场景是语料清洗、爬取内容去重。它不重述 Ch4 的 BM25 混合检索，只走 MinHash 这条新链路。

MinHash 的定位一句话：**DIDO**——输入是文档、输出是签名向量，整条转换由内核内置函数驱动，不用外部模型、不用预处理。它把文档切成 shingle、跑多路哈希取最小值，压成固定长度的二进制签名向量，用 `MHJACCARD` 指标近似 Jaccard 相似度。

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("document_content", DataType.VARCHAR, max_length=9000, enable_analyzer=True)  # 输入：原始语料
schema.add_field("binary_vector", DataType.BINARY_VECTOR, dim=8192)  # 输出：签名向量，dim = 32 * num_hashes

# 内核驱动：插入原文，签名向量自动生成，不用你算
minhash_function = Function(
    name="minhash_function",
    input_field_names=["document_content"],
    output_field_names=["binary_vector"],
    function_type=FunctionType.MINHASH,
    params={"num_hashes": 256, "shingle_size": 3},
)
schema.add_function(minhash_function)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="binary_vector", index_type="MINHASH_LSH", metric_type="MHJACCARD",
    params={"mh_lsh_band": 128, "mh_element_bit_width": 32, "with_raw_data": True},
)
client.create_collection(collection_name="dedup_collection", schema=schema, index_params=index_params)

# 写入三篇高度相似的语料
client.insert("dedup_collection", [
    {"document_content": "information retrieval is a field of study that helps users find relevant information in large datasets"},
    {"document_content": "information retrieval is a research field focused on helping users find relevant data in large collections"},
    {"document_content": "information retrieval is a field of research helping users search for relevant information in large datasets"},
])

# 近重复检索：查询原文直接当检索数据，签名在服务端生成
results = client.search(
    collection_name="dedup_collection",
    data=["information retrieval is a research field focused on helping users find relevant data in large collections"],
    anns_field="binary_vector",
    limit=3,
    search_params={"metric_type": "MHJACCARD", "params": {"mh_search_with_jaccard": True, "refine_k": 3}},
    output_fields=["document_content"],
)
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Jaccard distance: {hit['distance']:.3f}")
        print(f"  {hit['entity']['document_content']}")
```

预期输出：三篇都命中，`distance` 是 Jaccard 距离（越小越相似，`1 - distance` ≈ Jaccard 相似度）。去重后入库就是按阈值保留一条：命中里距离小于阈值（如 `distance < 0.2`）的视为近重复，只写一条。要点：`dim` 必须等于 `32 * num_hashes`（256 → 8192）；`MINHASH_LSH` 索引的 `mh_element_bit_width` 固定为 32。

![MinHash 工作流：分片 → 多路哈希取最小值 → 签名向量 → LSH 分桶找近重复候选](/images/milvus30_minhash_workflow.png)

> 图源：Milvus 官方文档（v3.0.x）


## 7.3.5 服务端排序 · 聚合 · 分面

### 7.3.5.1 检索结果按标量字段排序

这一节讲 3.0 把「结果排序」搬进了服务端。2.6 时代只能把 topK 结果全拿回客户端、自己 `sorted()` 一遍；3.0 用 `order_by_fields` 参数在请求里直接排序。它不重讲普通向量检索（Ch3 3.5.1），只写排序参数。

**多字段升降序 + 与 topK 的交互：**

```python
res = client.search(
    collection_name="product_catalog",
    data=query_vectors,
    anns_field="embedding",
    limit=20,
    output_fields=["id", "price", "rating", "category"],
    order_by_fields=[
        {"field": "price", "order": "asc"},    # 先按价格升序
        {"field": "rating", "order": "desc"},  # 价格相同再按评分降序
    ],
)
```

`order_by_fields` 是列表，每项一个 `{field, order}`；`order` 取 `"asc"` / `"desc"`，省略默认升序。要检查字段值记得把它加进 `output_fields`。

**排序发生在 topK 之内，不在之外——这是最容易误解的点。** 默认结果按向量相似度排；加了 `order_by_fields` 之后，Milvus 依然是**先用 ANN 按向量相似度召回 `limit` 个候选，再对这 `limit` 个候选按标量字段重排**。它不改变候选集合、不改变召回范围：标量最小/最大的那条如果没进 topK 候选，排序后也不会出现。（此语义为官方文档措辞的推论，官方页未逐字写明，**待验证**；要「按标量字段全局 Top-K」应走 7.3.5.2 的查询聚合路径。）要「按标量字段全局 Top-K」，不是这条参数的事，得走查询/聚合路径（见 7.3.5.2）。


下面几段从 7.0 原样迁来，从排序说起：

分面之外，另外四项各自补的是另一个缺口。

**服务端 ORDER BY** 的官方原话是 ：

`Search and Query now accept multi-field ordering, with the sort pushed down into the Milvus kernel and ASC / DESC settable per field.`，

它同时点出了要补的坑：`Top-K by distance alone often does not match the business need when the most similar item is not the cheapest, the most recent, or the most popular.`。

说人话就是：和你查询最像的东西不一定是最便宜、最新或最热的；以前“按距离取 top 再按价格重排”只能多取一批回然后在本地计算，现在把排序条件直接写进请求。


### 7.3.5.2 查询侧排序与聚合

这一节讲服务端聚合：把 COUNT / SUM / AVG / MIN / MAX 从客户端搬进内核。Ch3（3.6.4）的统计查询是「拉回全部匹配实体、客户端自己分组求和」的旧写法，3.0 的 Query Aggregation 让一次 `query` 在服务端算完。不重讲查询基础。

**旧写法 → 新写法，两步对照：**

- **旧写法（Ch3 3.6.4）**：`query` 拉回所有匹配实体 → 客户端 Python 里 `defaultdict` 分组、自己 `sum` / `avg`。数据一大就是过取 + 整批网络搬运 + 客户端 CPU，还容易把计数口径算错。
- **新写法**：`query` 带 `group_by_fields`，聚合表达式写进 `output_fields`，服务端在过滤之后直接算好返回。

```python
res = client.query(
    collection_name="my_collection",
    filter="",                          # 统计口径 = 过滤后命中的集合，不是全表
    group_by_fields=["color"],
    output_fields=["color", "count(*)", "avg(price)", "max(rating)"],
)
# 官方示例真实输出：
# [{'color': 'red',    'count(*)': 10, 'avg(price)': 65.22, 'max(rating)': 5},
#  {'color': 'orange', 'count(*)': 10, 'avg(price)': 48.67, 'max(rating)': 5},
#  {'color': 'yellow', 'count(*)': 10, 'avg(price)': 64.15, 'max(rating)': 3},
#  {'color': 'green',  'count(*)': 10, 'avg(price)': 58.28, 'max(rating)': 5},
#  {'color': 'blue',   'count(*)': 10, 'avg(price)': 50.20, 'max(rating)': 5}]
```

支持的聚合表达式：`count(*)`、`count(<field>)`、`min(<field>)`、`max(<field>)`、`sum(<field>)`、`avg(<field>)`，一次请求可混用多个。两条硬约束：`sum` / `avg` 只能作用在数值字段；`group_by_fields` 只支持 `INT8/16/32/64`、`VARCHAR`、`TIMESTAMPTZ`，用 `FLOAT` / `DOUBLE` / 向量 / `JSON` / `ARRAY` 分组会报错。

> ⚠️ **口径**：聚合在 `filter` 之后算——统计的是「命中集合」而非「全表」。要全表统计就留空 `filter`；要某个子集的分布就收紧 `filter`，两者得到的是完全不同的统计口径。


第二件是聚合，7.0 的原文：

**查询聚合** 官方给的是 `COUNT / SUM / AVG / MIN / MAX with group-by, evaluated server-side`，并强调 `Aggregation is evaluated server-side after filtering.（聚合是在过滤之后由Milvus服务端执行的）`。这里最关键的是“**过滤之后**”这个顺序——统计对象是符合条件的数据，不是整个集合。


### 7.3.5.3 分面检索

这一节给「商品搜索页左侧可点筛选栏」的最小例子：一次请求同时拿向量 topK + 各分面 Top 值 + 计数，不用把结果全拉回客户端自己数。它是 3.0 在**检索路径**新增的分面能力（Search Aggregation）；上一节的 Query Aggregation 是 query 路径的精确统计，这里是 search 路径的分面，按 ANN 召回候选计算。

```python
from pymilvus import MilvusClient, SearchAggregation, TopHits

aggregation = SearchAggregation(
    fields=["brand"],                              # 分面维度 = 左侧筛选栏的字段
    size=3,                                        # 返回前几个分面值
    metrics={"facet_count": {"count": "*"}, "avg_price": {"avg": "price"}},
    order=[{"facet_count": "desc"}],               # 分面桶按计数降序
    top_hits=TopHits(size=2, sort=[{"rating": "desc"}]),  # 每个分面带 2 个代表商品
)

result = client.search(
    collection_name="product_catalog",
    data=[query_vector],
    anns_field="embedding",
    limit=10,
    search_params={"metric_type": "COSINE", "params": {}},
    output_fields=["name", "brand", "price", "rating"],
    search_aggregation=aggregation,
)

for bucket in result.agg_buckets:
    print(bucket.key, bucket.count, bucket.metrics)
    for hit in bucket.hits:
        print("  ", hit.pk, hit.fields["name"])
```

预期输出：`agg_buckets` 里每个 `AggregationBucket` 带 `key`（分面值）、`count`（该分面命中候选数）、`metrics`（`facet_count` / `avg_price`）、`hits`（代表商品）。每个分面值由它在 ANN 排名里的最优匹配成员代表，附上 COUNT / AVG 等聚合注释——这正是「左侧筛选栏」需要的形态。开了 Search Aggregation 后，普通的扁平 hit 列表为空，结果从 `result.agg_buckets` 取。

![扁平向量检索结果被组织成可按品牌比较的分面桶](/images/milvus30_search_aggregation_overview.png)

> 图源：Milvus 官方文档（v3.0.x）

> ⚠️ **口径**：Search Aggregation 统计的是 ANN 召回的候选，**不是**全表精确聚合；桶的 count / metrics 随召回和每个键保留的候选数（`TopHits.size`）变化。要精确统计走 7.3.5.2 的 Query Aggregation。另注意它不能与 `offset`、Hybrid Search、Grouping Search 同用。


### 7.3.5.4 分组返回

这一节讲普通标量字段的 group-by 透传——按某个字段把命中结果分组、每组取前几个。它和 7.3.2.4 的 StructArray 分组要分清楚：那里 `group_by_field="doc_id"` 是「把同一父文档的多个切片元素折叠成一行」；这里是「普通字段分组透传，每组仍返回多个实体」，是给「每个类别都要几条」这类需求用的。

```python
res = client.search(
    collection_name="product_catalog",
    data=query_vectors,
    anns_field="embedding",
    limit=20,                    # 返回的组数
    group_by_field="category",   # 普通标量字段分组
    group_size=3,                # 每组最多返回 3 个实体
    strict_group_size=True,      # 不足 3 个也按 3 个返回
    output_fields=["category", "price", "rating"],
)
```

预期输出：结果按 `category` 分组，每组最多 `group_size` 个命中；`limit` 限制的是组数而非总条数。`strict_group_size=False`（默认）时不足 `group_size` 的组只返回实际命中的条数。可以和 `order_by_fields` 叠加（先分组、组内再按标量排序）。


## 7.3.6 服务端重排函数链

### 7.3.6.1 与客户端 Reranker 的分工

这一节回答「重排到底在哪做、服务端函数链和客户端 ranker 怎么选」。它不展开各 ranker 的 API 细节（7.3.6.2 给），只把两条路的边界划清。Ch6《Milvus Reranker 重排》讲的是**外部模型 / 客户端 ranker**，这里讲的是 3.0 的**服务端函数链**。

- **重排位置。** 客户端 ranker 在你的进程里跑（`pymilvus.model.reranker.CrossEncoderRerankFunction`），Milvus 只负责返回候选；服务端函数链是函数注册进 Milvus，检索后在内核内完成重排，候选不出 Milvus。

- **是否多一跳网络。** 客户端 ranker 要先从 Milvus 拉回全部候选、再发到外部模型服务打分，多一跳网络往返；服务端函数链在服务内完成，不额外回传候选。

- **延迟影响。** 客户端多一跳 + 回传全量候选，候选越多延迟越高；服务端只回传最终 topK，网络与序列化开销小。

- **可组合性。** 客户端每次手写「先融合、再提权、再衰减」的拼接逻辑；服务端函数链本身可组合——RRF 融合 → boost 提权 → decay 衰减可以串成一条链（7.3.6.2）。

- **适用场景。** 客户端 ranker 适合：模型只在客户端、旧版本无服务端函数、或打分逻辑高度自定义；服务端函数链适合：生产环境要低延迟、要可组合、要把重排固化进集合而非散落在各调用点。

> ⛳ **架构演进提示**：两条路都在、不是取代关系。3.0 的意义是「重排不再非得把数据拉回应用层」——纯函数组合（RRF / 加权 / 提权 / 衰减）交给服务端，外部大模型打分（Cohere / Voyage / vLLM 等）仍可挂在服务端函数里，但模型服务本身还得在集群外跑。


函数链这件事，7.0 只点到为止。原文连同官方讲解视频，一起搬到这里：

> 这里推荐 Zilliz 官方的讲解视频：
【商品推荐中，为什么不能只看相似度？Milvus FunctionChain检索、打分、重排精讲】 https://www.bilibili.com/video/BV16tJC6rENw/?share_source=copy_web&vd_source=2ee6373221bc82cfd994fdd5d96888f5

**Function Chain 重排** 的官方原话是 `Reranking is now composable through the Function Chain API, which executes an ordered, typed pipeline as part of a single search request（可以通过Function Chain API 将多个重排操作按照顺序组合成一个流程，并在一次搜索请求中执行）.`，这一版还加入了原生 XGBoost 打分用于 L0 重排。它替你省掉的是“检索调一次、重排再调一次、再自己把结果拼回来”这条客户端流水线：重排逻辑变成库里一段可声明、可组合的链。细节见 7.3.6。


### 7.3.6.2 声明式重排实操

这一节给三个递进的服务端重排函数例子：融合排名 → 提升/降权某表达式命中 → 外部模型打分融合。API 名与参数已对照官方页现值核对；第三个例子的前置条件是外部模型服务已在跑。

**1. 融合排名（RRF）——把多路向量检索按倒数排名融合：**

```python
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType

ranker = Function(
    name="rrf",
    input_field_names=[],          # RERANK 函数 input 必须为空列表
    function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 100},   # k 是平滑参数，可选
)

text_search = AnnSearchRequest(data=["modern dining table"], anns_field="text_vector", limit=10)
image_search = AnnSearchRequest(data=[image_embedding], anns_field="image_vector", limit=10)
hybrid_results = client.hybrid_search(
    collection_name, [text_search, image_search],
    ranker=ranker, limit=10,
    output_fields=["product_name", "price", "category"],
)
```

**2. 提升 / 降权某表达式命中（boost）——命中表达式的文档加权重：**

```python
ranker = Function(
    name="boost",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={"reranker": "boost", "filter": "doctype == 'abstract'", "weight": 0.5},
)

client.search(
    collection_name="my_collection",
    data=[query_vector], anns_field="vector",
    ranker=ranker, output_fields=["doctype"],
)
```

命中 `doctype == 'abstract'` 的文档得分被加权（`weight` 可正可负，正=提权、负=降权）。多个 boost 还能用 `FunctionScore` 组合成一条链（`boost_mode` / `function_mode` 控制组合方式）。

**3. 外部模型打分融合（model）——语义相关度重排：**

```python
# 前置条件：外部模型服务（如 TEI / vLLM）已在跑，endpoint 可达
model_ranker = Function(
    name="semantic_ranker",
    input_field_names=["document"],   # 用于重排的 VARCHAR 字段
    function_type=FunctionType.RERANK,
    params={
        "reranker": "model",
        "provider": "tei",            # tei / vllm 等
        "queries": ["machine learning for time series"],
        "endpoint": "http://model-service:8080",
    },
)

results = client.search(
    collection_name, data=[query_vector], anns_field="vector_field",
    limit=10, output_fields=["document"],
    ranker=model_ranker,
)
```

`data` 里的查询向量数必须与 `queries` 条数一致；模型 ranker 不能与分组检索同用，且重排字段必须为 `VARCHAR`（每次只能一个字段）。


## 7.3.7 中文检索体验增强

### 7.3.7.1 文件级词典资源

这一节回答「为什么词典文件要在集群级注册」，然后给上传 → 引用 → 生效三步。它不重讲分析器基础（Ch4 BM25 已讲），只讲词典这一侧的增量。

**为什么集群级注册，而不是每个集合内联数组：** 一是真词典很大——中文 jieba 词库几万行、同义词表上千条规则，内联进分析器配置不现实；二是同一份词典常被多个集合共用，注册一次、按名引用，schema 小而清晰，改词典也只是一次操作，不用逐个集合改配置。3.0 里四个分析器组件都能从文件资源加载词典：jieba 分词器（`extra_dict_file`）、停用词过滤器（`stop_words_file`）、解复合词过滤器（`word_list_file`）、同义词过滤器（`synonyms_file`）。

**三步：上传 → 注册 → 引用。**

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 第 1 步：把词典上传到对象存储（用 mc / aws s3 cp 等，这里省略）。
#          path 必须是含 rootPath 前缀的完整对象键，如 "file/chinese_terms.txt"。

# 第 2 步：注册（同步校验对象存在，幂等）
client.add_file_resource(name="chinese_terms", path="file/chinese_terms.txt")

# 第 3 步：在分析器里按名引用
jieba_extra_dict = {
    "type": "remote",
    "resource_name": "chinese_terms",   # 必须等于 add_file_resource 的 name
    "file_name": "chinese_terms.txt",   # 仅文件名
}
# 之后把 jieba_extra_dict 传给分析器的 extra_dict_file 参数即可

# 查看已注册资源（只返回 name + path，无 size/checksum 等元数据）
for r in client.list_file_resources():
    print(r.name, r.path)

# client.remove_file_resource(name="chinese_terms")  # 幂等；删前确认无集合引用
```

> ⚠️ **注意**：删除前先确认没有集合的分析器引用该资源（先 drop / alter 相关集合），否则删除后分析器查不到文件会报错。资源注册后同步到各组件有一小段窗口，引用它的集合若在这期间建可能失败，稍等重试即可；没生效先查路径与加载状态，别随口归因于「缓存」。


### 7.3.7.2 中文分析器与过滤链

这一节给一套可直接复用的中文 BM25 分析器配置（jieba 分词 + 拼音 + 同义词 + 数字规范化），并给「改前 / 改后」检索效果对照。Ch4 已讲 BM25 基本式，这里只写分析器 / 词典侧的增量。

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

analyzer_params = {
    "tokenizer": "jieba",
    "filter": [
        # 拼音：保留原词 + 全拼，让「用拼音搜中文」也能命中
        {"type": "pinyin", "keep_original": True, "keep_full_pinyin": True},
        # 同义词：expand=True 把同义组里的词互相展开
        {"type": "synonym", "synonyms": ["人工智能, AI"], "expand": True},
        # 数字规范化：把阿拉伯/泰文/梵文/全角等数字统一成标准数字
        "decimaldigit",
    ],
}

# 建集合时把 analyzer_params 挂到 enable_analyzer=True 的 VARCHAR/TEXT 字段上即可
```

**改前 / 改后检索效果对照（两行）：**

- 改前（只有 jieba 分词）：搜「AI」→ **0 命中**——原文只有「人工智能」，没有「AI」这个词项，`AI` 和 `人工智能` 是两个不相关的 token。
- 改后（+ 同义词 + 拼音）：搜「AI」→ 命中含「人工智能」的文档；搜「rengongzhineng」（全拼）→ 也能命中同一条。

> ⚠️ **注意**：`keep_full_pinyin=True` 会显著增大索引体积，拼音 + 同义词叠加会让 token 数膨胀，字段长、数据多时先小样本跑 `run_analyzer` 看 token 产出再全量入库。


## 7.3.8 本章小结

本章核心掌握：

- **StructArray（`ARRAY<STRUCT>`）**：把「文档 → 切片」的父子关系压进一个字段、一次写入；element-level 检索返回切片（带 `offset`）、EmbeddingList 检索返回实体，索引与检索都走 `chunks[emb]` 路径，EmbeddingList 与元素级检索必须拆两个向量子字段。
- **TEXT 与四类匹配算子**：TEXT 无长度上限、超阈值自动走大对象存储；关键词（`TEXT_MATCH` / `TEXT_MATCH_FUZZY`）、短语（`PHRASE_MATCH` + slop）、正则（`=~` / `!~`）、成员判定（`membership_match` + Bloom/Roaring）各管一段。
- **字段属性**：可空向量 / 默认值、`TIMESTAMPTZ`、行级 TTL 改的是字段属性、不改列结构；行级 TTL 是「绝对时刻」、与集合级 TTL 的「相对窗口」互斥。
- **服务端检索增强**：排序、聚合 / 分面、分组、重排函数链，把原来在客户端做的活搬进内核——排序发生在 topK 之内、聚合分 query 与 search 两条路径、重排函数链可组合。
- **中文检索增强**：词典集群级注册（多库共用、热更新、免重启）+ 中文分析器（jieba + 拼音 + 同义词 + 数字规范化）。

数据模型与检索增强到这里告一段落，7.4 看索引与写入引擎。

