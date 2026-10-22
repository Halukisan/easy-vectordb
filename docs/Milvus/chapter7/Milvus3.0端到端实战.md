# Chapter 7.5 Milvus 3.0 端到端实战：从湖上文件到可回滚的检索服务

## 7.5.1 学习目标

本章把 7.2–7.4 讲过的新能力拼成一条能从头跑到尾的链路：从湖上的一份 Parquet 文档表出发，挂成外部集合、在库里派生出检索字段、给切片数据建 StructArray、配齐四类索引、一次下推完成检索+排序+分面+重排，最后打快照做发布保护与回滚演练。本章不重复任何单点特性的语法——外部集合见 7.2.3、StructArray 见 7.3.2、稀疏索引见 7.4.2，这里只负责把它们组装起来，让你看到「3.0 与 2.6 的体验差」。

学完本章，你能：

1. 独立搭起一条 3.0 检索链路：从环境与开关自检（7.5.3.1）到外部集合挂载（7.5.3.2），环境准备自包含，不默认你读过前六章。
2. 用函数在库内派生检索字段：让湖上原文留在原地，BM25 稀疏向量与 MinHash 去重签名在库里生成（7.5.3.3）。
3. 一次下推完成检索 + 排序 + 分面 + 重排，并说清「过去要几次调用、现在几次下推」（7.5.3.6）。
4. 用快照做发布前保护与回滚演练，并记住「创建秒级、保留占存储」这两个代价（7.5.3.7）。

只想快速判断「3.0 值不值得上」的读者，读 7.5.2、7.5.5、7.5.6 三节即可；要动手跑的，从头跟 7.5.3 走到底。

## 7.5.2 场景与数据准备

本节给本章定一个贯穿到底的场景，并把要用的数据准备好；不解决「外部集合怎么挂、快照怎么打」——那些从 7.5.3 起逐节展开。

场景定成：一个团队把内部文档库（几十到几百篇文档）导出成一份 Parquet 表放进对象存储，希望**不把这些原文拷进 Milvus**，就能做三件事——按内容全文检索、按切片做向量检索、每次发布都留一个能回滚的快照。这正是 7.1.0 三条主线（数据放哪、一次请求算什么、写入与索引怎么落地）在同一个例子上的交汇。

数据分两层：

- **湖上原文表 `lake_docs.parquet`**：一篇文章一行，含 `doc_id` / `title` / `category` / `publish_ts` / `body` 五列，放对象存储（演示用本地目录模拟前缀）。这层只读、留在原地，由外部集合引用（7.5.3.2）。
- **库里切片语料**：把同一批文章按段落切成切片，用 StructArray 存进内部集合（7.5.3.4），做父子同集合的切片级检索。

目录结构沿用本项目惯例（数据放 `data/`、脚本放 `src/`）：

```text
data/milvus30_demo/
  lake_docs.parquet        # 湖上原文（对象存储里的源文件，演示用本地目录模拟）
src/
  make_lake_data.py        # 生成本地文档库 → 导出 Parquet
```

生成脚本（前置条件：`pip install pyarrow pandas`，Python 环境见 7.1.4）：

```python
# src/make_lake_data.py
import pandas as pd

docs = [
    {"doc_id": 1, "title": "Search Aggregation 入门", "category": "search",
     "publish_ts": 1720000000,
     "body": "Search Aggregation 把 ANN 候选按标量字段分桶，再算桶内统计。"},
    {"doc_id": 2, "title": "StructArray 建模", "category": "data-model",
     "publish_ts": 1720003600,
     "body": "一文档多切片时，用 StructArray 把切片存进同一行。"},
    # 其余文档按同一结构补全
]
df = pd.DataFrame(docs)
df.to_parquet("data/milvus30_demo/lake_docs.parquet", index=False)
print(df.dtypes)
```

预期输出：`data/milvus30_demo/lake_docs.parquet` 生成，`doc_id` 为 int64、`body` 为字符串列。这份 Parquet 就是后面外部集合要引用的「湖上原文」。

> 📌 **前置条件**：Parquet 的列名、类型在建外部集合时会被 `external_field` 逐列映射，所以这里的列名（doc_id / title / category / publish_ts / body）就是 7.5.3.2 里 `external_field` 的取值，类型也必须对得上；源表列结构变了怎么同步，见 7.2.3.7。

## 7.5.3 搭建步骤

### 7.5.3.1 环境与开关自检

本节是全章「跑不动」的第一道闸：确认服务端是 3.0.x、Loon 已开、索引版本开关已抬。它不重复 7.1.4 的开启步骤，只给一条版本/连通性检查脚本和「哪项没过」的判读。跑不动全章的读者，十有八九是这一节有一项没过。

先做版本与连通性自检：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
print("server_version =", client.get_server_version())  # 应为 v3.0.x（写作基线 v3.0.2）
print("collections =", client.list_collections())       # 连通性自检：空库应打印 []
```

预期输出：`server_version` 形如 `v3.0.2`，`collections` 为 `[]`（或你已有的集合）。版本停在 2.6.x 就去 7.1.4.2 原地升级。

再做两个开关自检（开启与复核步骤见 7.1.4.3 / 7.1.4.4，这里只判读）。打开 WebUI `http://127.0.0.1:9091/webui/` 查两项配置的当前值：

- `common.storage.useLoonFFI` 应为 `true`——它决定外部集合、快照能不能建。
- `dataCoord.targetVecIndexVersion` 应为 `10`、`dataCoord.targetScalarIndexVersion` 应为 `4`——它决定 7.5.3.5 的新稀疏算法到底走不走。

判读表（哪项没过，去读哪节）：

| 检查项 | 没过的表现 | 去读 |
| --- | --- | --- |
| 服务端版本 | `get_server_version()` 停在 2.6.x | 7.1.4.2 原地升级 |
| `useLoonFFI` | 外部集合/快照建不起来 | 7.1.4.3 打开 Loon |
| 两个索引版本 | 能建索引但不走新算法、**不报错** | 7.1.4.4 打开新版索引开关 |

> ⚠️ **注意**：后两项「不报错、只是没生效」是全章最高频的坑——7.4.2.2 会反复回指这里。三项全过，再往下走 7.5.3.2。

### 7.5.3.2 挂载外部集合

本节把 7.5.2 生成的湖上 Parquet 挂成一个外部集合，让原文留在原地、Milvus 只接管查询。它不重复外部集合语法——三个关键属性（原地读 / 只读 / 靠 refresh 追平）与支持的格式见 7.2.3.1、7.2.3.2，这里只贴最终配置并说清预期输出。

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# 外部源：S3 前缀；凭据换成你自己的（跨账号鉴权见 7.2.3.2）
schema = MilvusClient.create_schema(
    external_source="s3://<bucket>/milvus30_demo/",
    external_spec='{"format": "parquet", "extfs": {"access_key_id": "...", "access_key_value": "...", "region": "us-west-2", "cloud_provider": "aws", "use_ssl": "true", "use_virtual_host": "true"}}',
)
# 每个字段用 external_field 映射到湖上列名（与 7.5.2 的 Parquet 列一一对应）
schema.add_field(field_name="doc_id", datatype=DataType.INT64, is_primary=True, external_field="doc_id")
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512, external_field="title")
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128, external_field="category")
schema.add_field(field_name="publish_ts", datatype=DataType.INT64, external_field="publish_ts")
schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=65535, external_field="body")

client.create_collection(collection_name="lake_docs", schema=schema)
```

预期输出：`create_collection()` 正常返回即表示外部集合 `lake_docs` 建立，此时 `body` 等列仍在湖上 Parquet 里、没被拷进 Milvus。用 `client.query(collection_name="lake_docs", filter="doc_id >= 0", output_fields=["doc_id", "title", "category"], limit=3)` 能看到前几行是从源文件现读出来的。

> ⚠️ **注意**：没有对象存储的读者，可用 `volume://my_volume/path/to/a/folder` 形式挂载已配置的本地卷（官方外部集合页的 REST / Node 示例使用该写法；`my_volume` 需由部署的 volume 配置提供）。外部集合是只读的，别指望对它 insert——想改数据回源 Parquet 改、再 refresh（7.5.4）。

### 7.5.3.3 库内派生检索字段

本节一次配好两类函数：BM25 稀疏向量（全文检索）与 MinHash 签名（近重复检测），说明「原文留外部、检索字段在库里算」。它不展开函数原理——BM25/MinHash/embedding 的完整语法见 7.2.3.5（外部列直连函数输出字段）与 7.2.5.2（`add_function_field` 的完整写法），这里只做「挂上去、验一下」。

思路：`body` 的值来自湖上文件、不落库；`body_sparse`（BM25 稀疏向量）与 `body_minhash`（MinHash 二值签名）由函数在库内从 `body` 现算出来。这样「湖上原文 + 库内检索表示」不用双写，源表也不拷进来。下面是从零建集合时把函数一起定义进 schema 的写法；若集合已建好，改用 7.2.5.2 的 `add_function_field()` 追加。

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = MilvusClient.create_schema(
    external_source="s3://<bucket>/milvus30_demo/",
    external_spec='{"format": "parquet"}',
)
schema.add_field(field_name="doc_id", datatype=DataType.INT64, is_primary=True, external_field="doc_id")
schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=65535, external_field="body", enable_analyzer=True)
# 函数产出字段：都不是湖上列，库内现算
schema.add_field(field_name="body_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="body_minhash", datatype=DataType.BINARY_VECTOR, dim=8192)  # dim = 32 * num_hashes

bm25_fn = Function(
    name="body_bm25",
    input_field_names=["body"],
    output_field_names=["body_sparse"],
    function_type=FunctionType.BM25,
)
minhash_fn = Function(
    name="body_minhash_fn",
    input_field_names=["body"],
    output_field_names=["body_minhash"],
    function_type=FunctionType.MINHASH,
    params={"num_hashes": 256, "shingle_size": 3},  # 256 个哈希 → dim 8192
)
schema.add_function(bm25_fn)
schema.add_function(minhash_fn)

client.create_collection(collection_name="lake_docs", schema=schema)
```

预期输出：`client.describe_collection("lake_docs")` 能看到 `body_sparse`（SPARSE_FLOAT_VECTOR）与 `body_minhash`（BINARY_VECTOR, dim=8192）两个字段及挂在上面的函数。检索时 `body_sparse` 按词频相关度给全文打分，`body_minhash` 用于近似 Jaccard 的文档去重。

> ⚠️ **注意**：
> - MinHash 的输出字段必须是 `BINARY_VECTOR`，且维度满足 `dim % 32 == 0`（官方：每个签名是 32 位哈希，完整签名打包成 `32 * num_hashes` 维）。BM25/MinHash 的输入必须是 `VARCHAR` 字段且开 `enable_analyzer=True`。
> - 「外部列 + 函数」这一组合官方 release notes 明确支持（原文：External fields can now feed function output fields such as BM25 sparse vectors, MinHash signatures, and text embeddings），但完整成例本机缓存中未找到（待验证）；此处按 7.2.3.5 的 `Function` 组装拼成，写法以 7.2.3.5 为准。

### 7.5.3.4 用 StructArray 承载切片与多向量

本节给「一文档多切片、每切片一个向量 + 若干标量」的字段设计与写入代码，体现父子同集合。它不重讲 StructArray 语法细节——建 schema、`element_type=STRUCT`、路径语法与限制清单见 7.3.2，这里只把内部集合 `rag_corpus` 立起来。

与外部集合 `lake_docs` 分工：`lake_docs` 负责「原文留在湖上、只读检索」；`rag_corpus` 是普通集合，负责「切片级 RAG 检索 + 分面/排序」，两套能力在同一场景里互补。

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
# 集合级字段：主键 + 文章级标量（publish_ts / rating 留给 7.5.3.6 的排序与分面）
schema.add_field(field_name="doc_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="publish_ts", datatype=DataType.INT64)
schema.add_field(field_name="rating", datatype=DataType.FLOAT)
schema.add_field(field_name="title_vector", datatype=DataType.FLOAT_VECTOR, dim=4)

# 每个切片元素遵循的 schema（嵌套：集合 -> chunks 数组 -> 单个元素）
chunk_schema = client.create_struct_field_schema()
chunk_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
chunk_schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=128)
chunk_schema.add_field(field_name="quality_score", datatype=DataType.FLOAT)
chunk_schema.add_field(field_name="emb_list_vector", datatype=DataType.FLOAT_VECTOR, dim=4)  # EmbeddingList 检索
chunk_schema.add_field(field_name="emb", datatype=DataType.FLOAT_VECTOR, dim=4)              # 元素级检索

schema.add_field(
    field_name="chunks",
    datatype=DataType.ARRAY,
    element_type=DataType.STRUCT,
    struct_schema=chunk_schema,
    max_capacity=1000,   # 必填：限制单行最多存多少个切片
)
client.create_collection(collection_name="rag_corpus", schema=schema)
```

写入嵌套数据（`chunks` 是「对象数组」，键名是子字段名，**不要**用路径语法）：

```python
data = [
    {
        "doc_id": 1, "title": "Search Aggregation 入门", "category": "search",
        "publish_ts": 1720000000, "rating": 4.8,
        "title_vector": [0.12, 0.08, 0.32, 0.48],
        "chunks": [
            {"text": "Search Aggregation 把候选分桶后算桶内统计。",
             "section": "intro", "quality_score": 0.96,
             "emb_list_vector": [0.10, 0.20, 0.30, 0.40], "emb": [0.10, 0.20, 0.30, 0.40]},
            {"text": "分面检索一次请求返回各分面值与计数。",
             "section": "facet", "quality_score": 0.91,
             "emb_list_vector": [0.16, 0.24, 0.35, 0.45], "emb": [0.16, 0.24, 0.35, 0.45]},
        ],
    },
    # 其余文档同构
]
result = client.insert(collection_name="rag_corpus", data=data)
print(result)
```

预期输出类似 `{'insert_count': 2, 'ids': [1, ...], 'cost': 0}`。三个要点（`max_capacity` 必填、子字段类型白名单、同一向量子字段只能建一个索引）见 7.3.2.2 与 7.3.2.6，此处不重复。

### 7.5.3.5 索引配置

本节给两个集合一次性配齐四类索引：稠密向量、稀疏向量（新算法）、StructArray 子字段、标量；并给「确认实际生效」的验证行。它不重讲索引选型原理——HNSW/IVF 见 Chapter 2，稀疏新算法 `inverted_index_algo` 的六个取值见 7.4.2.1，这里只贴配置与验证。

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# lake_docs：稀疏（BM25 新算法）+ MinHash 索引
sparse_index = client.prepare_index_params()
sparse_index.add_index(field_name="body_sparse", index_type="SPARSE_INVERTED_INDEX",
                       metric_type="BM25", params={"inverted_index_algo": "DAAT_MAXSCORE"})
sparse_index.add_index(field_name="body_minhash", index_type="MINHASH_LSH",
                       metric_type="MHJACCARD", params={"mh_element_bit_width": 32})
client.create_index(collection_name="lake_docs", index_params=sparse_index)
client.load_collection("lake_docs")

# rag_corpus：稠密 + struct 子字段 + 标量
idx = client.prepare_index_params()
idx.add_index(field_name="title_vector", index_type="HNSW", metric_type="COSINE",
              params={"M": 16, "efConstruction": 200})
idx.add_index(field_name="chunks[emb_list_vector]", index_type="HNSW", metric_type="MAX_SIM_COSINE",
              params={"M": 16, "efConstruction": 200})
idx.add_index(field_name="chunks[emb]", index_type="HNSW", metric_type="COSINE",
              params={"M": 16, "efConstruction": 200})
idx.add_index(field_name="chunks[section]", index_type="INVERTED")
idx.add_index(field_name="chunks[quality_score]", index_type="STL_SORT")
idx.add_index(field_name="category", index_type="INVERTED")
idx.add_index(field_name="rating", index_type="STL_SORT")
client.create_index(collection_name="rag_corpus", index_params=idx)
client.load_collection("rag_corpus")

# 验证行：打印实际生效的索引，确认子字段路径与稀疏索引都挂上了
print(client.list_indexes(collection_name="lake_docs"))
print(client.list_indexes(collection_name="rag_corpus"))
```

预期输出：两段 `list_indexes` 里能看到 `body_sparse` / `body_minhash` / `title_vector` / `chunks[emb]` 等索引条目，StructArray 子字段以路径 `chunks[...]` 形式出现。

> ⚠️ **注意**：
> - 新稀疏算法（`inverted_index_algo`）生效前提是 `dataCoord.targetVecIndexVersion = 10` 已抬、且对**新建**索引才生效（7.4.2.2 / 7.1.4.4）；没抬时**不报错、仍走旧算法**。`mh_element_bit_width` 在 MinHash 函数输出上必须为 `32`（minhash-function.md）。
> - `list_indexes` 能否回读「当前走的 `inverted_index_algo`」，官方未给显式字段（待验证，见 7.4.2.2），所以确认「真走新路径」的可靠手段仍是改配置前后的延迟/体积对比。

### 7.5.3.6 一次请求完成检索 + 排序 + 分面 + 重排

本节是本章高潮：把检索、过滤、排序、聚合分面、重排这些过去要「多次客户端往返 + 拼装」的事，换成服务端一次下推，并贴真实返回片段。它不重讲每项 API 的原理——排序见 7.3.5.1、聚合分面见 7.3.5.3、重排函数链见 7.3.6.2，这里只做组装对照。

先看 2.6 要几次调用：按距离取 top-K → 客户端再查一次拉回排序字段（或过取一批）→ 客户端自己分桶计数 → 客户端/外部模型再重排，四步至少 3–4 次往返，且排序、分面、重排全在客户端拼。3.0 把这些下沉进请求本身：

**① 检索 + 过滤 + 排序（一次下推）**——`filter` 加顶层标量过滤，`order_by_fields` 让排序发生在服务端 top-K 之内：

```python
query_vector = [0.19, 0.24, 0.30, 0.37]

results = client.search(
    collection_name="rag_corpus",
    data=[query_vector],
    anns_field="title_vector",
    filter="category == 'search'",
    order_by_fields=[{"field": "rating", "order": "desc"}],   # 服务端排序
    limit=5,
    output_fields=["doc_id", "title", "category", "rating"],
)
```

预期输出：返回与查询向量最接近、且 `category == 'search'` 的 top-5，按 `rating` 从高到低排——排序发生在 top-K 之内，不用客户端过取再排。

**② 检索 + 过滤 + 聚合分面（一次下推）**——`search_aggregation` 把 ANN 候选按标量字段分桶、算桶内统计、返回代表命中：

```python
from pymilvus import SearchAggregation, TopHits

aggregation = SearchAggregation(
    fields=["category"],                # 按 category 分桶
    size=3,                             # 最多返回 3 个桶
    metrics={
        "doc_count": {"count": "*"},
        "avg_rating": {"avg": "rating"},
    },
    order=[{"avg_rating": "desc"}],     # 桶按平均分降序
    top_hits=TopHits(size=2, sort=[{"rating": "desc"}]),
)

result = client.search(
    collection_name="rag_corpus",
    data=[query_vector],
    anns_field="title_vector",
    filter="rating >= 4.0",
    search_aggregation=aggregation,
    output_fields=["doc_id", "title", "category", "rating"],
)

buckets = result.agg_buckets[0]   # 注意：设了 search_aggregation 后，result[0] 不再返回普通命中
for b in buckets:
    print(b)
```

预期输出：一个桶树——每个 `category` 一个桶，带 `doc_count`（命中数）、`avg_rating`（平均分），桶内 `hits` 是该分面下的代表文档。

![一次平坦的检索结果变成一组可比较的分桶](/images/milvus30_search_aggregation_overview.png)

> 图源：Milvus 官方文档（v3.0.x）

**③ 多向量融合 + 重排（一次下推）**——`hybrid_search` + Function 链把文章级向量与切片级向量融合，再用 RRF 重排：

```python
from pymilvus import AnnSearchRequest, Function, FunctionType

title_req = AnnSearchRequest(data=[query_vector], anns_field="title_vector", limit=10)
chunk_req = AnnSearchRequest(data=[query_vector], anns_field="chunks[emb]", limit=10)

rrf = Function(
    name="rrf",
    input_field_names=[],               # 重排函数必须为空列表
    function_type=FunctionType.RERANK,
    params={"reranker": "rrf", "k": 60},
)
results = client.hybrid_search(
    collection_name="rag_corpus",
    reqs=[title_req, chunk_req],
    ranker=rrf,
    limit=5,
    output_fields=["doc_id", "title", "category"],
)
```

预期输出：文章级与切片级两路候选按 RRF 融合后的最终排序。

> ⚠️ **注意**：标题里的「一次请求」是相对 2.6 的「多次往返 + 客户端拼装」说的——**每项能力都变成了服务端一次下推，客户端零拼装**。真要四项叠加时有两条官方边界要认清：`search_aggregation` 设了之后 `result[0]` 不再返回普通命中、`limit` 被忽略，且官方明确它不能与 `group_by_field(s)`、Hybrid Search 组合（PyMilvus 会抛 `ParamError`）。所以「排序」与「分面」是两个不同的下推点（排序走 `order_by_fields`，分面走 `search_aggregation`），不能字面上塞进同一次 HTTP——价值在「下推」，不在「一次 HTTP 请求数」。重排里 Boost 权重、`k` 取值等细节见 7.3.6.2。

### 7.5.3.7 发布前打快照与回滚演练

本节演练一次「故意写坏 → 从快照恢复 → 结果一致」，让你亲眼看到快照创建秒级、恢复要拷段文件、保留会占存储。它不重讲快照四个动作的语法——`create_snapshot` / `list_snapshots` / `restore_snapshot` / `drop_snapshot` 见 7.2.4.2，这里只做演练并记录三个数。

```python
import time
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
coll = "rag_corpus"

# 打快照前先 flush，避免漏掉没落盘的数据（官方建议）
client.flush(collection_name=coll)

# 1) 创建快照，记录耗时（只写元数据，官方口径毫秒级）
t0 = time.perf_counter()
client.create_snapshot(collection_name=coll, snapshot_name="pre_release")
create_ms = (time.perf_counter() - t0) * 1000
print(f"create_snapshot: {create_ms:.1f} ms")

# 2) 故意写坏：删掉一批数据，模拟发布事故
#    （外部集合只读，这里在内部集合 rag_corpus 上演练）
client.delete(collection_name=coll, filter="category == 'data-model'")

# 3) 恢复到新集合，异步返回 job_id（copy-segment 只拷现有段文件、不重导不重建索引）
t0 = time.perf_counter()
job_id = client.restore_snapshot(snapshot_name="pre_release", collection_name="rag_corpus_rollback")
restore_ms = (time.perf_counter() - t0) * 1000
print(f"restore_snapshot submitted: {restore_ms:.1f} ms (job_id={job_id})")
```

预期输出：`create_snapshot` 返回毫秒级耗时（只写元数据）；`restore_snapshot` 异步返回 `job_id`，用 `client.list_restore_snapshot_jobs()` 查进度（copy-segment 只拷现有段文件、不重导不重建索引）。

三个数必须记录（本次演示未实跑，标「待补测」）：

| 项 | 记录 | 本次 |
| --- | --- | --- |
| 快照创建耗时 | 只写元数据，官方口径毫秒级 | 待补测 |
| 恢复耗时 | copy-segment，随数据量增长 | 待补测 |
| 恢复后一致性 | 恢复出的新集合行数/向量数应与打快照时一致 | 待补测 |

验证一致性（等恢复 job 完成后再查）：

```python
before = client.get_collection_stats(collection_name="rag_corpus")["row_count"]
after = client.get_collection_stats(collection_name="rag_corpus_rollback")["row_count"]
print(f"before = {before} | after = {after}")
```

预期输出：`before` 是打快照时的行数（写坏之前记），`after` 是恢复集合的行数，两数一致即回滚成功。

> ⚠️ **注意**：快照只写元数据所以创建快，但会**钉住**被引用文件不被 GC，保留一份快照的成本约等于集合大小（极端可达 2×），要定期 `drop_snapshot()` 清理（见 7.2.4.1）。快照依赖 Storage V3 已开启——7.5.3.1 的第二项没过，这节直接跑不动。

## 7.5.4 日常运行：增量与演进

本节讲链路跑起来之后的两件日常事：外部数据变了怎么追平、上线后换嵌入模型怎么加列回填。它不重复语法——`refresh_external_collection` 见 7.2.3.6、`add_collection_field` 与回填见 7.2.5.1 / 7.2.5.2，这里只把这两件事接进本章的链路。

**外部数据刷新**：湖上 Parquet 更新后，外部集合不会自动跟上，要手动 `refresh`。再次强调：外部集合是**只读**的，更新靠「回源改 + refresh」，不是写入。

```python
job_id = client.refresh_external_collection(collection_name="lake_docs")
# 异步：轮询到 RefreshCompleted 即为追平（完整轮询写法见 7.2.3.6）
```

预期输出：refresh 只扫文件元数据、更新 manifest，通常 150–250 ms（官方口径），不搬数据、不改源文件。湖上列结构变化（如新增一列）则先 `add_collection_field()` 映射再 refresh，见 7.2.3.7。

**换嵌入模型：在线加列回填**。上线后想从「单向量」升级成「双向量混合检索」，不用重建集合，`add_collection_field()` 加一个可空向量字段，再 upsert 回填：

```python
from pymilvus import DataType

client.add_collection_field(
    collection_name="rag_corpus",
    field_name="title_vector_v2",
    data_type=DataType.FLOAT_VECTOR,
    dim=4,
    nullable=True,
)
# 新字段对老数据是 NULL，需按主键 upsert 回填（见 7.2.5.2 外部回填）
client.upsert(
    collection_name="rag_corpus",
    data=[{"doc_id": 1, "title_vector_v2": [0.11, 0.22, 0.33, 0.44]}],
)
```

预期输出：`describe_collection()` 看到新字段 `title_vector_v2`；老数据补上前按该字段检索会被直接跳过（NULL 向量不参与检索），补完后可进混合检索。

> ⚠️ **注意**：新增向量字段老数据为 `NULL`，补上向量前按该字段检索会被直接跳过（不是排末尾、是不出现）；删字段、回填的硬限制见 7.2.5.1 / 7.2.5.2。

## 7.5.5 结果观察与代价

本节做三段总结：这次链路用到了哪些 3.0 新能力、同样的事在 2.6 下怎么做、实测差异是什么；再诚实列出踩到的限制。它不重复各节限制清单的内容，只引用位置。

**用了哪些 3.0 新能力**（本链路逐一命中）：

- 外部集合（7.5.3.2）：湖上 Parquet 原地读，不拷数据。
- 函数派生字段（7.5.3.3）：BM25 稀疏向量 + MinHash 签名在库里现算。
- StructArray（7.5.3.4）：一文档多切片父子同集合。
- 新稀疏索引算法（7.5.3.5）：`inverted_index_algo` 按负载选算法。
- 服务端排序 / 聚合分面 / 重排（7.5.3.6）：从客户端拼装变成服务端下推。
- 快照（7.5.3.7）：发布前秒级打点、可回滚。

**同样的事在 2.6 下怎么做**：

- 湖上数据要先写 ETL 拷进 Milvus（多一份存储、多一条同步管道），没有外部集合；检索表示（BM25/去重）要自己算好再写回，没有函数派生。
- 切片级父子检索要么多建一张子表、要么把切片铺平成行，没有 StructArray 的父子同集合。
- 排序/分面/重排全是客户端做：过取、拉字段、分桶、调模型，多次往返 + 拼装。
- 发布回滚靠备份重导（重建索引、耗时），没有秒级快照。

**实测差异**：本教程作者未在本机完整跑过这条链路的端到端数字，以下全部标「待补测」，不搬任何宣传数字——快照创建/恢复耗时、稀疏新算法的延迟/体积差、一次下推相对客户端拼装的延迟差，都建议你在自己的数据上按 7.4.2.3 / 7.5.3.7 的自测脚本量一次。

**踩到的限制**（引用各节清单，不复述）：

- 外部集合只读、TEXT 在 3.0.0 不支持外部集合、只能加源里已有的字段（7.2.3）。
- StructArray 子字段类型白名单、同一向量子字段只能建一个索引、子字段结构不可增量改（7.3.2.6）。
- `search_aggregation` 不能与 Hybrid Search / Grouping 组合，设了后 `result[0]` 空、`limit` 被忽略（7.5.3.6）。
- 快照钉住文件、保留成本约等于集合大小，需定期清理（7.2.4.1）。
- 两个开关「不报错、只是没生效」的坑（7.5.3.1 / 7.4.2.2）。

## 7.5.6 本章小结

本章核心掌握：

1. **数据可以留在原地**：外部集合让湖上 Parquet 原地读、只读、靠 refresh 追平；BM25/MinHash 函数让检索字段在库里从外部列现算，不双写、不搬数据（7.5.3.2、7.5.3.3）。
2. **父子同集合用 StructArray**：一文档多切片存进同一行，切片级检索、分面、排序在一个请求里完成（7.5.3.4）。
3. **检索从「客户端拼装」变成「服务端下推」**：过滤、排序（`order_by_fields`）、聚合分面（`search_aggregation`）、重排（Function 链）各自一次下推，客户端零拼装（7.5.3.6）。
4. **发布可回滚**：快照秒级创建、恢复走 copy-segment，但要记住「保留占存储、需定期清理」（7.5.3.7）。

至此，Milvus 3.0.x 的新增能力全部覆盖：7.2 讲了「数据放哪」（Storage V3、外部集合、快照），7.3 讲了「一次请求能算什么」（StructArray、TEXT、排序/聚合/分面、重排），7.4 讲了「写入与索引怎么落地」（Woodpecker、稀疏索引换代），7.5 把它们串成了一条可回滚的链路。未列出的均为与 2.6 相同的行为，见前六章。
