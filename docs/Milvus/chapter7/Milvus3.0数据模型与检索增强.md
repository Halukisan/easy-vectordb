# Chapter 7.3 Milvus 3.0：FunctionChain 与多向量（实战）

## 7.3.0 两个场面

**场面一。** 用户在耳机商城搜「降噪耳机」。产品的要求是：最像的排前面，最近上架的要提前，卖得好的也要提前。向量检索只按相似度排——一条两年前上架、早该下架的老款，只要向量最像，就永远占着第一。过去只能把候选拉回应用层，自己算「上架多少天」「热度多少」，自己写加权和排序：多一次往返，多一段代码。

**场面二。** 换嵌入模型。300 万条商品向量要按新模型重算，可集合里没有地方放新向量，只能建新集合、全量重导——一周的迁移窗口，期间还要双写。

3.0 对这两个场面各给了一样东西：

1. **FunctionChain（函数链）**：把「改分数 → 排序 → 截断」声明成一条**有顺序、带类型的处理链**，作为**一次搜索请求的一部分**执行，对搜索结果的行做后处理。官方发布说明管它叫 *Function Chain reranking*，列的能力是「分数变换与合成、基于模型的重排、排序、候选裁剪」，并特别强调 **without client-side orchestration**——不需要客户端在中间编排。
2. **多向量列 + 部分更新**：一个集合里存两套向量，换模型只回填新的一列，查询时改一个字段名。

C++ API 文档对它的定义只有两句，但把边界划得很清楚：

> This class builds a function chain that post-processes search results. A function chain runs a pipeline of operations (map, sort, limit, …) on the result columns of a Search call.

翻译：**这个类用来构造一条给搜索结果做后处理的函数链；一条链就是在一次 Search 调用的结果列上，依次执行 map、sort、limit 这些操作。** 两个要点：链是**声明**出去、由服务端执行的；它处理的是**结果行**（result columns），不是集合里的数据。

下面用一个 **12 条耳机** 的小商品库把两件事各做一遍。数据小是为了每行结果你都能拿计算器核对；换成 300 万条，代码不用改一个字。

## 7.3.1 准备：12 条耳机

12 条商品，字段就是业务里常见的那些：名字、价格、**上架天数**、**热度**（0~1 的销量分），再加两列向量 `vec_v1`（老模型）、`vec_v2`（新模型）。

向量怎么造的：先随机生成一个查询向量 `q`（真实系统里它是嵌入模型对用户搜索词的输出），每条商品的向量 = `q` + 一个扰动，扰动越大越不像。所以相似度高低是预先设计好的，排名结果可以肉眼核对。

```python
import numpy as np
from pymilvus import MilvusClient, DataType, FunctionChain, FunctionChainStage
from pymilvus.function_chain import col, fn

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
C = "headphones"
if client.has_collection(C):
    client.drop_collection(C)

rng = np.random.default_rng(1)
q = rng.standard_normal(8).astype(np.float32)
items = [
    ("索尼 WH-1000XM5", 1999, 60, 0.95, 0.05),
    ("苹果 AirPods Max", 4399, 45, 0.80, 0.15),
    ("小米 降噪耳机 4 Pro", 699, 2, 0.30, 0.30),
    ("华为 FreeBuds Pro 3", 1099, 7, 0.60, 0.35),
    ("索尼 WF-1000XM5", 1899, 120, 0.85, 0.45),
    ("苹果 AirPods Pro 2", 1899, 3, 0.90, 0.55),
    ("小米 Redmi Buds 5", 199, 200, 0.70, 0.60),
    ("漫步者 NeoBuds Pro", 499, 1, 0.10, 0.65),
    ("华为 FreeBuds SE 2", 299, 90, 0.40, 0.70),
    ("三星 Galaxy Buds3", 999, 15, 0.55, 0.75),
    ("索尼 WH-CH720N", 899, 30, 0.45, 0.80),
    ("击音 入门耳机", 89, 5, 0.05, 0.90),
]
rows = []
for i, (name, price, days, pop, d) in enumerate(items, start=1):
    u = rng.standard_normal(8)
    u /= np.linalg.norm(u)
    rows.append({"id": i, "name": name, "price": price, "days_ago": days,
                 "popularity": pop,
                 "vec_v1": (q + d * u).astype(np.float32).tolist(),
                 "vec_v2": (q + ((i * 7) % 12 + 1) * 0.08 * u).astype(np.float32).tolist()})

schema = client.create_schema(auto_id=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("name", DataType.VARCHAR, max_length=64)
schema.add_field("price", DataType.DOUBLE)
schema.add_field("days_ago", DataType.INT64)
schema.add_field("popularity", DataType.DOUBLE)
schema.add_field("vec_v1", DataType.FLOAT_VECTOR, dim=8)
schema.add_field("vec_v2", DataType.FLOAT_VECTOR, dim=8)

ip = client.prepare_index_params()
ip.add_index(field_name="vec_v1", index_type="AUTOINDEX", metric_type="COSINE")
ip.add_index(field_name="vec_v2", index_type="AUTOINDEX", metric_type="COSINE")
client.create_collection(collection_name=C, schema=schema, index_params=ip)
client.insert(C, rows)
client.flush(C)
client.load_collection(C)
```

数一下行数：

```python
print(client.query(C, filter="id >= 0", output_fields=["count(*)"])[0])
```

```
{'count(*)': 12}
```

两列向量都建在 `COSINE` 上——为什么不选 `L2`，7.3.2 的坑里会说。

## 7.3.2 FunctionChain：一条链完成「新鲜度加权 → 排序 → 截断」

**问题。** 先看纯向量检索给出什么：

```python
def show(title, res):
    print(title, flush=True)
    for h in res[0]:
        e = h["entity"]
        print(f'  {e["name"]:<20}{e["days_ago"]:>4} 天前  热度 {e["popularity"]:.2f}  相似度 {h["distance"]:.3f}', flush=True)


res = client.search(C, data=[q], anns_field="vec_v1", limit=5,
                    output_fields=["name", "days_ago", "popularity"])
show("== 裸搜 top5 ==", res)

```

```
== 裸搜 top5 ==
  索尼 WH-1000XM5         60 天前  热度 0.95  相似度 1.000
  苹果 AirPods Max        45 天前  热度 0.80  相似度 0.997
  小米 降噪耳机 4 Pro          2 天前  热度 0.30  相似度 0.990
  华为 FreeBuds Pro 3      7 天前  热度 0.60  相似度 0.986
  索尼 WF-1000XM5        120 天前  热度 0.85  相似度 0.977
```

前两名是 60 天和 45 天前上架的老款，120 天前的 WF-1000XM5 也还在榜上；3 天前上架的 AirPods Pro 2 连前五都没进。**纯向量检索只认「像」，不认「新」和「热」。**

要补的正是这个：在返回前，给每行算出新鲜度、按权重合成新分数、排序、只留 5 条。FunctionChain 就是把这几步声明下来、随请求发过去：

```python
# 链：先建一个空链，再一段一段往上挂
chain = FunctionChain(FunctionChainStage.L2_RERANK, name="fresh_and_hot")
# 第 1 段 map：算出一列“新鲜度”，刚上架=1，7 天前=0.5，30 天前≈0.05
chain = chain.map("freshness", fn.decay(col("days_ago"), function="exp", origin=0, scale=7))
# 第 2 段 map：把相似度、新鲜度、热度按 5:3:2 合成新的 $score
chain = chain.map("$score", fn.num_combine(
    col("$score"), col("freshness"), col("popularity"),
    mode="weighted", weights=[0.5, 0.3, 0.2]))
# 第 3、4 段：按新分数排序，只留前 5
chain = chain.sort(col("$score"), desc=True).limit(5)

res = client.search(C, data=[q], anns_field="vec_v1", limit=12,
                    output_fields=["name", "days_ago", "popularity"],
                    function_chains=[chain])
show("== 加链 top5 ==", res)

```

```
== 加链 top5 ==
  苹果 AirPods Pro 2       3 天前  热度 0.90  相似度 0.890
  小米 降噪耳机 4 Pro          2 天前  热度 0.30  相似度 0.801
  漫步者 NeoBuds Pro        1 天前  热度 0.10  相似度 0.773
  华为 FreeBuds Pro 3      7 天前  热度 0.60  相似度 0.763
  索尼 WH-1000XM5         60 天前  热度 0.95  相似度 0.691
```

**同一批候选，结果翻了个面。** 逐条对比：

- **苹果 AirPods Pro 2**（3 天前，热度 0.90）：裸搜连前五都进不去 → 现在第 1；
- **小米 降噪耳机 4 Pro**（2 天前，热度 0.30）：第 3 → 第 2；
- **漫步者 NeoBuds Pro**（1 天前，热度只有 0.10）：裸搜进不了前五 → 挤到第 3；
- **索尼 WH-1000XM5**（60 天前，热度 0.95，最像）：第 1 → 掉到第 5。

**`L2_RERANK` 是什么意思。** 链的第一个参数是执行阶段。官方 C++ 文档说，给搜索结果做后处理用 `L0_RERANK`、`L1_RERANK` 或 `L2_RERANK` 三者之一；3.0.0 发布说明写得更具体：`L0` 在 QueryNode 上提前重打分，`L2` 在 Proxy 上把各分片结果归并之后再重排。本文要的是「候选都汇总回来、统一比一次总分」，所以用 `L2`。

**链上四段，各干一件事：**

- `chain.map("freshness", fn.decay(col("days_ago"), function="exp", origin=0, scale=7))`：给每行算一列新值。`col("days_ago")` 取「上架天数」列；`origin=0` 表示「0 天前」得满分 1；`scale=7` 表示每过 7 天衰减一半——1 天前 ≈ 0.91，3 天前 ≈ 0.74，7 天前 = 0.5，60 天前 ≈ 0.003。这一列叫 `freshness`，只活在链里。
- `chain.map("$score", fn.num_combine(..., weights=[0.5, 0.3, 0.2]))`：`$score` 是数据库给每行准备的分数列，裸搜时它就是相似度（COSINE 下 1.000 表示完全一样）。这里把它**覆盖**成三个数的加权和：五成相似度 + 三成新鲜度 + 两成热度。
- `chain.sort(col("$score"), desc=True)`：按新分数从大到小。
- `chain.limit(5)`：只留前 5 条。

| 链上的段 | 官方说明 | 本文这一段 |
| --- | --- | --- |
| `map` | 把表达式的结果写进指定的输出列 | 算 `freshness`；把 `$score` 覆盖成加权和 |
| `sort` | 按某列排序，可再指定一个 tie-break 列（同分时用） | 按 `$score` 降序 |
| `limit` | 截断到 N 条，可带 offset | 只留前 5 条 |

**换一套权重，排序就变一种。** 产品说「新品优先」，把权重从 5:3:2 改成 2:6:2：

```python
# 换一套权重：更看重新鲜度
chain2 = (FunctionChain(FunctionChainStage.L2_RERANK, name="fresher")
          .map("freshness", fn.decay(col("days_ago"), function="exp", origin=0, scale=7))
          .map("$score", fn.num_combine(col("$score"), col("freshness"), col("popularity"),
                                        mode="weighted", weights=[0.2, 0.6, 0.2]))
          .sort(col("$score"), desc=True).limit(5))
res = client.search(C, data=[q], anns_field="vec_v1", limit=12,
                    output_fields=["name", "days_ago", "popularity"],
                    function_chains=[chain2])
show("== 权重 2:6:2 ==", res)

```

```
== 权重 2:6:2 ==
  苹果 AirPods Pro 2       3 天前  热度 0.90  相似度 0.821
  漫步者 NeoBuds Pro        1 天前  热度 0.10  相似度 0.756
  小米 降噪耳机 4 Pro          2 天前  热度 0.30  相似度 0.750
  华为 FreeBuds Pro 3      7 天前  热度 0.60  相似度 0.617
  击音 入门耳机                5 天前  热度 0.05  相似度 0.557
```

「击音 入门耳机」（5 天前上架，热度只有 0.05）挤进第 5——更看重新鲜度后，冷门新品也有了机会；而 60 天前的索尼 XM5 直接出局。权重是业务口径，改一个数字就行，不用动代码结构。
**四个坑，都是踩出来的：**

- **L2 度量下不能这样加权。** 本文第一版索引建在 `L2` 上，同一条链跑出来第 1 名是「击音 入门耳机」——相似度倒数第一，只因为 5 天前上架。原因：L2 的 `$score` 是**距离，越小越好**，而链按「越大越好」加权，再叠上 0~1 的新鲜度和热度，距离那点差异被彻底淹没。要么索引用 `COSINE`/`IP`（越大越像），要么先用 `map` 把距离转成相似度再用。
- **分数列和中间列都取不到。** `output_fields=["$score"]` 报 `field $score not exist`，`output_fields=["freshness"]` 同样报错。想看分数就用返回里的 `h["distance"]`——链的终值就在这一列。
- **链和 `ranker` 互斥。** 同一个请求同时传 `function_chains` 和 `ranker`，直接抛 `ParamError: function_chains and ranker cannot be used together`。融合用 `hybrid_search`，重排用链，二选一。这不是实测的巧合：官方 C++ 文档原文就写着 *Function chains and rerank cannot be used together.*
- **外层 `limit` 是候选池，不是返回条数。** 本文 `limit=12` 把 12 条全给链；若写成 `limit=5`，链只能在这 5 条里排，「原本进不了前五」的 AirPods Pro 2 根本捞不上来。候选池太小，是链「看起来没生效」的最常见原因。

**做完之后：** 加权、排序、截断在一次请求里完成，客户端只负责展示；换权重改一个数字。

**边界。** 链只处理「召回出来的行」。分组统计（各品牌多少条、均价）要用 `query` 的 `group_by_fields`；纯关键词召回要用 BM25 全文检索——这两件不归链管。

## 7.3.3 换模型不重建集合：两列向量 + 部分回填

**问题。** 老模型 `vec_v1` 要换成新模型 `vec_v2`。集合里两列都建好了（7.3.1 建表时就是这么建的），先看两列搜同一句话的差别，再只改一条做回填：

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
C = "headphones"
q = np.random.default_rng(1).standard_normal(8).astype(np.float32)
q = q.tolist()

for f in ["vec_v1", "vec_v2"]:
    res = client.search(C, data=[q], anns_field=f, limit=3, output_fields=["name"])
    print(f, [(h["entity"]["name"], round(h["distance"], 3)) for h in res[0]], flush=True)

row = client.query(C, filter="id == 5", output_fields=["name", "price", "days_ago"])[0]
print("before:", row["name"], row["price"], row["days_ago"], flush=True)
client.upsert(C, [{"id": 5, "vec_v2": q}], partial_update=True)
client.flush(C)
client.load_collection(C)
row = client.query(C, filter="id == 5", output_fields=["name", "price", "days_ago"])[0]
print("after:", row["name"], row["price"], row["days_ago"], flush=True)
res = client.search(C, data=[q], anns_field="vec_v2", limit=3, output_fields=["name"])
print("v2 search:", [(h["entity"]["name"], round(h["distance"], 4)) for h in res[0]], flush=True)
```

```
vec_v1 [('索尼 WH-1000XM5', 1.0), ('苹果 AirPods Max', 0.997), ('小米 降噪耳机 4 Pro', 0.99)]
vec_v2 [('击音 入门耳机', 0.999), ('小米 Redmi Buds 5', 0.997), ('苹果 AirPods Max', 0.993)]
before: 索尼 WF-1000XM5 1899.0 120
after: 索尼 WF-1000XM5 1899.0 120
v2 search: [('索尼 WF-1000XM5', 1.0), ('击音 入门耳机', 0.9993), ('小米 Redmi Buds 5', 0.9971)]
```

**两个模型给出的近邻完全不同**——`vec_v1` 下最像的是索尼 XM5，`vec_v2` 下第一是「击音 入门耳机」。不同模型把商品映射到的向量空间不一样，不能混用，也不能拿旧向量凑合。

回填只动一列：`upsert([{"id": 5, "vec_v2": q}], partial_update=True)`，其余字段（名字、价格、上架天数）原样保留——`before` 和 `after` 两行输出完全一致就是证据。回填完再搜 `vec_v2`，索尼 WF-1000XM5（id=5）原本排不进前三，回填后直接第 1，相似度 1.0。

**要点：**

- **两列在建表时就要声明并建索引。** 少一列索引，`load_collection` 直接失败：`there is no vector index on field: [vec_v2], please create index firstly: invalid parameter`。
- **`partial_update=True` 是合并语义。** 只覆盖给出的列；不带这个参数，没给的字段会被写成默认值。真实迁移时按 `id` 分批回填，一组几千条。
- **切换只是改一个参数。** 查询的 `anns_field` 从 `vec_v1` 换成 `vec_v2`；稳妥做法是新旧两列各搜一批典型查询，比对 top 10 再切。
- **别等上线再加列。** v3.0.2 上给已有集合 `add_collection_field` 加向量列后再 `create_index` 会一直停在 `InProgress`（实测卡在 `pendingIndexRows`），新列用不了。第二列向量在第一次建表时就一起建。

**做完之后：** 换模型 = 回填新列 + 改 `anns_field`，集合、索引、其它字段全程不动，也没有双写和切换窗口。

## 7.3.4 小结

- **FunctionChain 是 3.0 的搜索结果后处理链**：官方定义是「对一次 Search 调用的结果列，依次执行 map、sort、limit 等操作的有序、带类型的管道」，客户端声明、服务端执行，随 `search(..., function_chains=[chain])` 一次发过去。
- **`$score` 是分数列**：裸搜时是相似度（COSINE，越大越好；L2 则是距离，越小越好），被链覆盖后就是综合分；它和 `map` 的中间列都不能放进 `output_fields`。
- **权重就是业务口径**：5:3:2 → 2:6:2，结果从「新+热」偏向「新」，改一个数组即可。
- **外层 `limit` 是候选池**，链内 `limit` 才是返回条数。
- **链与 `ranker` 互斥**；融合走 `hybrid_search`，重排走链。
- **换模型不重建**：建表时声明第二列向量并建索引；回填用 `upsert(..., partial_update=True)` 只给 `id` + 新向量；查询改 `anns_field`。

**参考文档**：[C++ FunctionChain API](https://milvus.io/api-reference/cpp/v3.0.x/Vector/FunctionChain.md#Example) · [Java FunctionChain API](https://milvus.io/api-reference/java/v3.0.x/v2/Vector/FunctionChain/FunctionChain.md) · [Python search() 的 function_chains 参数](https://milvus.io/api-reference/pymilvus/v3.0.x/MilvusClient/Vector/search.md) · [RESTful Search](https://milvus.io/api-reference/restful/v3.0.x/v2/Vector%20(v2)/Search.md) · [3.0.0 发布说明（Function Chain reranking）](https://milvus.io/docs/release_notes.md#v300)
