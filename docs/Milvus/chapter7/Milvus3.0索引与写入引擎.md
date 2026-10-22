# Chapter 7.4 Milvus 3.0 索引与写入引擎

## 7.4.1 学习目标

读完本章，你能独立完成四件事：为 3.0 的新稀疏索引配参数并验证它真的走新路径（7.4.2）；判断哪些场景该用索引引擎直通而不是内置索引类型（7.4.3）；说清写入引擎从消息队列换成 Woodpecker 换了什么、代价是什么（7.4.6）；以及用异步强制合并治理碎段（7.4.7）。本节不展开任何配置，只给"你该关心什么、读哪一节"的路标。

- **稀疏索引换代（7.4.2）**：稀疏索引从"单一倒排"升级为可按 BM25 / 学习式稀疏 / 延迟预算三选一。你会配 `inverted_index_algo`，并知道新算法必须手动抬高目标索引版本才生效——否则不报错，只是没用到新算法。
- **索引引擎直通（7.4.3）**：拿到一条 Faiss 索引配方串（如 `IVF64,Flat`）就能在 Milvus 里复现经典组合，检索参数透传。你会知道它的边界：只支持 FLOAT/BINARY 向量、跑在 CPU、部分工厂不支持标量过滤。
- **磁盘上的多向量检索（7.4.4）**：给数组结构体（StructArray）里的向量列表建磁盘索引（EmbList + DiskANN），并会按"精度—延迟—内存"三角选搜索策略。
- **写入引擎 Woodpecker（7.4.6）**：说清 3.x 默认写前日志（Woodpecker WAL）如何替换外部消息队列的角色，以及 Standalone / 集群的切换步骤与回退前提。
- **段合并与治理（7.4.7）**：看懂碎段为什么让查询延迟抖动、存储膨胀，会用异步强制合并收口，并牢记官方"生产环境不要用强制合并"的告诫。

如果你只是想给单个密集向量字段建索引，2.3.3 的索引类型总表与 3.7.5 的 AUTOINDEX 入门已经够用；本章讲的是 3.0 相对 2.6 的增量，读之前请先确认你有这批新能力的需求。



## 7.4.2 稀疏向量索引全面换代

### 7.4.2.1 新算法与新能力

本节回答"3.0 稀疏索引到底新增了什么"。它不重讲 `SPARSE_INVERTED_INDEX` 的老参数表（已列于 2.3.3），只列 3.0 的增量算法与增量参数，并各配一句"什么时候该关心"。

3.0 把稀疏向量索引从"单一倒排"升级为"可按负载选算法"，增量如下：

- **新搜索算法族**：SINDI、Block-Max WAND、Block-Max MaxScore。关心点：学习式稀疏向量（如 SPLADE 类）的 IP 检索默认走 SINDI，全文 BM25 默认走 MaxScore——在"召回优先"还是"延迟优先"之间选，就靠它们。
- **倒排列表压缩 + 可配置量化**：压缩与量化等级可配。关心点：稀疏索引体积大，想省存储或省加载内存时。
- **按负载选算法**：一个 `inverted_index_algo` 参数在六种算法间切换，无需改数据。关心点：同一批数据要在不同延迟预算下复用。
- **mmap 加载 / 序列化 / BM25 打分优化**：官方口径称这些优化降低了大规模稀疏与全文检索的索引存储与加载开销。关心点：集合大、加载慢时。

新增算法落到 `inverted_index_algo` 的六个取值：

| 取值 | 定位 | 默认 |
| --- | --- | --- |
| `DAAT_MAXSCORE` | Document-at-a-Time MaxScore，适合高 k、多词查询 | BM25 默认 |
| `DAAT_WAND` | Document-at-a-Time WAND，适合小 topK、短查询 | — |
| `TAAT_NAIVE` | 基础 Term-at-a-Time，作基线或需按全局统计动态打分时 | — |
| `BLOCK_MAX_MAXSCORE` | 带块级 max-score 元数据剪枝的 MaxScore | — |
| `BLOCK_MAX_WAND` | 带块级 max-score 元数据剪枝的 WAND | — |
| `SINDI` | 固定 doc-id 窗口 + SIMD 加速的稀疏倒排 | IP 默认 |

2.6 与 3.0 的差异：

| 维度 | Milvus 2.6 | Milvus 3.0 |
| --- | --- | --- |
| 索引形态 | 单一倒排 | 可按负载选算法，共 6 种取值 |
| 倒排列表压缩 / 可配置量化 | 无 | 有 |
| 默认算法 | 固定 | IP 走 SINDI、BM25 走 DAAT_MAXSCORE |

官方内部基准（非本教程复测）：召回率相近时，压缩后的 BM25 索引约为 2.6 稀疏索引体积的三分之一；学习式稀疏向量上，SINDI 的 QPS 最高约为 MaxScore 的 10 倍。

> 📌 **前置条件**：以上新增算法默认不生效，需手动把目标索引版本抬到 `dataCoord.targetVecIndexVersion = 10` 后重建索引才会走新路径；否则不报错、仍走旧算法。步骤见 7.4.2.2。

![Milvus 索引由数据结构、量化、精排三层构成](/images/milvus30_vector_index_anatomy.png)
> 图源：Milvus 官方文档（v3.0.x）

### 7.4.2.2 生效前提与自查

本节回答"3.0 的新稀疏算法到底要怎么才会真的生效"。它不重讲 7.1.4.4 里已经覆盖的全局开关全貌，只把"改配置 → 重建索引 → 验证走新路径"三步在这里落地，并说清"没开时不报错、只是没用到新算法"这个最高频的坑。

> 📌 **前置条件**：3.0 的新稀疏索引算法是"选择性启用（opt-in）"。官方原文：新引入的索引算法需要手动抬高目标索引版本——`dataCoord.targetVecIndexVersion` 抬到 `10`——之后才会生效，更晚的版本才会默认开启。**不抬高时不会报错，只是建出来的索引仍走旧算法路径。**

三步操作：

1. **改配置**：把 `dataCoord.targetVecIndexVersion` 设为 `10`。这是 Milvus 配置文件里 `dataCoord` 段下的配置项；改完是否需要重启 DataCoord、能否热更新，官方缓存未给出明确说明，标"待验证"。
2. **重建索引**：版本只影响"新建"的索引，已有索引不会自动升级。先把旧索引 `drop_index()` 掉，再按 7.4.2.1 的 `inverted_index_algo` 重新 `create_index()`。
3. **验证走新路径**：官方原文没有给出"运行时回读当前索引走的是哪个算法版本"的显式字段（待验证）。可操作的间接判据是——配置生效后，给 IP 稀疏字段配 `inverted_index_algo="SINDI"` 能真正走 SINDI；若版本没生效，Milvus 会静默回退到旧算法，不报任何错。因此建议先在低风险环境做"改配置前 vs 改配置后"的延迟/体积对比来确认（数值自己实测，见 7.4.2.3 的自测脚本）。

与 7.1.4.4 的分工：那里讲开关在全局配置层的位子与影响面，本节只讲稀疏索引这一处新算法的落地三步；标量索引版本（`dataCoord.targetScalarIndexVersion = 4`）在 7.4.5.2 单独讲。



### 7.4.2.3 选型与调优

本节回答"三种稀疏检索场景各该选哪个算法、配什么参数"。它不做跨索引的大横评（2.3.5 已有性能权衡），只比 3.0 新算法彼此之间，并给一份"同一份语料跑三种配置"的自测脚本。

按场景选型（算法取值均为 `inverted_index_algo`）：

- **全文 BM25 检索**：默认 `DAAT_MAXSCORE`，适合高 k、多词查询。关键参数 `bm25_k1`（默认 1.2，推荐 [1.2, 2.0]，调大给词频更高权重）与 `bm25_b`（默认 0.75，范围 [0, 1]，调大加强长度归一化）。代价：完整打分，词数多时延迟上升。
- **学习式稀疏（SPLADE 类）IP 检索**：默认 `SINDI`（SIMD 加速），是官方口径下稀疏 IP 检索的默认。搜索侧可调 `drop_ratio_search`（[0.0, 1.0)，丢弃最低幅值分量的比例）降噪换速度。代价：丢掉低幅值分量可能略微损失召回。
- **高并发低延迟 / 小 topK、短查询**：用 `DAAT_WAND` 或 `BLOCK_MAX_WAND`，以 max-score 元数据剪枝换延迟。代价：剪枝会牺牲一部分召回。需要"严格按全局统计动态打分"（如平均文档长度变化）时用 `TAAT_NAIVE` 作基线。

自测脚本：同一份 BM25 语料，依次用 `DAAT_MAXSCORE` / `DAAT_WAND` / `BLOCK_MAX_WAND` 建索引并各测一轮延迟。前置条件：Milvus 3.0.x + pymilvus 3.0.x，且已把 `dataCoord.targetVecIndexVersion` 抬到 10（见 7.4.2.2）。

```python
from time import perf_counter
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530")
coll = "sparse_algo_bench"

# 建 BM25 全文检索集合：text 字段 + 由函数生成的稀疏字段
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=1000, enable_analyzer=True)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(
    name="text_bm25_emb", input_field_names=["text"],
    output_field_names=["sparse"], function_type=FunctionType.BM25,
))
client.create_collection(coll, schema=schema)
client.insert(coll, [{"text": d} for d in
                     ["Milvus 是一个向量数据库", "全文检索依赖倒排索引", "BM25 是经典打分算法"]])

# 三种算法依次建索引 + 检索；换算法需先释放再删旧索引
for i, algo in enumerate(["DAAT_MAXSCORE", "DAAT_WAND", "BLOCK_MAX_WAND"]):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_idx", metric_type="BM25",
        params={"inverted_index_algo": algo},
    )
    if i > 0:
        client.release_collection(coll)
        client.drop_index(coll, index_name="sparse_idx")
    client.create_index(coll, index_params=index_params)
    client.load_collection(coll)

    t0 = perf_counter()
    res = client.search(coll, data=["向量数据库"], anns_field="sparse",
                        limit=3, output_fields=["text"])
    dt = perf_counter() - t0
    print(f"{algo}: 延迟 {dt*1000:.2f} ms, top1={res[0][0]['entity']['text']}")
```

预期输出是三行"算法名 + 延迟 + 命中文本"。绝对数值取决于机器与数据量，这里只看"同一环境、三种算法"的相对高低；召回差异需自行用带标注的语料计算，未实跑的数字不在这里给出。

> ⚠️ **注意**：换算法要 `drop_index` 后重建，不能原地热切换；三组对比必须在同一集合、同一份数据上跑才有意义。



## 7.4.3 索引引擎直通模式

### 7.4.3.1 参数与最小可用样例

本节回答"怎么用一条引擎配方串在 Milvus 里直接建出经典索引"。它不讲 Knowhere 内部如何调度（见 Ch6 架构分层），只给一个最小可跑的建索引 + 检索样例。

3.0 新增了 `FAISS` 这一种索引类型：你把一条引擎配方串（index-factory string）交给 `faiss_index_name` 参数，Milvus 把它透传给 Knowhere 的 FAISS 适配器，由后者调用引擎原生工厂生成索引；检索参数同样透传。读者只需记住"一条配方串 + 对应检索参数"即可复用经典组合。

下面以 `IVF64,Flat`（64 个倒排列表 + 未压缩向量）为例。前置条件：Milvus 3.0.0+ 与 pymilvus 3.0.0+，且集合里已有密集向量字段。

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
collection_name = "faiss_float_example"

# 建索引：index_type 用 FAISS，配方串给 faiss_index_name
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_name="faiss_ivf_flat",
    index_type="FAISS",                      # 3.0 新增的直通索引类型
    metric_type="L2",
    params={"faiss_index_name": "IVF64,Flat"},  # 引擎配方串，唯一必需参数
)
client.create_index(collection_name=collection_name, index_params=index_params)
client.load_collection(collection_name=collection_name)
```

预期输出：`create_index` 返回后集合带 `faiss_ivf_flat` 索引，`load_collection` 成功即可检索。

检索参数放进 `search_params.params` 透传，IVF 工厂用 `nprobe` 控制扫几个倒排列表：

```python
search_params = {"params": {"nprobe": 8}}   # 扫 64 个倒排列表中的 8 个

results = client.search(
    collection_name=collection_name,
    data=[query_vector],                    # 与建索引同维的向量
    anns_field="vector",
    filter='category == "reference"',
    search_params=search_params,
    limit=5,
    output_fields=["category"],
)
```

预期输出：返回按 L2 距离排序的 top-5，且只含 `category == "reference"` 的行。

二值向量同理，配方串换成 `BFlat`、度量换成 `HAMMING`：

```python
index_params.add_index(
    field_name="binary_vector", index_name="faiss_binary_flat",
    index_type="FAISS", metric_type="HAMMING",
    params={"faiss_index_name": "BFlat"},
)
```

常用配方串（官方 3.0.0 发布测试覆盖，非穷举）：`Flat`（精确）、`IVF64,Flat`（检索参数 nprobe）、`HNSW16,Flat`（efSearch）、`OPQ16,IVF64,PQ16x4`（组合）、`IVF64,PQ8x4,RFlat`（PQ + 精排，nprobe / k_factor）、`BFlat`（二值精确）。

![FAISS 直通流程：配方串 → Knowhere 适配器 → 原生 Faiss 索引](/images/milvus30_faiss_index_flow.png)
> 图源：Milvus 官方文档（v3.0.x）



### 7.4.3.2 适用场景与代价

本节回答"什么情况该用直通、什么情况不该"。适合的场景很窄：你已经有一条在引擎侧验证过的配方串，想原样复现；或需要把经典组合（如 OPQ + IVF + PQ）做参数化实验。不适合的场景：大多数日常检索——有专属索引类型（HNSW / IVF / DiskANN）就优先用专属类型，因为它有稳定、文档化的参数契约。

代价是拿"可复现"换"上层能力"，以下均为官方口径：

> ⚠️ **注意**：
> - 只支持 `FLOAT_VECTOR` / `BINARY_VECTOR`；不支持 FLOAT16 / BFLOAT16 / INT8 / 稀疏向量。
> - 适配器跑在 CPU，不是 GPU 索引类型。
> - 标量过滤要求底层引擎索引支持 ID 选择器：3.0.0 仅对 `Flat`、`IVF64,Flat`、`HNSW16,Flat` 覆盖过滤测试；`PQ8x4` 会拒绝选择器、过滤检索失败，二值 `FAISS` 未验证标量过滤。
> - 不支持 `search_iterator()`，不提供原始向量回读；范围检索取决于工厂，二值 `FAISS` 不要用范围检索。
> - 配方串能被引擎接受 ≠ Milvus 一定支持，取决于字段类型 / 度量 / 维度 / 镜像编译的引擎模块。

一句话总结：直通是"专家级逃生舱"，用它复现配方；用得上 Milvus 上层优化（自动选型、通用过滤、迭代器）的，回到专属索引类型。



## 7.4.4 磁盘上的多向量检索

### 7.4.4.1 多向量字段的新数据类型

本节回答"磁盘上的多向量检索是什么、怎么把一条含向量列表的记录写进去"。它不重复 StructArray 的语法教学（见 7.3.2），只讲"向量列表成员 + 磁盘索引（DiskANN）"这一组合。

向量列表（embedding list）指一行记录里存多个向量——典型如 ColBERT 把文档拆成 token 嵌入、ColPali 把页面拆成 patch 嵌入。检索时查询本身也是一个向量列表，用 `MAX_SIM*` 度量做实体级"晚交互"打分（取每个查询向量与文档向量最相似值之和）。3.0 把这种多向量检索放到了磁盘索引上：给 StructArray 里的向量子字段建 DiskANN 索引，数据量超过内存时也能检索。

写入一条含向量列表的记录，关键是字段类型用 `ARRAY` + `element_type=STRUCT`，结构体里放向量子字段：

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)

# 结构体子字段：一段文本 + 它的向量
struct_schema = client.create_struct_field_schema()
struct_schema.add_field("text", DataType.VARCHAR, max_length=65535)
struct_schema.add_field("emb", DataType.FLOAT_VECTOR, dim=512)

# 向量列表字段：ARRAY of STRUCT，max_capacity 是每行最多成员数
schema.add_field("paragraphs", DataType.ARRAY,
                 element_type=DataType.STRUCT,
                 struct_schema=struct_schema, max_capacity=200)

client.create_collection("wiki_documents", schema=schema)
```

预期输出：集合建好，`paragraphs` 字段每行可容纳最多 200 个 `{text, emb}` 成员。

接着在向量子字段上建磁盘索引，度量用 `MAX_SIM*`：

```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="paragraphs[emb]",   # 索引目标是结构体的向量子字段
    index_type="DISKANN",           # 磁盘索引
    metric_type="MAX_SIM_COSINE",
)
client.create_index("wiki_documents", index_params=index_params)
client.load_collection("wiki_documents")
```

写入并检索一条记录（查询用 `EmbeddingList` 对象承载向量列表，示例向量为占位值，替换成你的真实嵌入）：

```python
from pymilvus.client.embedding_list import EmbeddingList

client.insert("wiki_documents", [{
    "title": "向量数据库",
    "paragraphs": [
        {"text": "Milvus 支持多向量检索", "emb": [0.11] * 512},
        {"text": "磁盘索引用 DiskANN", "emb": [0.22] * 512},
    ],
}])

query = EmbeddingList()
query.add_batch([[0.10] * 512, [0.21] * 512])   # 查询也是向量列表
res = client.search("wiki_documents", data=[query],
                    anns_field="paragraphs[emb]", limit=3)
```

预期输出：按 `MAX_SIM_COSINE` 打分返回实体级 top-3（一行是一个实体，不是单个成员）。

> ⚠️ **注意**：一个向量子字段只能建一个索引；要同一子字段既做 EmbeddingList 搜索（`MAX_SIM*`）又做 element-level 搜索（`COSINE`/`IP`/`L2`），必须拆成两个独立向量子字段。



### 7.4.4.2 搜索策略选型

本节回答"向量列表搜索该选 tokenann / muvera / lemur 哪个策略"。它不做三个策略之外的横评，只按"精度—延迟—内存"三角给判断，并配典型配置片段。

先理清一件事：`emb_list_strategy` 是"建索引 + 第一阶段候选召回"的策略，决定第一阶段近似候选怎么产生；最终结果仍由 MaxSim 精排（`emb_list_rerank`）给出。三种策略分点如下。

- **`tokenann`（默认）**：把列表里的每个向量都建进索引，查询时逐个向量做 ANN 召回后聚合回所属行，再用 MaxSim 精排。定位：质量优先，最接近原始 MaxSim，无压缩损失。代价：索引更大、候选召回更贵。适合：短/中等长度的向量列表、强判别嵌入、以质量为基准的场景；不适合超长文档、上千 patch 的视觉页面、严格内存/延迟预算。
- **`muvera`**：用随机投影把每条向量列表压缩成一个固定维 FDE 向量，第一阶段退化成普通行级向量检索，再 MaxSim 精排。定位：tokenann 太重、又不想做训练时的中间态。关键参数 `muvera_num_projections`（默认 4）、`muvera_num_repeats`（默认 7）、`muvera_seed`（默认 42）。代价：随机投影引入近似损失，FDE 维度影响延迟。适合：长文档、高判别嵌入、想比 tokenann 更省索引。
- **`lemur`**：训练一个模型把每条向量列表学成固定维行向量，第一阶段用学习到的行向量检索，再 MaxSim 精排。定位：值得花训练成本换压缩质量。关键参数 `lemur_hidden_dim`（默认 256）、`lemur_num_train_samples`（默认 20000）等。代价：需要训练、对语料分布和文档长度偏差敏感。适合：低判别嵌入、多模态/视觉文档、大列表；不适合语料频繁变动。

检索侧默认（官方口径）：`retrieval_ann_ratio=3.0`（第一阶段多召回的倍数）、`emb_list_rerank=true`（开启 MaxSim 精排）。

典型配置片段（把策略写进建索引的 `params`）：

```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="paragraphs[emb]",
    index_type="DISKANN",
    metric_type="MAX_SIM_COSINE",
    params={"emb_list_strategy": "muvera",      # tokenann / muvera / lemur
            "muvera_num_projections": 4},
)
```

选型一句话：质量优先且列表不长用 `tokenann`；列表长、想省索引又不想训练用 `muvera`；低判别/多模态且能接受训练用 `lemur`。三者间的召回差异与内存量级，需在你自己的数据上实测（本教程未复测，不给出具体数值）。



## 7.4.5 索引能力其余增量

### 7.4.5.1 GPU 索引与镜像变化

本节说清 3.0 里 GPU 索引"变"与"不变"的边界。它不重讲 GPU 索引原理（GPU_CAGRA / GPU_IVF_FLAT / GPU_IVF_PQ / GPU_BRUTE_FORCE 的取舍见 2.3.3），只讲镜像这一处变化。

3.0 的 GPU 索引用法本身不变，变的是镜像底座（官方口径）：

- **CUDA 12.9**：GPU 镜像迁至 CUDA 12.9。
- **不再保留 Ubuntu 20.04**：旧基线的 GPU 兼容不再维护。

> ⚠️ **注意**：若你在 GPU 节点上靠旧镜像或 Ubuntu 20.04 环境运行，升级 3.0 前需先确认宿主驱动/运行时与新 CUDA 版本匹配；GPU 索引类型与内存池配置（`gpu.initMemSize` / `gpu.maxMemSize`）不需要改。



### 7.4.5.2 标量索引版本提升

本节回答"标量字段的新索引版本怎么开"。它只讲一个开关，不展开标量索引类型本身（BITMAP 用于低基数、INVERTED 用于高基数，见 2.3.3）。

官方口径：标量索引的新版本是"选择性启用（opt-in）"——要把 `dataCoord.targetScalarIndexVersion` 抬到 `4` 才会生效，更晚的版本才会默认开启。

> 📌 **前置条件**：
> - 配置位置：`dataCoord` 段下的配置项（与 7.4.2.2 的 `targetVecIndexVersion` 同层，写在 Milvus 配置文件里）。
> - 改完是否要重启 DataCoord、能否热更新，官方缓存未明确写，标"待验证"。
> - 版本只影响"新建"的标量索引；已有标量索引需 `drop_index` 后重建才会走新版本。
> - 不要假设默认已开：不抬版本时不报错，标量索引仍走旧版本。



## 7.4.6 写入引擎 Woodpecker

### 7.4.6.1 它替换了什么、为什么

本节回答"写入引擎从什么换成了什么、为什么值得换"。它不重讲整体架构分层（见 Ch6），只讲写入链路这一处变化。

> ⛳ **架构演进提示**：整体架构分层与节点职责见 Ch6《Milvus 底层架构详解》，这里只聚焦"写前日志（WAL）由谁承担"这一处。

变化分两点：

- **旧（2.6 及以前）**：写前日志由外部消息队列承担——集群默认 Pulsar、Standalone 默认 RocksMQ，可换 Kafka。要额外部署、运维一套独立的消息中间件。
- **新（3.x）**：由 Milvus 自研的 Woodpecker 承担。它是面向对象存储的云原生 WAL，零磁盘架构——日志数据落在对象存储（MinIO / S3 / GCS 等），元数据放 etcd，核心链路不依赖本地磁盘。

读者视角的收益：**少一个外部依赖、运维面缩小**。3.x 里 Woodpecker 是默认 WAL，Standalone 默认嵌入运行（`minio` 或 `local`），无需额外起消息服务；集群可用嵌入式，也可把 Woodpecker 作为独立服务部署（见 7.4.6.3）。

> ⚠️ **注意**：默认是嵌入模式；"独立服务"形态仅在集群（distributed）可用，Standalone 只能嵌入。旧的 Pulsar / Kafka / RocksMQ 仍受支持，只是不再是默认。



### 7.4.6.2 从消息队列切换到 Woodpecker

本节回答"已有一台 Milvus，怎么把消息队列切到 Woodpecker"。它只覆盖 Woodpecker 与其他消息队列之间的切换（Pulsar↔Kafka 互切不在范围内），分 Standalone 与集群两种给步骤，并给回退与一致性注意点。

> 📌 **前置条件**：Switch MQ 是 3.0 才有的能力，先升级到 3.0+ 再操作。官方口径是"在线切换、无需停机（online, without downtime）"。但生产切换务必先在测试集群演练一遍。

**Standalone（Docker Compose，RocksMQ → Woodpecker）**：

1. 确认实例正常（建集合、插数据、查询各跑一遍）。预期：服务可用。
2. 先配 Woodpecker 存储、但不改 `mqType`——进容器改 `/milvus/configs/user.yaml`：

```yaml
woodpecker:
  storage:
    type: minio   # minio 或 local
```

   然后 `docker compose restart`。预期：重启后仍是 RocksMQ，尚未切换。

3. 触发切换：

```shell
curl -X POST http://<mixcoord_addr>:9091/management/wal/alter \
  -H "Content-Type: application/json" \
  -d '{"target_wal_name": "woodpecker"}'
```

4. 验证：`docker logs milvus-standalone | grep "successfully updated mq.type configuration in etcd"`，成功会打 `[mqTypeValue=woodpecker]`。

**集群（Helm，Pulsar → Woodpecker）**：

1. 确认集群正常。
2. 暴露 MixCoord 管理口并触发切换：

```shell
kubectl port-forward --address 0.0.0.0 service/my-release-milvus-mixcoord 29091:9091
curl -X POST http://127.0.0.1:29091/management/wal/alter \
  -H "Content-Type: application/json" \
  -d '{"target_wal_name": "woodpecker"}'
```

3. 验证：`kubectl logs <mixcoord-pod> | grep "successfully updated mq.type configuration in etcd"`，成功打 `[mqTypeValue=woodpecker]`。
4. 停内置 Pulsar、启用 Woodpecker（`helm upgrade` 设 `pulsarv3.enabled=false`、`woodpecker.enabled=true`），再删 Pulsar 的 PVC。

回退：反方向用同样的 alter API 把 `target_wal_name` 换回 `rocksmq` / `pulsar` 即可；但**不要反复横跳**，每次切换前先清理上一轮残留数据（Woodpecker 的 etcd 前缀 `woodpecker/` 与对象存储 `files/wp/...`）。

> ⚠️ **注意**（数据一致性）：官方原文未写明"切换时旧 MQ 里尚未消费的数据是否回放"——标"待验证"，生产环境务必先在测试集群验证数据完整后再上生产。



### 7.4.6.3 独立部署与可观测

本节回答"把 Woodpecker 当独立服务跑起来意味着什么、怎么看它的状态"。写路径原理只在 7.4.6.1 讲过一次，这里不重述。

独立服务（service 模式）意味着把 WAL 从 Milvus 节点里拆出来、作为单独一组 Pod 运行，带来三件事：

- **资源隔离**：WAL 的 CPU/内存/IO 不再和 DataNode、QueryNode 抢。
- **单独扩缩**：写压力大时只扩 Woodpecker，不用动查询面。
- **故障定位**：写链路上的问题能直接落到 WAL 服务本身。

官方口径：service 模式带来毫秒级写延迟，写路径不必经 broker 转发，跨可用区时相比 broker 型日志约省 1/3 写入、2/3 读取的跨区网络流量。

> 📌 **前置条件**：service 模式仅在集群（distributed）可用，Standalone 只能嵌入（`minio` 或 `local`）。官方建议 Milvus 3.0.3+ 搭配 Woodpecker v0.1.46+。Helm 部署时是独立 StatefulSet（默认 4 副本）+ headless service + gossip 集群：

```shell
helm install my-release zilliztech/milvus \
  --set streaming.woodpecker.embedded=false
```

预期：多出 `my-release-milvus-woodpecker` 这一组独立 Pod，与 Milvus 计算节点分离。

可观测：官方缓存未列出成体系的指标清单（标"待验证"），目前文档化的位点有两个：

- 切换/配置生效看日志：`kubectl logs <mixcoord-pod> | grep "successfully updated mq.type configuration in etcd"`。
- 吞吐/延迟调优看配置段：`woodpecker.logstore.*`（如 `segmentSyncPolicy.maxFlushSize`、`maxFlushThreads`）与 `woodpecker.client.*` 是文档化的吞吐/延迟旋钮；具体 metric 名称需以你部署版本实际暴露的为准。



## 7.4.7 段合并与治理

### 7.4.7.1 合并策略与碎段问题

本节回答"碎段从哪里来、带来什么后果、怎么发现它"。它不重讲 Ch6 的 mmap/存储优化，只写"合并"这一条链路。

碎段（fragmented segments）的来源：标准合并（standard compaction）按 m→1 把多个小段合成接近 `dataCoord.segment.maxSize`（默认 512 MB）的大段，但"接近"不等于"能继续合"——两个已经 60% 满的段再合就超限，于是永久卡在中等大小，反复合并也消不掉。

碎段带来两个后果：

- **查询延迟抖动**：检索要在更多段上各跑一遍再归并，段越多单次查询开销越大、越不稳定。
- **存储膨胀**：段元数据、索引头等固定开销随段数线性增加，同样的数据占更多存储。

观测方法（官方页给出的入口）：用 `client.get_compaction_state(job_id)` 查合并进度，用 `list_persistent_segments` 数段数，判断是否已经碎得需要主动治理。

> ⚠️ **注意**：标准合并是尽力而为的增量清理，处理不了"中等段卡住"的问题；要主动治理碎段，用 7.4.7.2 的强制合并。



### 7.4.7.2 强制合并（异步）

本节回答"怎么主动把碎段合并成大段"。它只演示官方提供的那一个入口——异步 `compact(target_size=...)`——并说清什么时候该手动触发、什么时候绝对不要用。

> ⚠️ **注意**：强制合并（Force Merge）目前是 **public preview**，官方原文明确 **Do not use force merge compaction in production environments**——生产环境不要用，只在测试环境或低峰期窗口使用。

官方 API 只有异步接口，没有同步版：`compact()` 返回一个 job id，用 `get_compaction_state()` 轮询。释放说明里虽提到 sync/async 两种模式，但公开 API 里没有 sync 参数——按官方操作页的异步写法即可，不要编造同步参数。

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# 显式目标：把段合并到约 2 GB 每个（单位 MB，须 >= dataCoord.segment.maxSize）
job_id = client.compact("target_collection", target_size="2048")

# 异步轮询进度
state = client.get_compaction_state(job_id)
print(f"State: {state}")
```

`target_size` 三种取值：`0`/省略（走标准合并，用 `dataCoord.segment.maxSize`，默认 512 MB）、显式值（≥ maxSize，否则报错拒绝）、`max_int64 = (1<<63)-1`（让 Milvus 按段分布与节点内存自动算最优大小，官方推荐大多数场景用这个）。

何时手动触发：**批量导入后**（一次灌入大量小段）、**索引换代后**（7.4.2 换了索引版本/算法想重排段）。触发前想清楚代价——强制合并是重资源操作：读、并、重写整段数据，大集合耗时与资源占用需在你的环境实测（未实测，不给出具体数值），务必挑低峰期。

```python
max_int64 = (1 << 63) - 1
job_id = client.compact("target_collection", target_size=max_int64)
```

预期输出：返回 job id；随后 `get_compaction_state` 显示进度，完成后段数变少、单段变大（用 `list_persistent_segments` 前后对比）。

![强制合并把小碎段重组为更少、更大的段](/images/milvus30_compaction.png)
> 图源：Milvus 官方文档（v3.0.x）



## 7.4.8 本章小结

**本章核心掌握：**

1. **稀疏索引换代（7.4.2）**：3.0 把稀疏索引从"单一倒排"升级为可按负载选算法（`inverted_index_algo` 六种取值），新算法 opt-in——要把 `dataCoord.targetVecIndexVersion` 抬到 10、重建索引后才生效。
2. **索引引擎直通（7.4.3）**：`FAISS` 索引类型 + 一条 `faiss_index_name` 配方串即可复现经典组合；代价是只支持 FLOAT/BINARY、CPU、部分工厂不支持标量过滤。
3. **磁盘多向量检索与写入引擎（7.4.4 / 7.4.6）**：向量列表（embedding list）+ DiskANN + `MAX_SIM*` 做实体级多向量检索；写前日志由外部消息队列换成自研 Woodpecker（零磁盘架构），少一个外部依赖。
4. **段合并治理（7.4.7）**：碎段让查询延迟抖动、存储膨胀，用异步 `compact(target_size=...)` 强制合并收口——但它是 public preview，生产环境不要用。

掌握了这四条，就掌握了 3.0 在"索引与写入"这条链路上的增量。7.5 把本章与前三章的检索能力串成一条完整的端到端链路，是动手实践的入口。


