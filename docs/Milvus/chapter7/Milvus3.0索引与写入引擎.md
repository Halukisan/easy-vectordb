# Chapter 7.4 Milvus 3.0 索引、写入与存储引擎（进阶）

## 7.4.0 这一章给谁看

上一章是「用法」，这一章是「原理与开关」：稀疏索引换了什么算法、FAISS 配方怎么直通、写入路径为什么不再需要 Pulsar/Kafka、段碎片怎么削平，最后一节拆开 Storage V3——外部集合和快照的底座。用不上可以不读——各节会指出适用面和官方限制，遇到稀疏检索性能问题、大语料稀疏索引、或者想搞懂 7.2 里那些反直觉行为时回来看。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.4.1 稀疏索引换代：SINDI、Block-Max WAND、Block-Max MaxScore

官方原话：「Milvus 3.0 对稀疏向量索引进行了全面升级。它引入了新的搜索算法——SINDI、Block-Max WAND 和 Block-Max MaxScore——同时还支持倒排列表压缩、可配置量化以及按工作负载选择搜索算法。」

关键数字（官方内部基准）：

> 压缩后的 BM25 索引在可比召回率下比 2.6 稀疏索引小约 3 倍，而 SINDI 在学习型稀疏 Embeddings 上达到的 QPS 最高可达 MaxScore 的 10 倍左右。

换算法怎么生效？需要手动提升目标索引版本，官方发布说明原文：「新引入的索引算法在生效前需要手动将目标索引版本提升（例如将 `dataCoord.targetVecIndexVersion` 设为 10，将 `dataCoord.targetScalarIndexVersion` 设为 4）；在后续版本中，这些功能将默认启用。」

```yaml
dataCoord:
  targetVecIndexVersion: 10
  targetScalarIndexVersion: 4
```

（配置通过 `user.yaml` 覆盖，修改方法见 7.1.2。）启用后的默认变化，官方原话：「启用新索引版本后，SINDI 将成为稀疏 IP 搜索的默认选项，而 MaxScore 将成为 BM25 的默认选项。」

适用面（官方表述）：「mmap 加载、序列化和 BM25 评分机制也得到了优化，从而降低了大规模稀疏向量和全文搜索的索引存储及加载开销。」——收益在大规模稀疏索引上最明显。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.4.2 FAISS 直通：把现成配方搬进来

「FAISS 索引类型是 Milvus 3.0.0 及更高版本中提供的一种专家级直通功能。它允许您提供一个 Faiss 索引生成器字符串，而非选择固定的 Milvus 索引类型。」——你手里有调好的 Faiss 配方，原样搬进来。

适用面要看清：「上游 Faiss 接受的工厂字符串并不一定被 Milvus 自动支持。」兼容性取决于向量字段类型、度量、维度、编译进镜像的 Faiss 模块，以及索引是否支持 Milvus 所需的操作。官方也提醒：常见配方优先用专用索引类型，「因为它具有稳定且有文档记录的参数契约」。

硬性限制：只支持 `FLOAT_VECTOR` 和 `BINARY_VECTOR`（不支持 FLOAT16/BFLOAT16/INT8/SPARSE）；「通用 FAISS 适配器在 CPU 上运行。它不是 Faiss GPU 索引类型」；必须指定 `faiss_index_name` 构建参数。

**参考文档**：[FAISS 索引类型](https://milvus.io/docs/zh/faiss.md)

## 7.4.3 EmbList + DiskANN：嵌入列表的磁盘索引

这是 7.3.2 StructArray 的索引侧配套。官方描述：「EmbList + DISKANN——针对 StructArray 嵌入列表的磁盘上多向量索引，包含 Muvera 和 Lemur 等加速路径。」设计动机：「ColBERT 等晚期交互模型会为每个令牌生成一个向量，而多模态实体可能包含多种视图。EmbList 为每个实体存储一个可变长度的向量列表，并以 DISKANN 作为磁盘索引。当语料库超过内存预算时，磁盘路径可有效控制 RAM 占用量。」

怎么建？在 StructArray 的向量子字段上建 `DISKANN` 索引，搜索走 MAX_SIM（见 7.3.2 引擎依赖关系）。Muvera / Lemur 是同一条官方描述里列出的加速路径，具体用法以当前版本的官方支持说明为准。

什么时候值得换：ColBERT/ColPali 类晚期交互检索、向量总量超出内存。

**参考文档**：[使用 Embeddings 列表进行搜索](https://milvus.io/docs/zh/search-with-embedding-lists.md)

## 7.4.4 存储层新格式：Vortex 与 Lance

官方 3.0.0 原文：

> 存储层新增了两种开放的列式格式：Vortex 作为下一代内部格式——支持自适应编码（字典编码、RLE、位打包、浮点数专用压缩）、零拷贝解压，并针对向量与标量混合工作负载进行了优化；此外，Lance 与 Parquet 并行，用于开放生态系统的数据交换。Vortex 将成为默认的内部格式，其路线图中还包含过滤器下推和本地变体功能。

两条线要区分：**Vortex 是 Milvus 内部存储的新格式**（官方口径：「Vortex 将成为默认的内部格式」，TEXT 的 LOB 文件就用它）；**Lance 是开放交换格式**，和 Parquet 并列——外部 Collection 可以直接引用 Lance 表（7.2.1）。作为使用者你基本不用手动选，知道「Vortex 在内部、Lance 在交换」就够了。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.4.5 写入引擎 Woodpecker：不再需要 Pulsar/Kafka

「在 Milvus 3.x 中，Woodpecker 是默认的 WAL/消息队列，作为日志服务提供有序写入和恢复功能。无需外部消息队列服务（如 Pulsar 或 Kafka）。」

两种运行形态：「可以嵌入 Milvus/流处理节点中运行（默认），也可以作为拥有独立 Pod 的专用服务运行（仅限分布式/集群环境）。」三种存储模式：「对象存储（minio，默认）、本地文件系统（local）以及专用 service」。

想显式指定：

```yaml
mq:
  type: woodpecker
```

⚠ 官方警告：「对正在运行的集群切换 mq.type 属于升级操作。请仔细遵循升级流程，并在切换至生产环境前，先在全新集群上进行验证。」

什么时候值得换：官方说「无需外部消息队列服务（如 Pulsar 或 Kafka）」——省一套 MQ 的部署与运维。

**参考文档**：[Woodpecker](https://milvus.io/docs/zh/woodpecker.md)

## 7.4.6 强制合并：削平段碎片

「强制合并的目的是将小而零散的数据段合并为较少而较大的数据段，以提高查询性能和存储效率。」它解决标准压缩的遗留问题：标准压缩按「多对一合并」把段压到 `maxSize` 附近，但会留下无法再合并的中等碎片——比上限略小、两两合并又都超限的那种。

用法要点：`target_size` 参数指定目标段大小（0 或省略时取 `dataCoord.segment.maxSize`，默认 512 MB）；支持同步和异步两种模式，适合在非高峰时段显式触发。两个算法细节：段数不超过 `dataCoord.compaction.maxFullSegmentThreshold`（默认 100）时用标准算法（更优但更慢），超过时用贪婪算法（更快，分组略差）。

⚠ 官方标注：此功能处于公开预览阶段，「请勿在生产环境中使用」。

**参考文档**：[强制合并压缩](https://milvus.io/docs/zh/force-merge.md)

## 7.4.7 外部集合与快照的底层：Storage V3 清单

7.2 里那些「反直觉」的行为——刷新后主键全变、两份文件不去重、不 flush 的快照恢复出 0 行——根子都在这一节。三个问题一次讲清：外部集合刷新时到底做了什么、快照为什么便宜、milvus-table 是什么。

**底座：不可变清单。** 官方《存储 V3》原文：

> 每个数据集版本都被封装为不可变的清单快照——这是一个 Avro 编码的文件，记录了构成该数据集的列组、增量日志和统计信息。清单是紧凑的 Avro 文件，增量日志记录实体级别的删除操作，而无需重写数据文件。这使得随着数据集的增长，元数据开销得以保持在较低水平。
>
> 由于状态存储在对象存储中，数据集具有自描述性：任何能够访问存储路径的读取者，无需中央目录即可发现并解析数据集。这一特性为外部 Collection、快照以及未来的湖存储集成提供了基础。
>
> Storage V3 使用清单文件来描述 Collection 数据，使其独立于底层数据格式。这使得同一存储层既能处理由 Milvus 管理的数据，也能处理仍保存在外部系统中的数据。

两个关键词：**不可变**（每次变化写新清单，旧清单还在，天然就是版本）、**自描述**（清单里记着数据在哪、怎么解码，不依赖 Milvus 目录服务）。

**外部集合的刷新，是「建视图」不是「导入」。** 刷新时 Milvus 做三件事：

1. 扫描 `external_source` 前缀下的文件，采样每个文件的 schema（这也是「同前缀下文件 schema 必须一致」这条限制的来源——采样时发现缺列，整个作业失败）；
2. 生成清单，记录「Milvus 字段 → 外部文件行」的映射；
3. 把清单注册成外部分段，查询节点按清单去湖上读。

所以刷新天然是**文件级、全量**的：两个文件就是两份行集，没有跨文件的主键合并；刷新重建清单，`__virtual_pk__` 是这次清单里的行号，自然每次都不一样。也因为没有写路径（外部文件归湖管，Milvus 无权改），外部集合只读。

**快照为什么便宜。** 「拍快照」实际做的事是写一个新的清单文件引用现有分段，不复制数据本身。恢复走的是另一条路——官方对恢复机制的说明：

> 恢复过程采用分段复制机制而非数据导入……它会直接从快照存储中复制分段文件（二进制日志、增量日志、索引文件），保留字段 ID 和索引 ID，以确保与现有数据文件的兼容性，避免了数据重写和索引重建，从而显著缩短了恢复时间，并且与传统的备份和恢复方法相比，性能提升 10 到 100 倍。

关键点：清单只能引用**已经落盘成独立文件的分段**。还在 growing segment 里的数据没有独立文件可引用，快照就拍不到它——7.2.8 里「不 flush 恢复出 0 行」就是这么来的。这不是 bug，是清单机制的直接后果。

**外部集合也能拍快照。** 实测对 `products_ext` 拍快照（`snap_ext`）成功。原因同上：拍的是清单，清单本来指向的就是湖上的文件——所以外部表不用搬数据，也能秒级拍、分钟级恢复。

一个使用上的注意（官方《存储 V3》原文，开关见 7.1.2）：Loon 开启后新旧两种布局并存——「新的写入操作和压缩输出将使用 Storage V3。现有数据将保留在当前布局中，直到符合条件的数据被后台压缩重写为止。在此过渡期间，Milvus 可以读取两种布局。」

**milvus-table：把快照变成外部表。** 3.0.0 新增的 `milvus-table` 格式，把「快照」和「外部表」接了起来。官方原文：

> 此版本还新增了 milvus-table 外部格式，该格式将 Milvus 快照元数据和 Storage V3 清单视为外部数据源，因此 Collection 快照本身可作为外部表提供服务——批处理和服务系统可获得基于同一数据清单的共享视图。

官方《快照用例》页给了完整流程，并有一条必须遵守的警告：

> 对于 Milvus 快照，请使用 milvus-table 外部 Collection 格式，而非直接将快照文件作为通用 Spark 输入进行读取。

原因是「Milvus 快照存储了 Collection 元数据、分段清单、删除日志和主键统计信息，因此 Milvus 需要快照元数据 JSON 以及 milvus-table 读取器，以保留正确的 Schema 和删除语义」。快照给批处理用，走的是「注册成外部 Collection」这条路，不是把文件路径直接丢给 Spark。

流程两步（代码来自官方原文，Python）：

**第 1 步：拿到快照的存储位置。** `describe_snapshot` 带上 `include_collection_info=True`，返回里有 `s3_location`：

```python
snapshot_info = client.describe_snapshot(
    snapshot_name="analytics_snapshot_20260321",
    collection_name="my_collection",
    include_collection_info=True,
)
external_source = f"s3://bucket/{snapshot_info.s3_location}"
```

**第 2 步：按 milvus-table 格式建外部 Collection 并刷新。** `external_spec.format` 设为 `"milvus-table"`，字段用 `external_field` 映射：

```python
schema = client.create_schema(
    external_source=external_source,
    external_spec='{"format": "milvus-table", "extfs": {...}}',
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, external_field="id")
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768, external_field="embedding")
client.create_collection(collection_name="snapshot_external_collection", schema=schema)
job_id = client.refresh_external_collection(collection_name="snapshot_external_collection")
```

官方对机制的说明：「此工作流会在快照数据上创建一个可查询的外部 Collection。主列数据仍从快照源进行引用，而刷新操作会将源 Storage V3 清单映射到目标外部分段。」刷新完成后建索引、load，就能对快照视图搜索查询。

**参考文档**：[存储 V3](https://milvus.io/docs/zh/storage-v3.md) · [快照用例](https://milvus.io/docs/zh/snapshot-use-cases.md) · [创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md) · [Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.4.8 小结

- 稀疏索引换代三件套：SINDI、Block-Max WAND、Block-Max MaxScore；BM25 索引小约 3 倍，SINDI 学习型稀疏 QPS 可达 MaxScore 的 10 倍；靠 7.1.2 的索引版本开关启用，SINDI/MaxScore 成为各自领域的默认；
- FAISS 直通是专家级功能：给生成器字符串，只支持 FLOAT/BINARY，CPU，必须 `faiss_index_name`；常规场景用专用索引类型；
- EmbList + DiskANN：ColBERT/ColPali 的磁盘索引方案，RAM 预算紧时用；
- 存储层新增 Vortex（内部格式）与 Lance（开放交换格式），Vortex 将成为默认内部格式；
- Woodpecker 是 3.x 默认 WAL，不再需要 Pulsar/Kafka；`mq.type: woodpecker` 显式指定，切换属于升级操作；
- 强制合并削段碎片，`target_size` 调目标，当前是预览阶段功能，别上生产；
- 外部集合与快照的底座都是 Storage V3 的不可变清单：刷新生成「Milvus 字段 → 外部文件」映射，快照写新清单引用旧分段；清单只能引用已落盘分段，所以快照前必须 flush。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md) · [FAISS 索引类型](https://milvus.io/docs/zh/faiss.md) · [Woodpecker](https://milvus.io/docs/zh/woodpecker.md) · [强制合并压缩](https://milvus.io/docs/zh/force-merge.md)

下一章把所有东西串起来：从一份 Parquet 文件到快照、重排，完整演一遍。

> **下一章**：[7.5　Milvus 3.0 端到端实战](./Milvus3.0端到端实战)
