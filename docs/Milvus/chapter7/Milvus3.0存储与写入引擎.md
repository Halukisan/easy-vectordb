# Chapter 7.4 深入 · 存储与写入引擎

> 这是 7.4《索引与性能选择》的底层延伸：那一篇讲「三类新索引各管什么、怎么开、怎么选」，这一篇往下走一层——存储格式、写入引擎、段合并，以及外部集合和快照共用的底座 Storage V3 清单。结论逐条给官方出处；标注测试版/预览的功能会明确提醒。

## 一、存储层新格式：Vortex 与 Lance

官方 3.0.0 原文：

> 存储层新增了两种开放的列式格式：Vortex 作为下一代内部格式——支持自适应编码（字典编码、RLE、位打包、浮点数专用压缩）、零拷贝解压，并针对向量与标量混合工作负载进行了优化；此外，Lance 与 Parquet 并行，用于开放生态系统的数据交换。Vortex 将成为默认的内部格式，其路线图中还包含过滤器下推和本地变体功能。

两条线要区分：**Vortex 是 Milvus 内部存储的新格式**（官方口径：「Vortex 将成为默认的内部格式」，TEXT 的 LOB 文件就用它）；**Lance 是开放交换格式**，和 Parquet 并列——外部 Collection 可以直接引用 Lance 表（7.2.1）。作为使用者你基本不用手动选，知道「Vortex 在内部、Lance 在交换」就够了。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 二、写入引擎 Woodpecker：不再需要 Pulsar/Kafka

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

## 三、强制合并：削平段碎片

「强制合并的目的是将小而零散的数据段合并为较少而较大的数据段，以提高查询性能和存储效率。」它解决标准压缩的遗留问题：标准压缩按「多对一合并」把段压到 `maxSize` 附近，但会留下无法再合并的中等碎片——比上限略小、两两合并又都超限的那种。

用法要点：`target_size` 参数指定目标段大小（0 或省略时取 `dataCoord.segment.maxSize`，默认 512 MB）；支持同步和异步两种模式，适合在非高峰时段显式触发。两个算法细节：段数不超过 `dataCoord.compaction.maxFullSegmentThreshold`（默认 100）时用标准算法（更优但更慢），超过时用贪婪算法（更快，分组略差）。

⚠ 官方标注：此功能处于公开预览阶段，「请勿在生产环境中使用」。

**参考文档**：[强制合并压缩](https://milvus.io/docs/zh/force-merge.md)

## 四、外部集合与快照的底层：Storage V3 清单

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

## 五、小结

- 存储层新增 Vortex（内部格式）与 Lance（开放交换格式），Vortex 将成为默认内部格式；
- Woodpecker 是 3.x 默认 WAL，不再需要 Pulsar/Kafka；`mq.type: woodpecker` 显式指定，切换属于升级操作；
- 强制合并削段碎片，`target_size` 调目标，当前是预览阶段功能，别上生产；
- 外部集合与快照的底座都是 Storage V3 的不可变清单：刷新生成「Milvus 字段 → 外部文件」映射，快照写新清单引用旧分段；清单只能引用已落盘分段，所以快照前必须 flush。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md) · [Woodpecker](https://milvus.io/docs/zh/woodpecker.md) · [强制合并压缩](https://milvus.io/docs/zh/force-merge.md) · [存储 V3](https://milvus.io/docs/zh/storage-v3.md) · [快照用例](https://milvus.io/docs/zh/snapshot-use-cases.md) · [创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md)

> 返回：[7.4　索引与性能选择](./Milvus3.0索引与性能选择)
