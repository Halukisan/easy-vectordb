# Chapter 7.5 Milvus 3.0 端到端实战

## 7.5.0 要搭的东西

一句话：湖上一份商品 Parquet，原地建索引；集合随时拍快照；一次搜索请求里完成召回、过滤、统计。全程不把数据拷进 Milvus 的业务表，也不写一行 ETL 定时任务。

**最终效果**（按官方口径描述，具体数值以你的数据为准）：

1. `search("无线降噪耳机")` 直接在 Parquet 数据上返回结果——数据零拷贝；
2. 搜索附带 `filter='price < 500'`，过滤在服务端完成，不拉全量到客户端；
3. 再用 `group_by_fields` 在服务端聚合，拿到品牌分布（7.3.5 的聚合查询）；
4. 更新湖上文件后再 `refresh` 一次，新数据可查——刷新耗时官方口径 150-250 ms；
5. 任意时刻 `create_snapshot` 拍一份（毫秒级），`restore_snapshot` 恢复到新集合（官方口径几秒到几分钟），比传统备份恢复快 10-100 倍。

**环境**：v3.0.x、Storage V3 已开（开关见 7.1.2）、`pymilvus>=3.0.1`、一份可被 Milvus 访问的 S3/MinIO 路径（含 Parquet 文件）。

**参考文档**：[创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md) · [管理快照](https://milvus.io/docs/zh/manage-snapshots.md)

## 7.5.1 数据准备：湖上放一份 Parquet

准备一个 `products.parquet`，至少四列：`id`（int64）、`product_name`（string）、`price`（double）、`embedding`（768 维 float）。上传到 Milvus 能访问的 S3 桶。确认路径能列出文件：

```python
import pyarrow.dataset as ds
dataset = ds.dataset("s3://my-bucket/products/", format="parquet")
print(dataset.schema)
```

这一步和 Milvus 无关，是数据湖侧的准备。记住两个约束：**文件格式要能被引擎识别**（Parquet/Lance/Iceberg/Vortex，官方列的四种）；**列名与 Schema 映射要对上**（下一步的 `external_field` 负责）。

**参考文档**：[创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md)

## 7.5.2 搭建步骤

### 第一步：建外部 Collection（7.2.2～7.2.5 五步浓缩）

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

schema = MilvusClient.create_schema(
    external_source="s3://my-bucket/products/",
    external_spec='{"format": "parquet", "extfs": {"storage_type": "remote"}}',
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, external_field="id")
schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=512, external_field="product_name")
schema.add_field(field_name="price", datatype=DataType.DOUBLE, external_field="price")
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768, external_field="embedding")

client.create_collection(collection_name="catalog", schema=schema)
```

### 第二步：建索引并刷新

```python
index_params = client.prepare_index_params()
index_params.add_index(field_name="id", index_type="AUTOINDEX")
index_params.add_index(field_name="product_name", index_type="AUTOINDEX")
index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
client.create_index(collection_name="catalog", index_params=index_params)

client.refresh_external_collection(collection_name="catalog")
```

### 第三步：验证——搜索 + 过滤 + 统计

```python
client.load_collection(collection_name="catalog")

res = client.search(
    collection_name="catalog",
    data=[query_vec],                 # 768 维查询向量
    anns_field="embedding",
    search_params={"metric_type": "COSINE"},
    filter="price < 500",
    limit=10,
    output_fields=["product_name", "price"],
)
for hit in res[0]:
    print(hit["distance"], hit["entity"]["product_name"], hit["entity"]["price"])
```

### 第四步：拍快照、演练恢复

```python
client.flush(collection_name="catalog")           # 官方建议：拍前 flush
client.create_snapshot(collection_name="catalog", snapshot_name="catalog_base")

# … 模拟湖上文件被意外覆盖 …

job_id = client.restore_snapshot(
    snapshot_name="catalog_base",
    collection_name="catalog_restored",
)
while True:
    state = client.get_restore_snapshot_state(job_id=job_id)
    print(f"State: {state.state}, Progress: {state.progress} %")
    if state.progress >= 100:
        break
    time.sleep(2)
```

### 第五步（可选）：排序、聚合、重排

在同一个集合上试 7.3.5 的能力：`group_by` 聚合品牌分布、`ORDER BY` 服务端排序、`hybrid_search` 加重排。这些是纯检索侧的增强，不需要改数据。

**参考文档**：[创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md) · [管理快照](https://milvus.io/docs/zh/manage-snapshots.md)

## 7.5.3 日常运行：增量与演进

三个高频动作，各配一句官方约束：

**湖上数据变了 → 再刷一次。**「如果源数据发生更新，您需要手动再次调用刷新操作，以保持 Milvus 的数据同步。」150-250 ms 是官方给出的典型耗时。

**Schema 要加列 → 在线加。** 往外部集合加字段后「再次刷新外部 Collection」（官方《修改外部 Collection Schema》流程）；内部托管集合则用 `add_collection_field()`（7.3.1），注意新增用户定义字段必须 `nullable=True`。

**批处理要读 → 用 milvus-table。** 快照按 `milvus-table` 格式注册成外部 Collection（7.4.7），官方说「批处理和服务系统可获得基于同一数据清单的共享视图」。

**要回滚 → 用快照，不用时间机器。** `restore_snapshot` 的产出是一个新集合（官方参数就是 `collection_name`），不是把原集合原地退回——恢复后改业务代码指向新集合。

**参考文档**：[创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md) · [管理快照](https://milvus.io/docs/zh/manage-snapshots.md) · [快照用例](https://milvus.io/docs/zh/snapshot-use-cases.md)

## 7.5.4 结果观察与代价

跑完这套，用三组指标观察收益：

- **存储**：对象存储里只有索引和清单（快照边际空间近乎为零），没有第二份全量副本；
- **刷新延迟**：150-250 ms（官方口径）量级，只是扫描元数据生成清单，不重建索引；
- **查询**：`filter` 下推后只传回命中行；`group_by`/分面在服务端算完再返回统计量。

代价也要说清：外部数据是只读的（写入要回湖上或另建托管集合）；刷新不会自动发生；Storage V3 一旦启用不可回滚（7.1.2）。官方对外部 Collection 的定位是「作为数据湖之上的查询层」——数据主副本在湖仓、Milvus 负责检索，这是它设计的场景。

**参考文档**：[创建外部 Collection](https://milvus.io/docs/zh/create-an-external-collection.md) · [存储 V3](https://milvus.io/docs/zh/storage-v3.md) · [Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.5.5 小结

这一章把 7.1–7.4 串了一遍：

- 湖上 Parquet 原地建外部 Collection——零拷贝、增量刷新；
- 快照 = 时间点只读视图，恢复是分段复制，10-100 倍加速（官方口径）；
- Schema 在线演进、排序/聚合/分面/重排都在同一集合上叠加；
- 全链路依赖 Storage V3，开关与不可回滚约束见 7.1.2；
- 进阶原理（SINDI、FAISS 直通、Woodpecker、强制合并、Storage V3 清单）在 7.4，按需回看。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md) · [快照用例](https://milvus.io/docs/zh/snapshot-use-cases.md)

到这儿，Milvus 3.0 从「是什么」到「怎么用」已经闭环。剩下的事交给你的数据。

> **返回**：[序章：Milvus 3.0](./Milvus3.0)　·　**上一章**：[7.4　索引、写入与存储引擎（进阶）](./Milvus3.0索引与写入引擎)
