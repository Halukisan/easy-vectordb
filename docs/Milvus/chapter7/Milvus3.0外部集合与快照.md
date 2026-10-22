# Chapter 7.2 Milvus 3.0 湖仓原生：外部集合与快照

## 7.2.1 学习目标

本章聚焦 Milvus 3.0 的"湖仓原生"能力，回答三个问题：数据能不能不搬进库里、能不能随时按时间点看、schema 能不能在线演进。本章不重述集合/字段/索引/搜索这些基础概念——它们在 Chapter 2、Chapter 3 已经讲清，这里只讲 3.0 新增的能力。

学完本章，你能：

1. 说清 Storage V3 与外部集合、快照这两层新范式的因果关系（7.2.2）。
2. 独立完成一个外部集合的挂载、建索引、检索与增量追平（7.2.3）。
3. 用快照做误操作回滚与 A/B 版本对照，并说清快照与备份的分工（7.2.4）。
4. 在不停机前提下给已有集合加列、回填并删列（7.2.5）。


## 7.2.2 存储引擎 V3：新底座

### 7.2.2.1 为什么先讲底座

本节解决“为什么 3.0 的新能力要先从存储底座讲起”，不解决“具体怎么开 Loon”——那在 7.1.4.3，这里只讲两代存储模型各是什么样、为什么要换。

Milvus 2.6 的存储是“三线分置”的：集合 schema、节点状态、消费位点这类元数据放在 etcd；体量最大的索引文件与二进制日志（binlog）落在对象存储（MinIO / S3 兼容存储）；写入顺序由消息队列（Kafka / Pulsar / RocksMQ）保证。三条线各自维护一份“谁在哪、进展到哪”的记录，整体够稳；但当数据要“留在原地、不搬进库”时，它缺少一个能自描述、又能按版本演进的底座。

存储引擎 V3（Storage V3 / Loon）把这一点收敛了：数据与元数据统一落在 S3 兼容对象存储上，用一个 **manifest**（清单）文件当唯一真相。每份数据集版本就是一条不可变的 manifest 快照（Avro 编码），记录当前数据集由哪些列组（column group）、增量日志（delta log）与统计信息组成；删除只写实体级 delta log，不复写数据文件，元数据开销不随数据膨胀。manifest 把元数据跟踪与查询路径解耦，集合能管理更多段而不拖慢查询；又因为状态都落在对象存储上，数据集是自描述的——任何能访问该存储路径的读取方都能直接读懂它，不再依赖中心化 catalog。外部集合、快照这两块 3.0 的新能力，正是架在这个自描述底座上。文件格式默认仍是 `parquet`，`vortex` 是可选的新一代列式格式（字典 / RLE / bit-packing / 浮点压缩、零拷贝解压），官方建议先做基准测试再切换。

> 📌 **前置条件**：Storage V3（Loon）**默认关闭**，依赖它的能力需手工打开 `common.storage.useLoonFFI`——开启步骤见 7.1.4.3，本节不重复配置。

### 7.2.2.2 验证是否真用上了 V3

本节解决"怎么确认 Loon 真的生效了"，不解决"怎么开启"——开启步骤见 7.1.4.3，这里只给验证判据。

Storage V3 默认关闭，而且 `common.storage.useLoonFFI` 是**可刷新配置**：改静态配置文件不等于运行中的实例已收到新值，必须走部署支持的配置更新流程。更隐蔽的是，开启前已有的数据只会在后台 compaction 逐渐迁到新布局，不会立即整体转换。所以"配置里写了 true"和"数据真的落在 V3 布局上"是两件事，必须靠观察产物来确认。

官方给出的开启片段如下（storage-v3.md）：

```yaml
common:
  storage:
    useLoonFFI: true
```

配置生效后，**新写入与 compaction 产物**改用 Storage V3 布局。验证分两步：

1. **触发新写入并 flush**：建一个集合、插入若干实体、调用 `flush()`，让这批数据落到对象存储。
2. **去对象存储里看目录**：V3 生效的判据是——存储路径里出现以 **manifest（清单）** 为核心的产物：Avro 编码的清单文件，记录该数据集由哪些列组（column group）、增量日志（delta log）与统计信息组成；旁边是按配置格式（默认 `parquet`）写的列式数据文件；删除只写实体级 delta log，不复写数据文件。如果看到的仍是 2.x 那套 binlog 目录、没有任何 manifest 文件，说明 Loon 并未生效。

> ⚠️ **注意**：数据文件格式由 `dataNode.storage.format` 决定，默认 `parquet`、可选 `vortex`，改它只影响新的 V3 写入，旧文件要等 compaction 重写才换格式。"manifest / 列组 / delta log / Avro"都出自官方对 Storage V3 布局的描述；但对象存储里这些文件的**具体目录命名与扩展名**我未实跑核对（待验证），请以你自己实例的产物为准。


## 7.2.3 外部集合（External Collection）

### 7.2.3.1 三个关键属性与适用场景

本节解决"外部集合到底是什么、什么时候该用它"，不解决"怎么写第一行代码"——那在 7.2.3.3。

外部集合（External Collection）在官方文档里的定位是一句话：让 Milvus 以查询层的方式访问外部存储里的数据（S3、Iceberg 等），**不把数据拷贝进 Milvus**，同时保持与 Milvus 查询接口兼容。它有三个关键属性，缺一不可：

- **原地读（zero-copy）**：Milvus 只维护 schema、索引与查询执行，数据文件留在原处。后台生成的 manifest 文件记录"Milvus 元数据 ↔ 外部文件里每一行"的映射，读取时直接扫源文件。
- **只读（read-only）**：外部集合没有写入接口，不能 insert / upsert / delete。官方 release notes 明确写 read-only，改数据只能回源湖仓改、再 refresh 追平。**记住这条，否则第一行就想 insert。**
- **靠增量 refresh 追平**：源数据变化后，手动触发亚秒级的 refresh 更新元数据，让 Milvus 始终对齐外部文件。

两张官方流程图把"搬数据"和"不搬数据"的差异画得很直白：

![外部集合之前：把数据搬进 Milvus 再算](/images/milvus30_external_collection_bring_data_to_compute.png)

> 图源：Milvus 官方文档（v3.0.x）

![外部集合之后：把算力带到数据旁](/images/milvus30_external_collection_bring_compute_to_data.png)

> 图源：Milvus 官方文档（v3.0.x）

什么时候该用外部集合：数据因合规或成本"离不开湖"（金融、医疗常见）；数据已是 TB 级 Parquet/Lance/Iceberg 表、不想再拷一份；同一份湖上数据要给多个 Milvus 实例共用。什么时候仍该用普通集合：需要 insert/upsert/delete 与实时一致性；需要 TEXT 长文本字段（3.0.0 里 TEXT 还不支持外部集合）；数据本来就在 Milvus 里流转。


### 7.2.3.2 支持的存储格式与前置准备

本节解决"外部集合能吃哪些格式、建之前要备什么"，不解决"这些生态产品怎么用"——这里只讲 Milvus 视角，不展开 Parquet/Iceberg 背后的产品本身。

Milvus 外部集合支持的存储格式分三组（口径统一为"Milvus 外部集合支持的存储格式"）：

- **文件格式**：`parquet`（默认、生态最广的列式格式）、`vortex`（可选的新一代列式格式，靠可组合编码与统计信息支持列投影、范围读与随机读）。两者都以"目录或对象存储前缀"为源。
- **表格式**：`lance-table`（Lance 数据集目录）、`iceberg-table`（Iceberg 元数据 JSON + snapshot ID；保留 position-delete，equality-delete 不支持，须先转成 position-delete 再 refresh）。用 `iceberg-table` 时，`external_spec` 里要带 `snapshot_id` 指定 Iceberg 快照。
- **Milvus 原生**：`milvus-table`（3.0.0 新增）。它把 Milvus **快照的 metadata + Storage V3 manifest** 直接当外部源，等于让一份集合快照本身就能被当成外部表来查，批处理与在线服务共享同一份 manifest 支撑的视图。

字段类型映射在**建集合时定死**：每个 Milvus 字段通过 `external_field` 指向源文件里的列名，源列类型必须和 Milvus 字段类型对得上（写法见 7.2.3.3）。

> 📌 **前置条件**：建外部集合前，要先有**对象存储访问凭据**。数据就存在 Milvus 自己的桶里时，可直接复用全局 MinIO 凭据（`storage_type: "remote"`）。跨账号读别的云账户数据时，按云厂商有对应鉴权方式：AWS S3 用 IAM Role ARN（走 STS `AssumeRole`）+ 可选 External ID；GCS 用目标服务账号（impersonation）；Azure Blob / ADLS Gen2 经 `milvus-tools` 拿短期 SAS；阿里云 OSS 用 RAM Role ARN。其中 Azure Blob 不支持 `iceberg-table`。


### 7.2.3.3 实操：挂载一个外部集合

本节解决"从零挂载一个外部集合的最小闭环"，不解决"索引与检索"——那在 7.2.3.4。普通集合的创建写法见 Chapter 3（3.3.7/3.3.8），此处只给外部集合的差异参数。

三步走：建 schema（带外部源）→ 加字段（带外部字段映射）→ 建集合。

第一步，`create_schema()` 指定外部源 URI 与数据格式：

```python
from pymilvus import MilvusClient, DataType

schema = MilvusClient.create_schema(
    external_source="s3://s3.<region-id>.amazonaws.com/<bucket>/",
    external_spec='{"format": "parquet", "extfs": {"access_key_id": "AKIA..", "access_key_value": "u4Lh...", "region": "us-west-2", "cloud_provider": "aws", "use_ssl": "true", "use_virtual_host": "true"}}'
)
```

第二步，`add_field()` 里用 `external_field` 把 Milvus 字段名映射到外部文件里的列名（官方示例：库内 `product_id` ↔ 文件列 `id`）：

```python
schema.add_field(field_name="product_id", datatype=DataType.INT64, external_field="id")
schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=512, external_field="name")
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768, external_field="vector")
```

第三步，照常建集合：

```python
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
client.create_collection(collection_name="test_collection", schema=schema)
```

预期输出：`create_collection()` 正常返回即表示外部集合已建立，此时数据仍在源处、未拷贝进 Milvus。注意 `external_spec` 里的 `extfs` 换成你自己的凭据与 region 才能跑通；官方还支持 `volume://my_volume/path/to/a/folder` 这种本地卷路径形态。


### 7.2.3.4 在外部集合上建索引与检索

本节解决"外部集合上怎么建索引、怎么检索"，不解决"索引原理"——索引类型科普在 Chapter 2（2.3.3），此处只讲外部集合上的可用性差异。

核心心智一句话：**索引在库内、数据在库外**。建索引和普通集合写法完全一致，Milvus 把索引算在库内，检索时再按 manifest 映射去读库外的源文件。

建索引：

```python
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="product_name", index_type="AUTOINDEX")
client.create_index(db_name="my_database", collection_name="test_collection", index_params=index_params)
```

检索（语义与普通集合一致）：

```python
results = client.search(
    collection_name="test_collection",
    data=[[0.1] * 768],
    anns_field="embedding",
    limit=10,
    output_fields=["product_id", "product_name"],
)
```

预期输出：返回与查询向量最接近的 top-10，且能带出 `product_name` 等外部列的值——这些值在检索时才从源文件读出来。

外部列上能用的能力与普通集合对齐：全文索引/BM25、标量索引、JSON、动态字段、分区键、clustering key 都支持。唯一要留意的差异在部署侧：官方 Next steps 提示，按需计算数据库里的集合，DQL 前要先建 session 挂上按需集群的算力，否则 search/query 不可用。


### 7.2.3.5 外部列直连函数输出字段

本节解决"湖上原文怎么变成库内的检索表示"，不解决"函数语法细节"——BM25/MinHash/embedding 的完整语法见 7.3.4.4，这里只讲"外部列当函数输入"这一能力。

这是本子章节最有价值的一点。3.0.0 起，**外部列可以直接作为函数输入**，产出 BM25 稀疏向量、MinHash 签名、text embedding（官方 release notes 原话：External fields can now feed function output fields such as BM25 sparse vectors, MinHash signatures, and text embeddings）。它解掉了"湖上原文 + 库内检索表示"的双写难题：文本仍留在湖里，检索用的稀疏/稠密向量在库里生成，不用把源表拷进来。

首选 BM25——它是本地文本函数，在 Milvus 内运行、不调外部模型、结果可解释。概念示意（外部 VARCHAR 列 `text` 作为输入，产出稀疏向量列 `text_sparse`）：

```python
from pymilvus import DataType, Function, FunctionType, MilvusClient

# 建外部集合时，把湖上原文列映射进来
schema = MilvusClient.create_schema(
    external_source="s3://<bucket>/docs/",
    external_spec='{"format": "parquet"}',
)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512, external_field="content")
# 函数产出稀疏向量列（完整语法见 7.3.4.4）
bm25_function = Function(
    name="text_bm25",
    input_field_names=["text"],
    output_field_names=["text_sparse"],
    function_type=FunctionType.BM25,
)
```

预期效果：`text` 的值来自湖上文件、不落库；`text_sparse` 由 BM25 在库内算出，检索时按词频相关度打分。MinHash（近重复检测、产出 `BINARY_VECTOR`）与远端 text embedding（产出 `FLOAT_VECTOR`）同理，只是函数类型与输出字段类型不同。

> ⚠️ **注意**：远端 embedding 函数要对模型做批量调用，开销不小且依赖外部服务可用性；数据量一大，优先选 BM25/MinHash 这类库内函数，确需语义向量再上远端模型。另外，"外部列 + 函数"这一组合的完整成例我未在官方缓存中找到（待验证）；"外部列经 `external_field` 映射进来"与"`Function` 组装"各自是官方确定的，二者拼装的具体写法以 7.3.4.4 为准。


### 7.2.3.6 增量 refresh：让外部数据追平

本节解决"湖上数据变了，外部集合怎么追平"，不解决"第一次挂载"——那在 7.2.3.3。

refresh 不是实时同步：源数据有更新后，要**手动**再调一次 refresh，Milvus 才会重新扫文件元数据、更新 manifest。官方口径是 refresh 扫描数据文件元数据并据此生成 manifest，通常耗时 150–250 ms；它只更新元数据映射、不搬数据、也不改源文件。

调用与进度轮询（官方示例，refresh 是异步的）：

```python
job_id = client.refresh_external_collection(
    db_name="my_database",
    collection_name="test_collection",
)
while True:
    progress = client.get_refresh_external_collection_progress(job_id=job_id)
    print(f"  {progress.state}: {progress.progress}%")
    if progress.state == "RefreshCompleted":
        print(f"  完成，耗时 {progress.end_time - progress.start_time}ms")
        break
    elif progress.state == "RefreshFailed":
        print(f"  失败：{progress.reason}")
        break
    time.sleep(2)
```

预期输出：轮询到 `RefreshCompleted` 时，说明新元数据已生效，检索即可看到湖上的变化。

什么时候该刷的决策建议：

- **源数据变一次刷一次**：refresh 是显式操作，不刷就一直看旧视图。
- **离线批处理可以攒着刷**：多轮源更新后刷一次即可，省掉反复扫元数据。
- **注意一个边界**：官方明确"一次把全部活跃元数据移除、且没有任何新增"的 refresh 会被拒绝——把源目录清空这类操作不会让你把集合刷成空壳。


### 7.2.3.7 外部集合的 schema 调整

本节解决"湖上加了列，外部集合怎么办"，不解决"普通集合在线加列"——那是 7.2.5，这里专指外部集合。

规则只有一条：**只能"加"，且只能加源里已有的字段**。湖上表新增一列后，用 `add_collection_field()` 把该列映射进来（`external_field` 指向源列名），再 refresh 即可，无需重建集合：

```python
client.add_collection_field(
    collection_name="product_embeddings",
    field_name="score",
    data_type=DataType.DOUBLE,
    nullable=True,
    external_field="score",
)
client.refresh_external_collection(collection_name="product_embeddings")
```

预期输出：refresh 后 `score` 可用于查询输出与过滤。官方 Limits 明确：删字段、改名、改类型、改向量维度、改 `external_field` 映射都不支持；`SPARSE_FLOAT_VECTOR` 与 StructArray 字段不能事后加。类型不兼容无解——源列类型和 Milvus 字段类型必须本来就对得上，否则只能重建。


### 7.2.3.8 常见报错与排查

本节解决"外部集合挂了、刷不动、查不到时怎么排"，不解决"服务端集群故障"——那是 Chapter 6。以下按"现象 / 原因 / 处理"给出排查条目（依据官方限制与操作页，非实跑报错原文）：

#### 1. 建集合报凭据或路径错

- **现象**：`create_collection()` 或首次 refresh 失败，报对象存储访问错误。
- **原因**：`external_source` 路径写错，或 `extfs` 里 access_key_id / access_key_value / region / use_ssl / use_virtual_host 有一项不对；跨账号场景 role_arn / external_id 配错。
- **处理**：用同一凭据在对象存储侧 `ls` 该前缀确认可读；对照 7.2.3.2 的鉴权表逐项核对 `extfs`。

#### 2. 格式不匹配

- **现象**：refresh 扫不出文件，或 manifest 生成失败。
- **原因**：`external_spec.format` 写错（如把 Iceberg 表写成 `parquet`），或 `iceberg-table` 没给 `snapshot_id`。
- **处理**：按 7.2.3.2 的格式清单核对 `format` 值；`iceberg-table` 补 `snapshot_id`。

#### 3. 想在只读集合上写入

- **现象**：想调 insert / upsert / delete 发现没有接口或直接被拒。
- **原因**：外部集合是 read-only，官方未提供写入接口。
- **处理**：改数据回源湖仓改，再 refresh 追平（见 7.2.3.6）。

#### 4. 外部字段映射不到

- **现象**：加了字段但查询输出该列为空，或 refresh 报字段不存在。
- **原因**：`external_field` 指向的列在源文件里不存在——外部集合只能映射"源里已有"的列。
- **处理**：核对源 schema，确保 `external_field` 的列名、类型都对得上。

#### 5. refresh 后结果没变化

- **现象**：源数据改了，检索结果还是旧的。
- **原因**：refresh 是显式操作，没刷就一直看旧视图；或触发"移除全部元数据且无新增"被拒。
- **处理**：显式再调 `refresh_external_collection()` 并轮询到 `RefreshCompleted`（见 7.2.3.6）。


## 7.2.4 快照（Snapshot）

### 7.2.4.1 快照是什么

本节解决"快照到底是什么、它和备份/副本差在哪"，不解决"怎么调 API"——那在 7.2.4.2。分区（Chapter 3 3.9）是另一码事，别拿分区当快照讲。

快照（Snapshot）的官方定义：按段引用（segment reference）实现的**时点只读视图**（point-in-time read-only collection views）。它只写元数据/manifest、不复制向量数据，所以创建极快；但要注意——快照会**钉住**被引用的段文件不被 GC，因此保留快照会占用与这些段相当的对象存储空间，多份快照还会叠加；官方建议定期删除不再需要的快照以释放存储。它依赖 Storage V3 已开启。

三个最容易混的词，分三段辨析：

- **快照 vs 备份**：快照是"时点只读视图"，创建只写元数据（毫秒级）、恢复走 copy-segment 只拷现有文件；备份是独立完整拷贝，创建要复制所有数据文件、恢复要重导数据重建索引。快照适合几天到几周的快速回滚与测试，备份适合几周到几年的灾备。
- **快照 vs 副本**：快照是同一份数据在某个时间点的只读视图，数据不复制、只按段引用；副本（多副本/冗余）是同一份数据在另一处/另一实例的完整冗余拷贝，用于容灾与高可用，两者目标不同。
- **快照 ≠ 分区**：分区是给集合内部切分管理，快照是给整份数据拍时点照，不要混用。

### 7.2.4.2 实操：创建·列举·恢复·删除

本节解决"快照四个动作怎么写"，不解决"快照原理"——那在 7.2.4.1。四个动作的参数名以 manage-snapshots.md 为准。

> 💡 **版本说明**：Milvus Lite 的 snapshot 是嵌入式引擎机制，与服务端快照不是一回事（见 7.1.5.1），本节只讲服务端。

创建前建议先 `flush()`，避免漏掉还没落盘的数据（官方："Calling flush() is not mandatory but highly recommended"）：

```python
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
client.flush(collection_name="my_collection")
client.create_snapshot(
    collection_name="my_collection",
    snapshot_name="backup_20240101",
    description="Daily backup for January 1st, 2024",
)
```

预期输出：返回即表示快照已建立（只写了元数据，毫秒级）。

列举：

```python
snapshots = client.list_snapshots(collection_name="my_collection")
```

恢复——**恢复到新集合**，异步返回 job_id 供跟踪（copy-segment 机制，只拷现有 segment 文件、不重导不重建索引）：

```python
job_id = client.restore_snapshot(
    snapshot_name="backup_20240101",
    collection_name="restored_collection",
)
```

删除（官方建议定期清理旧快照省存储）：

```python
client.drop_snapshot(snapshot_name="backup_20240101")
```

预期输出：`restore_snapshot()` 返回 job_id 后，用 `list_restore_snapshot_jobs()` 查进度；`drop_snapshot()` 返回即删。恢复目标是新集合，不是覆盖原集合。


### 7.2.4.3 四个典型用例

本节解决"快照能拿来干什么"，不解决"怎么创建"——那在 7.2.4.2。每个用例 = 场景一句话 + 几行代码，不贴整页 API 表。

**用例一：误删回滚**。批量删改前先拍快照，删错了恢复到新集合比对：

```python
client.create_snapshot(collection_name="my_collection", snapshot_name="pre_cleanup")
# ... 执行有风险的删除/改写 ...
job_id = client.restore_snapshot(snapshot_name="pre_cleanup", collection_name="rollback_collection")
```

**用例二：A/B 版本对照检索**。同一查询打在不同版本的快照视图上比结果：

```python
client.create_snapshot(collection_name="my_collection", snapshot_name="v1_model")
# ... 升级模型、重灌数据 ...
client.create_snapshot(collection_name="my_collection", snapshot_name="v2_model")
# 分别在 v1_model / v2_model 恢复出的集合上跑同一 query，对照排序差异
```

**用例三：灌数据中途失败回退到干净基线**（对应官方 2.6→3.0 灌数据失败案例）：灌数据前拍基线，跑挂了从基线重来：

```python
client.create_snapshot(collection_name="my_collection", snapshot_name="clean_baseline")
# ... 灌数据失败 ...
job_id = client.restore_snapshot(snapshot_name="clean_baseline", collection_name="retry_collection")
```

**用例四：配合备份做时间点还原**。快照给"时点视图"、备份给"长期灾备"，两者分工见 7.2.4.4；快照还可用 `milvus-table` 外部格式被当成外部表查（见 7.2.3.2）。


### 7.2.4.4 快照与备份恢复的分工

本节解决"什么时候用快照、什么时候用备份恢复"，不解决"Chapter 6 的架构容错"——此处只讲数据面。

四件事分点说清：

- **解决什么问题**：快照 = 同实例内的快速回滚/测试（时点视图，天到周）；备份 = 独立完整拷贝，用于跨实例、跨存储、长周期的灾备（周到年）。
- **是否跨实例**：快照留在本实例的对象存储里，恢复也是本实例内恢复到新集合；备份靠 `milvus-backup` 工具，能在不同实例、甚至不同对象存储间迁移（single-instance / cross-bucket / multi-storage 三种场景）。
- **是否依赖对象存储**：两者都落在对象存储上；快照引用现有 segment 文件、只写元数据，备份则把数据文件复制成独立完整拷贝。
- **能回滚到什么粒度**：快照回滚到"集合的某个时点"；备份还原到"备份时刻的完整集合"。快照创建毫秒级、恢复秒到分钟级；备份创建耗时、恢复要重导数据重建索引。

一句话选型：同实例、要快、要常做，用快照；要跨实例、要抗整库故障、要长期留，用备份。官方口径是快照毫秒级创建、秒到分钟级恢复，具体数值随数据量变化（待补测）。


## 7.2.5 Schema 在线演进：加列·回填·删列

7.0 说过，在线改字段有两条限制值得记住，第一条就是这里的事。这段原文从 7.0 迁来，先读它，再看 7.2.5.1 的实操：

**一是新增字段必须可空，回填也不是自动的。**

官方原文：Added user-defined fields must be nullable. Set `nullable=True` when calling `add_collection_field()`. For existing entities, the added field is `NULL` unless you add a scalar field with a `default_value`. 

也就是说，给已经有数据的 collection 加字段，老数据不会自动"长"出新值。把它想成给一张已经有 100 万行的表加一列：

- **这一列对老数据是空的（`NULL`）。** 注意 `NULL` 不是 0，也不是空字符串，而是"这里没有值"。数据库不会猜你想填什么，所以它就空着。
- **唯一会自动有值的情况**：新加的是标量字段（int、float、varchar 这类），而且你在添加时给了默认值。这样 100 万条老数据会统一读到这个默认值。
- **如果要给每条老数据填不同的值，就只能自己补。** 官方给的两条路是 upsert（按主键把新值写回对应实体）或者回填流程。加完字段就放着不管，老数据会一直是 `NULL`。
- **如果新加的是向量字段**，老数据在补上向量之前，按这个字段检索时会被直接跳过——不是排在末尾，而是根本不出现在结果里。

一句话：**加字段只是开了一个空槽；填不填、填什么，取决于有没有给默认值、有没有补写数据。**

### 7.2.5.1 在线加列与删列

本节解决"已有集合怎么不停机加列、删列"，不解决"建表时定 schema"——那在 Chapter 3（3.3.3 创建 Schema / 3.3.6 动态字段），此处只讲**建完之后改**。

3.0 把过去"重建集合或双写"的加列流程，收敛成一次 `add_collection_field()` 调用，服务不中断、老数据照常可查。新增的用户自定义字段**必须 nullable**，老数据该字段为 `NULL`（除非加了 `default_value`）。

加列（标量 + 默认值）：

```python
client = MilvusClient(uri="http://localhost:19530")
client.add_collection_field(
    collection_name="product_catalog",
    field_name="review_status",
    data_type=DataType.VARCHAR,
    max_length=32,
    nullable=True,
    default_value="unreviewed",
)
```

预期输出：返回后 `describe_collection()` 能看到新字段；老实体读到默认值 `"unreviewed"`，新实体可显式写别的值。

删列（标量）：

```python
client.drop_collection_field(
    collection_name="product_catalog",
    field_name="experiment_tag",
)
```

预期输出：删后 `describe_collection()` 不再返回该字段，建在它上面的索引一并清理；存储空间回收走单独流程、异步完成。

> ⚠️ **注意**：新增**向量字段**时老数据该列为 `NULL`，补上向量前按该字段检索会被直接跳过（不出现在结果里）；要可检索须 upsert 或回填（见 7.2.5.2）。删字段有硬限制：不能删主键、分区键、clustering key，也不能删掉集合里最后一个向量字段。


### 7.2.5.2 双向回填

本节解决"新加的字段怎么把老数据填上"，不解决"加列本身"——那在 7.2.5.1。回填分两个方向：外部回填、内部回填。

**外部回填**：值在库外算好再灌入。典型流程是"加列 → 拍快照当一致起点 → 离线算值 → 写回 → Milvus 增量建索引"。写回用 upsert 按主键把新值写回对应实体：

```python
client.upsert(
    collection_name="product_catalog",
    data=[
        {"id": 1, "review_status": "approved"},
        {"id": 2, "review_status": "rejected"},
    ],
)
```

预期输出：按主键把 `review_status` 写回老实体，之后检索/过滤可命中。

**内部回填**：给已有集合挂 BM25/MinHash 函数，由内核从已有字段派生新向量列，后台 compaction 异步回填：

```python
from pymilvus import DataType, Function, FunctionType, MilvusClient

sparse_field = client.create_field_schema(
    name="text_sparse", data_type=DataType.SPARSE_FLOAT_VECTOR,
)
bm25_function = Function(
    name="text_bm25",
    input_field_names=["text"],
    output_field_names=["text_sparse"],
    function_type=FunctionType.BM25,
)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="text_sparse", index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)
client.add_function_field(
    collection_name="product_catalog",
    field_schema=sparse_field,
    func=bm25_function,
    index_params=index_params,
)
```

预期输出：`describe_collection()` 能看到新函数与新字段，老数据的 `text_sparse` 由后台 compaction 异步生成。

三个硬事实要记住：

1. **异步**：`describe_collection()` 看到新字段 ≠ 回填完成，老数据的值在后台 compaction 里慢慢生成。
2. **双开关**：给已有集合加函数，要求 schema version compaction 与 storage version compaction 同时开启（`dataCoord.compaction.bumpSchemaVersion.enabled` 与 `dataCoord.compaction.storageVersion.enabled`），任一关闭请求被拒。
3. **输入必须 VARCHAR**：BM25/MinHash 的输入必须是 `VARCHAR` 字段，`TEXT` 不支持这条回填路径。


### 7.2.5.3 改造前后对照

本节解决"同样的需求，2.6 和 3.0 各怎么做"，不解决"逐行代码"——那在 7.2.5.1 / 7.2.5.2，这里只做前后对照。

**2.6 的真实做法**：加字段并非 2.6 做不到——2.6.x 起就能加可空标量字段（`add_collection_field` + `nullable=True` + 可选 `default_value`），2.6.18 起还能加用户向量字段。真正的短板在"收尾"：删列没有接口、给已有数据补 BM25/MinHash 派生向量也没有，于是退役一个字段或给老数据补向量，只能走重建集合或双写——停服重建有 downtime，双写要长期维护两套写入。

**3.0 的做法**：加列、回填、删列一条龙在线完成。加列后老数据 `NULL`（或默认值）→ 外部回填或内部回填把值填上（7.2.5.2）→ 不需要的字段 `drop_collection_field()` 删掉（7.2.5.1），全程服务不中断。

> ⛳ **架构演进提示**：这一步能成立，是因为 3.0 的 Storage V3 底座把 schema 版本演进、后台 compaction 与增量日志统一到了一张 manifest 上（7.2.2）。2.6 的"三线分置"存储缺的正是这个能按版本演进的自描述底座，所以同样一件事，2.6 只能靠"重建/双写"这类应用侧补丁，3.0 则把它下沉成了存储引擎原语。


## 7.2.6 本章小结

本章核心掌握：

1. **Storage V3（Loon）是新底座**：manifest 把数据与元数据统一落到对象存储，数据自描述、按版本演进；默认关闭，用 `common.storage.useLoonFFI` 开启，开启后新写入与 compaction 产物才换新布局（7.2.2）。
2. **外部集合让数据不必搬进来**：原地读、只读、靠增量 refresh 追平；支持的格式从 Parquet/Vortex/Lance/Iceberg 延伸到 `milvus-table`，外部列还能直连函数产出 BM25/MinHash/embedding（7.2.3）。
3. **快照是时点只读视图**：只写元数据、按段引用，创建毫秒级；但会钉住被引用文件，保留成本约等于集合大小，与备份/副本要分清（7.2.4）。
4. **schema 可在线演进**：加列、外部/内部双向回填、删列一条龙，取代过去的重建集合或双写（7.2.5）。

7.3 将在 7.2 打下的数据底座之上，看 3.0 的数据模型与检索能力——那里会继续用到这里讲清的 manifest 与外部集合概念。

