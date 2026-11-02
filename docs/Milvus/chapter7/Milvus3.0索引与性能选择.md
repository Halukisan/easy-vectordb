# Chapter 7.4 Milvus 3.0 索引与性能选择（实战）

## 7.4.0 这一章干什么

三个真实场景：全文检索、稀疏召回变慢了；稠密向量把内存吃满；手里有一份调好的 Faiss 配方，不想在 Milvus 里从头调参。3.0 对应给了三类新索引——SINDI 稀疏索引、FAISS 直通、EmbList + DiskANN 磁盘索引——这一章讲清「各管什么、怎么启用、怎么选」，选完建索引、小批量验证召回，两步收工。

上一章（7.3）把排序、重排塞进了同一次搜索请求；这一章往下走一层：决定这次请求落在哪个索引上。存储格式、写入引擎、段合并、Storage V3 清单这些底层不在这——想搞懂 7.2 里那些反直觉行为的根子，见 [存储与写入引擎·深入](./Milvus3.0存储与写入引擎)。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)

## 7.4.1 稀疏索引换代：SINDI、Block-Max WAND、Block-Max MaxScore

全文检索、学习型稀疏 Embedding 用这个：3.0 给稀疏索引整代升级，代价是要手动开。

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

手里有调好的 Faiss 工厂字符串、专用索引类型又覆盖不了时用——官方定性「专家级直通功能」，先看硬限制再上手。

「FAISS 索引类型是 Milvus 3.0.0 及更高版本中提供的一种专家级直通功能。它允许您提供一个 Faiss 索引生成器字符串，而非选择固定的 Milvus 索引类型。」——你手里有调好的 Faiss 配方，原样搬进来。

适用面要看清：「上游 Faiss 接受的工厂字符串并不一定被 Milvus 自动支持。」兼容性取决于向量字段类型、度量、维度、编译进镜像的 Faiss 模块，以及索引是否支持 Milvus 所需的操作。官方也提醒：常见配方优先用专用索引类型，「因为它具有稳定且有文档记录的参数契约」。

硬性限制：只支持 `FLOAT_VECTOR` 和 `BINARY_VECTOR`（不支持 FLOAT16/BFLOAT16/INT8/SPARSE）；「通用 FAISS 适配器在 CPU 上运行。它不是 Faiss GPU 索引类型」；必须指定 `faiss_index_name` 构建参数。

**参考文档**：[FAISS 索引类型](https://milvus.io/docs/zh/faiss.md)

## 7.4.3 EmbList + DiskANN：嵌入列表的磁盘索引

ColBERT/ColPali 这类晚期交互模型一行就是一串向量，或向量总量超出内存预算时，用磁盘路径。

这是 7.3.2 StructArray 的索引侧配套。官方描述：「EmbList + DISKANN——针对 StructArray 嵌入列表的磁盘上多向量索引，包含 Muvera 和 Lemur 等加速路径。」设计动机：「ColBERT 等晚期交互模型会为每个令牌生成一个向量，而多模态实体可能包含多种视图。EmbList 为每个实体存储一个可变长度的向量列表，并以 DISKANN 作为磁盘索引。当语料库超过内存预算时，磁盘路径可有效控制 RAM 占用量。」

怎么建？在 StructArray 的向量子字段上建 `DISKANN` 索引，搜索走 MAX_SIM（见 7.3.2 引擎依赖关系）。Muvera / Lemur 是同一条官方描述里列出的加速路径，具体用法以当前版本的官方支持说明为准。

什么时候值得换：ColBERT/ColPali 类晚期交互检索、向量总量超出内存。

**参考文档**：[使用 Embeddings 列表进行搜索](https://milvus.io/docs/zh/search-with-embedding-lists.md)

## 7.4.4 怎么选：负载 → 索引对照

不用背三节细节，先看这张表——每行都是前三节官方表述的汇总，不新增论断：

| 你的负载 | 用什么 | 理由 |
|---|---|---|
| 稠密向量、内存够 | 专用索引类型（`AUTOINDEX` 起步） | 官方提醒「常见配方优先用专用索引类型」，参数契约稳定（见 7.4.2） |
| 全文 / BM25 / 稀疏 Embedding | 开新索引版本 → SINDI（IP）/ MaxScore（BM25） | 官方「SINDI 将成为稀疏 IP 搜索的默认选项，而 MaxScore 将成为 BM25 的默认选项」（见 7.4.1） |
| ColBERT/ColPali、一行多向量 | EmbList + DISKANN | 官方「针对 StructArray 嵌入列表的磁盘上多向量索引」（见 7.4.3） |
| 向量总量超内存预算 | DISKANN 磁盘路径 | 官方「磁盘路径可有效控制 RAM 占用量」（见 7.4.3） |
| 手里有现成 Faiss 配方、专用类型覆盖不了 | FAISS 直通 | 专家级功能，先核对硬限制（见 7.4.2） |

选完怎么验证：建索引 → 拿一批典型查询跑小批量 → 对比 QPS 和召回。建索引与刷新的操作在 7.2.4，检索的写法在 7.3.2，都是现成的，不重复贴代码。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md) · [FAISS 索引类型](https://milvus.io/docs/zh/faiss.md) · [使用 Embeddings 列表进行搜索](https://milvus.io/docs/zh/search-with-embedding-lists.md)

## 7.4.5 小结

- 稀疏索引换代三件套：SINDI、Block-Max WAND、Block-Max MaxScore；BM25 索引小约 3 倍，SINDI 学习型稀疏 QPS 可达 MaxScore 的 10 倍；靠 7.1.2 的索引版本开关启用，SINDI/MaxScore 成为各自领域的默认；
- FAISS 直通是专家级功能：给生成器字符串，只支持 FLOAT/BINARY，CPU，必须 `faiss_index_name`；常规场景用专用索引类型；
- EmbList + DiskANN：ColBERT/ColPali 的磁盘索引方案，RAM 预算紧时用；

- 存储格式（Vortex/Lance）、写入引擎（Woodpecker）、强制合并、Storage V3 清单是另一层，见 [存储与写入引擎·深入](./Milvus3.0存储与写入引擎)。

**参考文档**：[Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md) · [FAISS 索引类型](https://milvus.io/docs/zh/faiss.md) · [使用 Embeddings 列表进行搜索](https://milvus.io/docs/zh/search-with-embedding-lists.md)

下一章把所有东西串起来：从一份 Parquet 文件到快照、重排，完整演一遍。

> **下一章**：[7.5　Milvus 3.0 端到端实战](./Milvus3.0端到端实战)
