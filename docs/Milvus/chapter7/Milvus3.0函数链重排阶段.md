# Chapter 7.3 深入 · FunctionChain 重排阶段（L0 / L1 / L2）

> 这是 7.3《数据模型与检索增强》里 FunctionChain 重排的深入版。7.3 只给了「三个阶段各一句话 + 本文为什么选 L2」；这一篇把三者的执行位置、每条候选如何流转、提前处理到底省了什么讲清楚。结论逐条标了出处层级（正式文档 / 设计文档 / 源码），查不到的一律标「官方未明确」。

## 一、三个阶段：在哪执行、看到多少

链的第一个参数是「执行阶段」。官方 C++ 文档写明，给搜索结果做后处理用 `L0_RERANK`、`L1_RERANK`、`L2_RERANK` 三者之一；但 `FunctionChainStage` 枚举本身只把三者定义成「第一 / 第二 / 第三个 rerank 阶段」，没说清各自在哪跑、能看到多少数据。通过阅读 Function Chain 设计文档与源码，发现它们是按「结果在哪一层被归并」划出三道执行边界：

- **`L0_RERANK`（第一道）**：在 QueryNode 上、**每个 segment 各自**执行，发生在跨 segment 归并**之前**；只支持 `map`（逐行改分），没有全局视角。
- **`L1_RERANK`（第二道）**：同样在 QueryNode 上，但发生在**本节点各 segment 归并之后、结果送回分片 leader 之前**；拿得到本节点归并后的那批局部候选，支持 `map` / `sort` / `limit`，但看到的仍是局部。它 3.0.1 才加入（3.0.0 只有 L0、L2）。
- **`L2_RERANK`（第三道）**：在 **Proxy** 上、**全局归并之后**执行——所有分片的结果都汇总到 Proxy、合并成一份之后才跑这条链。这是唯一能看到「全部候选」的阶段，直接复用 Proxy 既有的 rerank。

| 阶段 | 在哪执行 | 相对归并的位置 | 能看到的数据 | 支持的操作 |
| --- | --- | --- | --- | --- |
| `L0_RERANK` | QueryNode | 跨 segment 归并**前**（逐 segment） | 单个 segment 的结果 | `map` |
| `L1_RERANK` | QueryNode | 本节点 segment 归并**后**、送回分片 leader **前** | 本节点归并后的局部候选 | `map` / `sort` / `limit` |
| `L2_RERANK` | Proxy | 全局归并**后** | 全部分片汇总后的全部候选 | 通用（`map`/`sort`/`limit`/`filter` 等，含模型重排） |

## 二、数据在哪：collection → shard → segment → QueryNode

一条检索进来后，数据是这样分布的：collection 先被切成多个 shard（由 `shard_num` 决定），每个 shard 的数据又聚成多个 segment；每个 shard 有一个 delegator（分片 leader，本身是个 QueryNode）负责把这些 segment 分发给若干 worker（QueryNode）去持有、去检索。所以单个 worker 从头到尾只持有并检索**本 shard 落在它身上的那几段 segment**，从没见过整库。

## 三、四层归并骨架与候选数实例

向量检索（ANN）在每一段 segment 上只吐 topK 条，之后的每一层都在重复同一件事：把下面交上来的多份 topK 按分数归并成一份 topK。候选数是这样缩小的（以 100 万条、3 个 shard、topK=100、每 shard 10 个 segment、每 shard 2 个 worker 为例）：

- **segment 层**：30 段各做一次 ANN，每段出 100 条，累计 30×100 = 3000 条，但每段手里只有自己的 100 条；
- **worker 层**：每个 worker 把它的 5 段 × 100 = 500 条归并成自己的 100 条，6 个 worker 合计 600 条；
- **shard leader 层**：每个 leader 把本 shard 2 个 worker 的 100 + 100 = 200 条归并成 100 条，3 个 shard 合计 300 条；
- **Proxy 层**：把 300 条归并成最终的 100 条。

`L0` / `L1` / `L2` 不改变这套骨架，只是插在三个不同的归并口上，看到的都是**同一次 ANN 结果、但范围不同**：`L0` 在归并前逐 segment 执行，看到的是**单个 segment 的那 100 条**；`L1` 在本 worker 归并后、送回 leader 前执行，看到的是**这个 worker 归并出的 100 条**；`L2` 在 Proxy 全局归并后执行，看到的是**最终那 100 条**。

所以「querynode 会不会查到和原始查询不一样的数据」——不会。L0/L1 处理的候选，就是这次查询本会在该 segment / 该节点返回的那批 topK 候选，差别只在截取于哪一层、能看见多大范围。真正把 100 万条都扫一遍的距离计算只发生在 ANN 检索里，和函数链在哪一层无关，链省不掉它。

## 四、提前处理省什么、省不掉什么

- **省的是跨层传输量和上层归并/排序的输入，不是向量检索。** 全放 `L2`，就是 300 条一路传到 Proxy，由 Proxy 这台单点串行做最后一次全局归并、全局排序、模型重排——这是 `L2` 的代价：动作最晚、中心点最烫、没有早期裁剪。`L1` 把排序/截断分发到 6 个 worker 并行做；若 `limit` 把每个 worker 从 100 裁到 50，离开 worker 的行数就从 600 减到 300，后面 leader、Proxy 的归并输入也跟着变小。代价是：`L1` 的 `limit` 是「每个 worker 各自的候选预算」，被裁掉的那条不会在 leader / Proxy 上补回来，定太小就降召回。L1 还有个更细的省法——它要用的额外标量字段只在「归并后活下来的候选」身上才去段里取，而不是对每个 segment 的全部候选都取（late materialization）。
- **`L0` 省的是另一种东西：让归并「按对分数选人」。** 它逐段改写 `$score`，发生在跨 segment 归并之前，于是 worker、leader、Proxy 每层归并都拿着改完的新分数去挑 topK。若拖到 `L2` 才改分，前面几层早已按旧分数把 topK 定死，新分数下本该进 topK 的候选可能压根没被传上来。`L0` 的价值主要在「选人阶段就用对分数」，而不是「少算几个数」。

## 五、选型判断：什么时候用 L0 / L1 / L2

- 只想逐行改分、且希望后续归并按新分选人（距离转相似度、逐行调分），用 `L0`（只有 `map`，无全局视角）。
- 量大、追低延迟、想「先在本节点内排好序、裁掉一批再上传」，用 `L1`（`map`/`sort`/`limit`），典型是「`L1` 先压一轮候选 + `L2` 最终全局重排」的两段式；注意 `limit` 是 per-worker 预算，别为省传输把召回裁没。
- 要**全局唯一排序**、要跑外部模型 rerank（`rerank_model` 首版只能在 `L2`），或要把所有分片汇总后统一比一次总分（7.3 的耳机场景），用 `L2`。代价是中心点全量排序、无早期裁剪。
- hybrid search 多路子查询：每个子查询是独立的一次 ANN，可各自挂 `L0`/`L1`（在子查询内部、其归并前/归并后执行），但跨子查询的融合与重排只能在顶层 `L2` 做——顶层 `L2` 链的第一个算子必须是 `merge`；各子查询自己的 `L0`/`L1` 只看得到该子查询的候选，彼此之间、和顶层 `L2` 之间都不共享中间列。

## 六、结论出处（按证据强度分层）

**正式文档（官方 release notes）**

- `L1` 是 3.0.1 才加入的：v3.0.1 发布说明「Added L1 reranking support to search function chains（#52745）」。
- hybrid 的组合方式：v3.0.3 发布说明「Function Chain support to Hybrid Search：顶层 L2 链以 `merge` 开头做融合重排，每个子查询各跑自己的 L0/L1」。

**设计文档（milvus 仓库 docs/design-docs/design_docs/）**

- 三层执行边界与完整流程：`20260624-function-chain-api.md`（MEP，High-level Search flow）：`worker 导出逐 segment DataFrame → L0(逐 segment) → worker 跨 segment 归并 → L1(归并后的 DataFrame) → shard leader 与 Proxy 归并 → L2 rerankOperator`。
- `L1` 的 `limit` 是「每个 worker、每个 query-chunk 独立的候选预算」、被裁不会在上层重现 → 会降召回（同文档）。
- 用户链之后 QueryNode 做内部规范化排序（`$score` desc、`$id` asc）恢复下游归并契约（同文档）。
- L1 输入字段只对「归并后活下来的候选」做 late materialization（同文档）。
- `rerank_model` 首版只能在 L2（同文档）。
- shard / delegator / worker / segment 的持有关系：`20230418-querynode_v2.md`。
- hybrid 顶层必须 `merge` 开局、子查询 L0/L1 执行位：`20260818-hybrid-search-function-chain.md`。

**源码（internal/querynodev2/tasks/、internal/proxy/）**

- 每层归并目标条数 = 请求 topK：`search_task.go`（`topk = req.GetReq().GetTopk()`）。
- segment → worker 的跨段归并是 k 路堆归并，按 score desc / PK asc 预排序契约：`heap_merge_reduce.go`。
- L0 改分发生在跨段归并之前、归并随即用新分数选人：`search_task.go`（applyL0Rerank 在 executeGoReduce 之前）。
- Proxy 端的跨 shard 归并与 topK 执行归 `searchReduceOperator`：`internal/proxy/search_agg/computer.go` 注释。

## 七、仍不确定 / 官方未明确

- 纯 search（不挂函数链）路径下 `offset` 是否让底层多取 `topK+offset` 条——本次未逐行确认；函数链路径的设计文档只写明 `offset` 是请求级分页，在 `L2`（无显式 limit 时）之后兜底应用。上面的候选数实例只讲 topK 骨架，未断言 offset。
- 实例里「每 shard 10 段、每 shard 2 个 worker」是假设，只为把候选数走通；实际取决于 `shard_num` 与 segment 分布。设计文档只说一个 QN 可同时是 Delegator 和 Worker，未规定一个 worker 是否跨 shard 持有 segment——这不影响「每跳 topK」的结论，只影响具体数字。
- 「`L0` 改分会改变哪些候选进入上层 topK」是源码级强推断（applyL0Rerank 在归并前 + 归并以新分数取前 topK），官方没有一句话明文声明；宜当作「实现机制」而非「官方承诺语义」。
- L0/L1 各阶段的量化延迟/成本对比，官方未给出可复用基准数字，本文只做定性。

## 附：换模型没提前建向量列怎么办

> 本文是 7.3.3「换模型不重建集合」的延伸阅读：那里假设第二列 `vec_v2` 在建表时就一起建好了。这里回答另一件事——线上集合跑了很久、当初没预留新向量列，现在要换模型怎么办。

**官方正路：事后补列三步。** Milvus 2.6.18+ 支持用 `add_collection_field` 在线加向量列，但有硬性要求：新向量列必须 `nullable=True`，已有实体在回填前该列是 `NULL`，向量检索会跳过 `NULL` 行。正确顺序是「先建索引、后回填」：

1. 加列，`nullable=True` 必填：`add_collection_field(collection_name=…, field_name="vec_v2", data_type=FLOAT_VECTOR, dim=…, nullable=True)`；
2. 先建索引：`prepare_index_params().add_index(field_name="vec_v2", index_type="AUTOINDEX", metric_type="COSINE")` 再 `create_index(…)`；
3. 再回填：`upsert(…, partial_update=True)` 按 id 分批写非 NULL 向量；期间旧列照常服务，`NULL` 行被检索自动跳过。

**卡 `InProgress` 的解除阶梯。** 若 `create_index` 后索引长时间停在 `InProgress`（`describe_index` 看状态，或 datacoord 日志里 `pendingIndexRows` 一直不动）：

1. `drop_index` 删掉卡住的索引 → 重新 `create_index`；
2. 不行 `drop_collection_field` 删列重来；
3. 还不行重启 datacoord / indexnode 容器；
4. 都无效再考虑升级 Milvus 或到 GitHub 提 issue。

**生产兜底：新集合 + 双写切换。** 不想在存量集合上冒险，就新建一个带 `vec_v2` 的集合，新数据同时写新旧两个集合，老数据按 id 分批导出重插（或 `milvus-backup` 导出后导入新集合），验证后应用切读新集合。代价是迁移期要双写，但最稳。

**决策树**

```text
现在要换模型，vec_v2 列还没建？
├─ 集合是新建的 / 数据少
│   └─ 建表时就带上 vec_v2 + 索引（首选，零风险）
├─ 集合已上线、有大量数据
│   ├─ 试「官方事后补列」：add_collection_field(nullable=True)
│   │     → create_index → 分批评分回填（先小批量验证）
│   │   ├─ 索引正常建成 → 走 7.3.3 两列并存 + 切 anns_field ✓
│   │   └─ 卡 InProgress → drop_index / 删列重来，还卡 → 见下
│   └─ 兜底：新集合 + 双写切换（导出重插 / milvus-backup）✓
```

**证据来源**

- 官方 [Alter Collection Schema](https://milvus.io/docs/add-fields-to-an-existing-collection.md)：加向量字段受支持（2.6.18+）、必须 `nullable=True`、已有实体为 `NULL`、`NULL` 行被检索跳过、示例顺序 = `add_collection_field(nullable=True)` → `create_index` → 回填。
- v3.0.2 实测：本文写作时在 `milvusdb/milvus:v3.0.2`（digest `sha256:5f13bf88…`）上复现，datacoord 原始日志 `state=InProgress, totalRows=200, indexRows=0, pendingIndexRows=200`，即「零行被索引、永不推进」。
- 相关 PR：`#52612`（nullable vector 的 ID 映射移进索引层，v3.0.1）、`#53363`（nullable vector 无 binlog 导入修复，v3.0.2）、`#54020`（`IS NULL`/`IS NOT NULL` 过滤，v3.0.3）。这三条都没有明文声明「加列后建索引卡 InProgress」是已修复的 bug。

> 返回：[7.3　数据模型与检索增强](./Milvus3.0数据模型与检索增强)
