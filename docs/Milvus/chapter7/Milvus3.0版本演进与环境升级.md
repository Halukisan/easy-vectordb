# Chapter 7.1 Milvus 3.0 版本演进与环境升级

## 7.1.0 本章导读：3.0 改的不是功能清单，是默认做法

如果你只读过前面 2.x 的内容，3.0 很容易被误解成“又加了一批索引和参数”。实际上它同时改了三件事的默认做法：

1. **数据放在哪**：Storage V3 把底层换成“对象存储 + manifest”，并由此长出了外部集合（数据不动、Milvus 来读）和快照（按段引用、秒级创建）。
2. **一次请求能算什么**：聚合、分面、标量排序、重排从“客户端多次调用”变成“服务端一次下推”。
3. **写入与索引怎么落地**：Woodpecker 取代外部消息队列，SINDI / ADAPT 等新索引把稀疏与标量检索拉回同一套执行链路。

这三条是本章的主线：7.2 讲第 1 条，7.3 讲第 2 条，7.4 讲第 3 条，7.5 把三条串成一条可回滚的链路。

### 三条阅读路线

三条路线按目的分：

- **只想判断要不要升级**：走 7.1.2 → 7.1.3 → 7.2.3 → 7.3.5 → 7.5.6，大约 30 分钟；
- **要动手跑一遍新能力**：先读 7.1.5 把环境定下来，再做 7.2 与 7.3，约 2 小时；
- **在做上线评估**：按 7.1.4 过一遍开关与升级操作，再看 7.4.6 的写入引擎形态与 7.5.5 的代价观察，需要半天。

### 读之前需要什么

- 已经能建集合、建索引、做一次向量检索（第 2、3 章）；
- 知道 segment / compaction / WAL 大概是什么（第 6 章）；
- 一个能跑 Milvus 3.0.x 的环境。**注意：Storage V3 默认关闭**，外部集合、快照、TEXT、函数派生字段都依赖它；没有对象存储的读者，按 7.1.5.2 的“替代观察方式”做，不要硬跑。

### 版本基线

本章以 **Milvus v3.0.2（2026-09-20）** 为写作基线，首发 GA 为 v3.0.0（2026-07-29）。3.0.1 补充了 REST v2 快照管理、扩展 rerank、TEXT 的 Go/REST 支持；3.0.2 以检索 / 索引 / 外部集合修复为主。正文涉及版本差异处会单独标注。

## 7.1.1 学习目标

本节只划定"本章要让你会什么"，不展开任何特性原理——原理从 7.2 起逐个展开。学完本章，你应当能：

- **说出** 3.0 的定位转折：主叙事不是"更快"，而是"数据不再需要搬进库里"；
- **说清** 2.6 → 3.0 的四条兼容红线，尤其是"启用改序列化格式的特性后，回滚窗口永久关闭"这一条；
- **完成** 3.0 单机部署，或把已有 2.6 单机原地升级到 3.0；
- **打开** 两个默认关闭的开关——存储引擎 V3（Loon）与新版索引版本，并说清"不开"会怎样；
- **判断** 某个 3.0 特性在 Lite / Standalone / 集群三种形态下是否可用、需要什么前置条件。

## 7.1.2 Milvus 3.0 的定位与演进脉络

### 7.1.2.1 一句话转折

3.0 的主叙事不是"更快"，而是"数据不再需要搬进库里"。在 2.x 时代，湖上的表（Parquet / Lance / Iceberg）要参与检索，得先写一段 ETL 把它们拷进 Milvus——多付一份存储，多维护一条同步管道，数据治理权也被挪出了团队。3.0 用存储引擎 V3（Storage V3）把底层换成"manifest + 对象存储"，外部集合（External Collection）直接引用湖文件原地建索引、原地查询，快照（Snapshot）按段引用给出时点只读视图。数据留在原地，Milvus 只接管 schema、索引与查询执行。

### 7.1.2.2 beta 与 GA 的两步节奏

3.0 分两步走完：2026-05-09 的 3.0-beta 先把"湖仓原生"骨架立起来，2026-07-29 的 v3.0.0 GA 再把工作流补齐成可上线闭环。理解这个节奏，能避免把两条 release notes 混在一起读。

**beta 立骨架**：外部集合、快照、Query / Search 的 ORDER BY、查询聚合（COUNT / SUM / AVG / MIN / MAX + group-by）、可空向量、Entity TTL、文件资源（词典 / 同义词 / 停用词）、MinHash DIDO、EmbList + DiskANN、Force Merge、Storage V3——这些范式性能力都在 beta 首次出现。

**GA 补闭环**：外部集合工作流补全（外部列可喂函数输出字段、refresh 支持增量 schema 演进）、在线加列 / 回填 / 删列、StructArray 与分面检索补齐检索引擎、Function Chain 重排、稀疏索引换代（SINDI / Block-Max WAND / Block-Max MaxScore）、FAISS 索引直通、TEXT 长文本、Vortex 与 Lance 格式支持、Woodpecker 独立部署。

一句话总结：beta 回答"能不能"，GA 回答"能不能上生产"。

### 7.1.2.3 四条兼容性红线

这四条来自 v3.0.0 release notes 的 "Compatibility and behavior notes"，是全章的公共前提，后文会反复回指，数值与措辞一个字都不要改：

> ⚠️ **注意**
>
> 1. **存储引擎 V3（Loon）默认关闭**。依赖它的能力（快照、TEXT 字段）需要手工打开 `common.storage.useLoonFFI`；官方口径是"后续版本才会默认开启"。
> 2. **2.6 → 3.0 兼容、可回滚**，但一旦启用或使用了会改变序列化格式的特性（例如 Storage V3），回滚就再也回不去——回滚窗口永久关闭。
> 3. **新索引版本默认不生效**。新引入的索引算法需要手工把目标版本抬到 `dataCoord.targetVecIndexVersion=10`、`dataCoord.targetScalarIndexVersion=4` 才会生效；后续版本才会默认开启。
> 4. **GPU 镜像迁移到 CUDA 12.9**，不再保留 Ubuntu 20.04 的 GPU 兼容。

其中第 2 条是全章最贵的一句话：要不要开 Loon，本质是"要不要永久放弃回滚 2.6 的能力"。做决定前，先在实验环境把 7.1.4 的升级与开关流程完整演练一遍。

## 7.1.3 版本对照与客户端门槛

### 7.1.3.1 服务端与 SDK 对照说明

Milvus v3.0.0（2026-07-29 GA）官方配套的 SDK 版本如下，以 GitHub 上 v3.0.0 release 页的版本表为准：

- **Python 客户端（pymilvus）**：配套 3.0.1，写作时最新为 3.0.2；
- **Node.js**：3.0.3；
- **Java**：3.0.5；
- **Go**：3.0.0。

选型时有两点要分清：

- 服务端是 **3.0.x** 一条轨；**Milvus Lite 走 3.1 / 3.2.x 另一条轨**，两者号段不同——不要写成"Milvus Lite 3.0"。
- SDK 版本号不必逐位追平服务端，但官方建议尽量对齐（release notes 每版都附带一份配套矩阵）。判断"要不要上 3.0"的唯一依据是服务端是否已在 3.0.x，而不是 pymilvus 版本号的尾数。

### 7.1.3.2 Python 版本门槛

3.0 的 Python 客户端有两条硬门槛，我已在本机 PyPI 上实跑核过：

| 包 | 版本 | `requires_python` |
|---|---|---|
| pymilvus | 3.0.2 | `>=3.9` |
| milvus-lite | 3.2.2 | `>=3.10` |

> 📌 **前置条件**
> 官方安装页 install-pymilvus 仍写着"Python 3.7+"，那是滞后口径，以 PyPI 上各包 wheel 的 `requires_python` 元数据为准。**pymilvus 3.x 要求 Python ≥ 3.9（3.8 停止支持）；milvus-lite 3.x 要求 Python ≥ 3.10。** 如果你还在用 ≤3.9 的环境，给新章节单独建 venv：旧章节（Chapter 4 的 `pymilvus==2.5.4`）与新章节（`pymilvus>=3.0.1`）不要共用同一个 Python 环境。

## 7.1.4 环境升级实操

### 7.1.4.1 全新部署 3.0 单机版

3.0 单机版与 2.6 的差异只有一圈，其余安装步骤（装 Docker、准备目录）沿用 Chapter 1（1.5.2–1.5.4）。两版单机的默认形态对比如下：

| 项 | Milvus 2.6 单机 | Milvus 3.0 单机 |
|---|---|---|
| 镜像 tag | `milvusdb/milvus:v2.6.x` | `milvusdb/milvus:v3.0.x` |
| 消息队列 | RocksMQ（嵌入式，默认；Pulsar / Kafka 可选） | Woodpecker（嵌入式 WAL，默认；也可独立部署） |
| 依赖容器 | etcd + MinIO | etcd + MinIO（compose 自带，未变） |
| 存储引擎 | Storage V2 | Storage V2 默认，Storage V3 需开 Loon |

做法：下载官方 compose，无需改任何依赖声明，直接拉起——compose 自带 etcd 与 MinIO 两个依赖容器，Woodpecker 作为 WAL 内置在 standalone 中。

```bash
# 下载官方 standalone compose；URL 里的版本号换成本机要装的版本
wget https://github.com/milvus-io/milvus/releases/download/v3.0.2/milvus-standalone-docker-compose.yaml -O docker-compose.yml
# 启动
docker compose up -d
# 健康检查：容器状态 + gRPC 端口回连
docker compose ps
python3 -c "from pymilvus import MilvusClient; print(MilvusClient(uri='http://localhost:19530').list_collections())"
```

预期输出：`docker compose ps` 中 `standalone` 为 running / healthy；最后一行打印 `[]`（空集合列表，说明 gRPC 已连通）。也可打开 WebUI `http://127.0.0.1:9091/webui/` 确认实例健康。

> 📌 **前置条件**
> 镜像 tag 需以官方安装页现值为准：官方页面用 `&#123;&#123;var.milvus_release_tag&#125;&#125;` 占位，本机无法直连 milvus.io 与 Docker Hub 复核，上面的 `v3.0.2` 是本章写作基线（见 7.1.0）。若你拉取时官方已推进到 v3.0.3，替换为对应 tag，其余步骤不变；2.6 单机的 MQ 差异来自 Woodpecker 的定位表述，具体部署形态的历史默认值未逐版核对。

### 7.1.4.2 从 2.6 原地升级

把 2.6 单机原地升到 3.0，官方验证过的路径是 Docker Compose **只换镜像 tag、其余一律不动**。整条链是：备份 → 只改镜像 → 拉起 → 验证 → 再决定开不开新开关。

**第 1 步：备份并确认现状**

```bash
cp docker-compose.yml docker-compose-before-upgrade.yml
docker compose ps
```

预期输出：`docker compose ps` 显示 2.6 各服务为 running，元数据与数据目录完好。

**第 2 步：只改镜像 tag**

compose 里只动 `standalone` 服务这一行，etcd、对象存储、数据卷、依赖版本全部保持原样：

```yaml
services:
  standalone:
    image: milvusdb/milvus:v3.0.2   # 2.6 → 3.0 只动这一行
```

```bash
docker compose pull standalone
docker compose up --detach standalone
```

预期输出：pull 下载 v3.0.2 镜像；`up -d` 只重建 standalone 容器，etcd 与对象存储容器继续运行，数据目录复用到新版本。

**第 3 步：验证**

```bash
docker compose ps
docker compose images standalone
docker compose logs --tail 100 standalone
```

预期输出：`standalone` healthy、镜像 tag 为 `milvusdb/milvus:v3.0.2`，已有集合仍可查询与检索。**这步通过之前，不要开启任何 3.0 专属特性。**

**第 4 步：再决定开不开新开关**——回到 7.1.4.3 / 7.1.4.4，分别评估 Loon 与新版索引。

> ⚠️ **注意**
> 官方明确：这条流程**不验证"把镜像改回 2.6.x 做回滚"**——一旦 3.0 写入过数据，仅靠换镜像回滚可能读不了新状态。真出问题，用第 1 步的元数据与数据备份做恢复，恢复方案先在非生产环境演练过。判断"要不要冒险开 Loon"只一句话：要用快照或 TEXT 吗？要 → 必须开 Loon；开了 → 回滚窗口关闭。

### 7.1.4.3 打开存储引擎 V3（Loon）

开 Loon 是启用快照、TEXT 等能力的入口，默认关闭，改一个键：

```yaml
common:
  storage:
    useLoonFFI: true
```

改完让配置生效：官方把该项标为 refreshable（见 `configure_common.md`），应通过部署方式支持的配置更新流程下发，改完在 Milvus WebUI（`http://127.0.0.1:9091/webui/`）里核对当前值；同时官方 Docker 配置页又写明"当前版本所有参数都要重启后才生效"，两处口径并存，稳妥做法是改完重启并复核。

验证方式：在 WebUI 里查 `common.storage.useLoonFFI` 的当前值是否为 `true`；若走 etcdctl，key 是 `<rootPath>/config/common/storage/useLoonFFI`，其中 rootPath 依安装方式而异（Helm 与部分部署默认 `by-dev`，standalone embed etcd 以其自身配置为准，此处仅示意）：

```bash
etcdctl get by-dev/config/common/storage/useLoonFFI
```

预期输出：返回 `true`。此后新写入与 compaction 产物走 Storage V3，既有数据在其被后台 compaction 重写前保持原布局，过渡期两种布局都能读。

> 📌 **前置条件**
> Loon 的原理放到 7.2.2，本页只解决"怎么开、开了意味着什么"。开启后后台 compaction 会把符合条件的老数据逐步迁到 V3 布局；**关掉 `useLoonFFI` 不会把已有 V3 数据同步转回去，也不等于让降级变安全**——这正是红线第 2 条的物理含义。

### 7.1.4.4 打开新版索引开关

两个键一起抬，少一个都不算开（官方口径：两个都要手工设）：

```yaml
dataCoord:
  targetVecIndexVersion: 10
  targetScalarIndexVersion: 4
```

> ⚠️ **注意**
> 不设这两个键**不报错**——索引照样能建，只是仍走旧路径、用旧算法。这是全章最高频的坑：你建了 SINDI / Block-Max WAND 等新索引、以为生效了，实际上还落在旧实现上。改完这组键要重启，并对**已有段重建索引**后验证效果（这两个键目前只见于 release notes 的 Compatibility notes，未收录进 configure_datacoord 配置手册，按静态配置处理）。

预期输出：重建索引后，新段按目标版本（向量 10 / 标量 4）落盘；排查时通过索引版本或行为差异确认确实走了新实现。

> 💡 **版本说明**
> 官方口径是"后续版本会默认开启"这两个开关。现阶段被锁在 opt-in，7.4.2.2 会回指这里当作前提。经典索引本身（HNSW / IVF 等）在 Chapter 2 已讲，这里不重复。

### 7.1.4.5 客户端 pymilvus 升级

```bash
pip install -U "pymilvus>=3.0.1,<3.1"
```

预期输出：`pip show pymilvus` 显示 Version 在 3.0.x（写作时 3.0.2）。

> 📌 **前置条件**
> Python ≥ 3.9（见 7.1.3.2），且与旧章节的 `pymilvus==2.5.4` 分 venv。

升级后，后续子章节会用到这些 3.0 客户端能力（这里只列名与出处，用法各自展开）：

- **在线改 schema**：`add_collection_field()` / `add_collection_struct_field()` / `add_function_field()` / `drop_collection_field()`（见 add-fields-to-an-existing-collection、alter-collection-field 页）；
- **Function Chain 重排与打分表达式**：函数输出字段挂到 Function、在打分表达式里做重排；
- **Bloom / Roaring 过滤**：`build_bloom_filter()` / `build_roaring_bitmap()` + `membership_match()`（membership-match 页）；
- **搜索聚合 / 分组 / 分面**：`group_by_fields` 透传与聚合参数（search-aggregation 页）；
- **长文本**：`DataType.TEXT`（text 页）；
- **FAISS 索引直通**：`faiss_index_name`（faiss 页）。

> ⚠️ **注意**
> 另有几个 3.0 客户端能力在编辑线索里被提及、但本机缓存中找不到官方页面原文来核对拼写，暂标"待验证"，使用前先查 pymilvus 3.0.x API 参考：compaction 生命周期观测（`dump_messages()` / `get_replicate_info()`）、权限管理（`alter_role()` / `describe_role()`）。

## 7.1.5 三种形态的支持边界

### 7.1.5.1 各特性在三种形态下的支持情况

> 💡 **版本说明**
> Milvus Lite 是嵌入式引擎，版本序列走 3.1 / 3.2.x，与服务端 3.0.x 不同轨。它文档里的"snapshot"指"嵌入式引擎钉住不可变数据 / 增量 / 索引文件并保存 schema 与 manifest 副本"的迁移机制，**不是**服务端的快照特性——别拿 Lite 去试外部集合与快照。

各特性在三种形态下的支持情况（Lite 一列以 milvus_lite 官方页现文为准，官方没写明的标"待验证"）：

| 特性 | Lite | Standalone | 集群 | 备注 |
|---|---|---|---|---|
| 外部集合（External Collection） | 待验证 | 支持 | 支持 | 需对象存储 + Loon；只读、靠 refresh 追平 |
| 快照（Snapshot） | 无（Lite 的 snapshot 是另一概念） | 支持 | 支持 | 依赖 Storage V3（Loon 开） |
| 存储引擎 V3（Loon） | 待验证 | 支持（默认关） | 支持（默认关） | 需 `common.storage.useLoonFFI` |
| TEXT 长文本 | 待验证 | 支持 | 支持 | 需 Loon |
| StructArray 数组结构体 | 待验证 | 支持 | 支持 | 有限制（见 structarray-limits 页） |
| 排序 / 聚合 / 分面 | 待验证 | 支持 | 支持 | ORDER BY、COUNT/SUM/AVG/MIN/MAX、group-by |
| Function Chain 重排 | 待验证 | 支持 | 支持 | — |
| 新版稀疏索引（SINDI 等） | N/A（Lite 仅 FLAT 索引） | 支持 | 支持 | 需抬 `targetVecIndexVersion` |
| Woodpecker 部署模式 | N/A | embedded（默认） | service（可独立部署） | 详见 7.5 |
| Force Merge | 待验证 | 支持 | 支持 | compaction 类操作 |

> 💡 **版本说明**
> 经典索引里 Lite 只会跑 FLAT（无论集合声明什么索引类型）；Lite 支持稀疏向量 / 多向量 / 混合检索，但**不支持 partition、用户/角色、alias**。这些是 Lite 的固定边界，与 3.0 无关，遇到别再逐个踩。

### 7.1.5.2 本教程的实验环境选择

照我做：用 **Standalone（Docker Compose）+ 嵌入式 Woodpecker + 嵌入式 etcd** 作为本章默认环境。这是唯一"无需额外依赖、又能覆盖大多数 3.0 特性"的形态；除非你要跑外部集合、快照、Woodpecker 独立部署，否则不必上对象存储与集群。

必须额外加依赖的小节：

- **外部集合 / 快照 / 湖仓原生**（7.2 起）：需要对象存储（MinIO / S3 等）+ Loon；
- **Woodpecker 独立部署**（7.5）：需要集群形态（service 模式）。

跑不动时的替代观察方式（不要硬跑、更不要伪造输出）：

- **看配置**：7.1.4.3 / 7.1.4.4 的开关先看 yaml 落盘、再到 WebUI / etcdctl 里读当前值；
- **看 WebUI / Attu**：`http://127.0.0.1:9091/webui/` 看集合、段、索引与 compaction 状态；
- **看日志**：`docker compose logs --tail 100 standalone` 看 Woodpecker、compaction 与脏数据处理。

## 7.1.6 本章小结

本章核心掌握：

- **版本对照**：3.0 是"湖仓原生"重构，不是性能增强版；2.6 → 3.0 可回滚，前提是没启用改序列化格式的特性。
- **SDK 对照**：服务端 3.0.x 配套 pymilvus 3.0.1+（Python ≥ 3.9）、Node.js 3.0.3、Java 3.0.5、Go 3.0.0；Lite 走 3.1 / 3.2.x 另一套号段。
- **三条形态边界**：Lite 只做本地快速验证（仅 FLAT、无分区/角色）；Standalone 单机够用；集群才谈规模化与独立 Woodpecker。
- **两个默认关闭的开关**：Loon（`common.storage.useLoonFFI`）与新版索引（`targetVecIndexVersion=10` / `targetScalarIndexVersion=4`）；开 Loon 前先想清楚"要不要保留回滚 2.6 的能力"。

从 7.2 起，我们就在这套环境上实操新特性：先看外部集合与快照怎么把数据留在原地，再谈 schema 与检索增强。
