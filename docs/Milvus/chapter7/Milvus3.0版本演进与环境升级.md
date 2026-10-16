# Chapter 7.1 Milvus 3.0 版本演进与环境升级

<!-- EDITOR:BEGIN 总说明（审核通过后，用 .temp/strip_editor_notes.py 一键删掉所有 EDITOR 块）
   状态：草稿。未写入 docs/.vitepress/config.js 的 sidebar / nav，读者不可见。
   挂载片段（审核通过后贴到 '/Milvus/' 数组尾部）：
   { text: 'Chapter 7 · Milvus 3.0x 新特性', items: [
     { text: '版本演进与环境升级', link: '/Milvus/chapter7/Milvus3.0版本演进与环境升级' },
     { text: '外部集合与快照', link: '/Milvus/chapter7/Milvus3.0外部集合与快照' },
     { text: '数据模型与检索增强', link: '/Milvus/chapter7/Milvus3.0数据模型与检索增强' },
     { text: '索引与写入引擎', link: '/Milvus/chapter7/Milvus3.0索引与写入引擎' },
     { text: '端到端实战', link: '/Milvus/chapter7/Milvus3.0端到端实战' },
   ]},
   编号约定：大章节 = Chapter 7，本子章节 = 7.1（沿用 Chapter 6《Milvus底层架构详解》用 "Chapter 1.1 / 1.1.x" 的先例）。
   范围红线：只讲 Milvus 本体（Milvus + pymilvus + 官方 3.0.x 文档）。对比只允许发生在 Milvus 2.6 ↔ 3.0 之间。
   统一术语：外部集合（External Collection）、快照（Snapshot）、存储引擎 V3（Storage V3 / Loon）、数组结构体（StructArray）、写前日志（Woodpecker WAL）。
EDITOR:END -->

## 7.1.1 学习目标

<!-- EDITOR:BEGIN 7.1.1
🎯 要求：动词起头 4–6 条，每条一句。建议覆盖：说出 3.0 的定位转折 / 完成 3.0 部署与 2.6 原地升级 / 正确打开两个默认关闭的开关 / 判断某特性在 Lite·Standalone·集群是否可用。
⚠️ 别写成"掌握所有 3.0 新特性"，本章只管版本与环境认知。
EDITOR:END -->

## 7.1.2 Milvus 3.0 的定位与演进脉络

### 7.1.2.1 一句话转折

<!-- EDITOR:BEGIN 7.1.2.1
🎯 要求：3–5 行讲完，不铺陈。落点=3.0 的主叙事不是"更快"，而是"数据不再需要搬进库里"。
🔗 参考：https://milvus.io/docs/release_notes.md （v3.0.0 开头两段）｜ https://github.com/milvus-io/milvus/releases/tag/v3.0.0
⭐ 重点：GA 时间 2026-07-29；叙事三段——Storage V3 打底座 → 外部集合 + 快照 两种新范式 → schema 在线演进 / 超大规模索引 / WAL 解耦 补齐工程闭环。
🚫 去重边界：Chapter 1（1.3.1 Milvus 的定位与发展）已讲"Milvus 是什么"，此处只写 2.x→3.0 的转折，不复述发展史。
EDITOR:END -->

### 7.1.2.2 beta 与 GA 的两步节奏

<!-- EDITOR:BEGIN 7.1.2.2
🎯 要求：一小段 + 一张两列表（beta 立骨架 / GA 补闭环）。
🔗 参考：https://milvus.io/docs/release_notes.md#v30-beta ｜ https://github.com/milvus-io/milvus/releases/tag/v3.0.0 （"What's new in 3.0.0 (since 3.0-beta)"）
⭐ 重点：beta 首次出现＝外部集合、快照、TTL、ORDER BY、查询聚合、MinHash、EmbList+DiskANN、Woodpecker、Storage V3；GA 补＝外部集合工作流完整、在线加列/回填/删列、schema 与索引同步演进、重排函数链、聚合上搜索路径（分面）、稀疏索引换代、超大规模优化、索引引擎直通、Woodpecker 独立部署。
🚫 去重边界：不要在此逐条解释特性做什么（7.2–7.4 各自展开）。
EDITOR:END -->

### 7.1.2.3 四条兼容性红线

<!-- EDITOR:BEGIN 7.1.2.3
🎯 要求：用 > ⚠️ **注意** 逐条列。这四条是全大章的公共前提，后续子章节会反复回指，务必写准。
🔗 参考：https://milvus.io/docs/release_notes.md （"Compatibility and behavior notes"）
⭐ 重点（照此口径，数值勿改）：
1) Storage V3（Loon）默认关闭，快照、TEXT 依赖它，需手工开 common.storage.useLoonFFI；后续版本才会默认启用；
2) 2.6 → 3.0 兼容且可回滚，但一旦启用/使用了改变序列化格式的特性（如 Storage V3），回滚窗口永久关闭；
3) 新引入的索引算法默认不生效，需 dataCoord.targetVecIndexVersion=10、dataCoord.targetScalarIndexVersion=4；
4) GPU 镜像迁至 CUDA 12.9，不再保留 Ubuntu 20.04 兼容。
🚫 去重边界：Chapter 1 的"系统要求/硬件建议"不重写。
EDITOR:END -->

## 7.1.3 版本矩阵与客户端门槛

### 7.1.3.1 服务端与 SDK 对照表

<!-- EDITOR:BEGIN 7.1.3.1
🎯 要求：只放一张表，不写步骤。列＝Milvus / pymilvus / Node.js SDK / Java SDK / Go SDK。
🔗 参考：https://github.com/milvus-io/milvus/releases/tag/v3.0.0 ｜ https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/limitations.md
⭐ 重点：v3.0.0 官方配套＝Milvus 3.0.0 / pymilvus 3.0.1 / Node.js 3.0.3 / Java 3.0.5 / Go 3.0.0；pymilvus 现行 3.0.2。
⚠️ 关键提醒：服务端 3.0.x 与 Lite 的 3.1/3.2.x 是两套号段，别写成"Milvus Lite 3.0"。
🚫 去重边界：Chapter 1（1.3.3）已对比过 Lite/Standalone/集群的差异定位，此处只做版本与 SDK 对应。
EDITOR:END -->

### 7.1.3.2 Python 版本门槛

<!-- EDITOR:BEGIN 7.1.3.2
🎯 要求：两三行 + 一条 > 📌 **前置条件**。
🔗 参考：https://github.com/milvus-io/pymilvus/releases ｜ https://pypi.org/project/milvus-lite/ ｜ https://milvus.io/docs/install-pymilvus.md
⭐ 重点：pymilvus 3.x 要求 Python ≥3.9（3.8 停支持）；milvus-lite 3.2.x 要求 Python ≥3.10。
⚠️ 不要改动 Chapter 4 现有 pymilvus==2.5.4 的 pin；建议本章加一句"新旧章节各用独立 venv"。
EDITOR:END -->

## 7.1.4 环境升级实操

### 7.1.4.1 全新部署 3.0 单机版

<!-- EDITOR:BEGIN 7.1.4.1
🎯 要求：一段 bash/powershell + 一句"预期输出"。只给**差异片段**，不要重贴整份 compose。
🔗 参考：https://milvus.io/docs/install_standalone-docker.md ｜ https://milvus.io/docs/configure-docker.md ｜ https://milvus.io/docs/dynamic_config.md ｜ https://milvus.io/docs/tune_consistency.md
⭐ 重点：镜像 tag 换 3.0.x；standalone 配置文件与 volume 路径差异项；服务健康检查方式。
⚠️ 待你浏览器复核：本机打不开 milvus.io，官方 compose 的确切镜像 tag 请以安装页现值为准。
🚫 去重边界：装 Attu、基础 docker 流程、停止与卸载在 Chapter 1（1.5.2–1.5.4）已写，一律不重复。
EDITOR:END -->

### 7.1.4.2 从 2.6 原地升级

<!-- EDITOR:BEGIN 7.1.4.2
🎯 要求：写成检查单式步骤（升级前备份 → 升级 → 验证 → 再决定是否开新开关）。
🔗 参考：https://milvus.io/docs/upgrade_milvus_standalone-docker.md ｜ https://milvus.io/docs/upgrade_milvus_cluster-helm.md ｜ https://milvus.io/docs/upgrade_milvus_standalone-operator.md ｜ https://milvus.io/docs/upgrade_milvus_cluster-operator.md
⭐ 重点：升级前必看 preupgrade 检查项；2.6→3.0 可回滚的前提是没有启用改序列化格式的特性；升级后索引版本/元数据的处理。
⚠️ 建议配一张决策表：要用快照/TEXT？→ 要开 Loon？→ 开了就不能回滚 → 先在实验环境演练。
EDITOR:END -->

### 7.1.4.3 打开存储引擎 V3（Loon）

<!-- EDITOR:BEGIN 7.1.4.3
🎯 要求：milvus.yaml 片段（语言标 yaml）+ 一行验证方式 + 一句"预期输出"。
🔗 参考：https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/release_notes.md （Compatibility notes）
⭐ 重点：键名 common.storage.useLoonFFI；默认 false。
🚫 去重边界：Loon 的原理放到 7.2.2，本章只讲"怎么开、开了意味着什么"。
EDITOR:END -->

### 7.1.4.4 打开新版索引开关

<!-- EDITOR:BEGIN 7.1.4.4
🎯 要求：两个键一起给（yaml 片段），并写明"不开会怎样"——索引建了却仍走老路径，而且**不报错**（高频坑，务必上 > ⚠️ **注意**）。
🔗 参考：https://milvus.io/docs/release_notes.md ｜ https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index_selection.md
⭐ 重点：dataCoord.targetVecIndexVersion=10；dataCoord.targetScalarIndexVersion=4；改后需重启生效。
🚫 去重边界：Chapter 2（2.3.3 索引类型 / 2.3.7 索引参数）已讲经典索引，此处不重讲。
EDITOR:END -->

### 7.1.4.5 客户端 pymilvus 升级

<!-- EDITOR:BEGIN 7.1.4.5
🎯 要求：一行安装命令 + 一张"后续子章节会用到哪些新客户端能力"预告表（只列名，不讲用法）。
🔗 参考：https://milvus.io/docs/install-pymilvus.md ｜ https://milvus.io/docs/using_milvusclient.md ｜ https://github.com/milvus-io/pymilvus/releases
⭐ 重点：pip install -U "pymilvus>=3.0.1,<3.1"。3.0.x 客户端线索：alter_collection_schema()（在线改 schema）、FunctionChain 与 XGBoost 表达式辅助、Bloom/Roaring bitmap 构建器、搜索聚合与 group_by_fields、compaction 生命周期观测、dump_messages / get_replicate_info、DataType.TEXT、alter_role/describe_role。
⚠️ 每个接口名写前用官方 API 页或 releases 原文核对参数拼写，勿凭记忆。
EDITOR:END -->

## 7.1.5 三种形态的支持边界

### 7.1.5.1 特性 × 形态支持矩阵

<!-- EDITOR:BEGIN 7.1.5.1
🎯 要求：一张表。行＝外部集合 / 快照 / Storage V3 / TEXT / StructArray / 服务端排序聚合与分面 / Function Chain / 新版稀疏索引 / Woodpecker / Force Merge；列＝Lite / Standalone / 集群 / 备注。
🔗 参考：https://milvus.io/docs/milvus_lite.md ｜ https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/woodpecker.md ｜ https://pypi.org/project/milvus-lite/
⭐ 重点：Lite 是嵌入式引擎，其 snapshot 概念（钉住不可变数据/增量/索引文件并保存 schema 与 manifest 副本）≠ 服务端快照特性，必须显式区分，否则读者会拿 Lite 去试外部集合。
⚠️ 官方未逐项明示 Lite 的 3.0 能力边界：不确定的格子写"待验证"，不要猜。
🚫 去重边界：Chapter 1（1.3.3）与 Chapter 6《Milvus Lite部署与应用》已讲形态定位与 Lite 装法，此处只填这张新表。
EDITOR:END -->

### 7.1.5.2 本教程的实验环境选择

<!-- EDITOR:BEGIN 7.1.5.2
🎯 要求：给读者一句"照我做"的环境建议；明确哪些小节必须有对象存储/集群才跑得动（外部集合、快照备份、Woodpecker 独立部署），并给替代观察方式（看配置、看 Attu、看日志）。
⚠️ 跑不动的一律标注前置条件，不要假装能跑。
EDITOR:END -->

## 7.1.6 本章小结

<!-- EDITOR:BEGIN 7.1.6
🎯 要求：沿用你既有模板——`本章核心掌握：` + 3–5 条 + 一句衔接（"7.2 起在此环境上实操新特性"）。
⚠️ 本章只保留三张表（版本矩阵 / SDK 对照 / 支持矩阵）+ 升级操作，特性原理一律不写，防止与 7.2–7.4 重复。
EDITOR:END -->
