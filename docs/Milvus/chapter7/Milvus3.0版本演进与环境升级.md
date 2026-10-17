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
   篇幅配比：讲 55% ／ 练 35% ／ 读 10%；全章约 36 页（500 字/页）。练习只给验收标准，不给答案。
   共通规矩（每节都适用）：
     1) 开头一句回答"本节解决什么"，第二句说"不解决什么"；
     2) 每段代码前说清前提、后说清预期输出，贴出的代码必须能跑；
     3) 数字只写自己跑出来的，未跑写"待补测"；只有官方原文明确写的才能称"官方口径"；
     4) 限制单独成块，不混进正文；已知限制不准用"暂不支持"一句带过；
     5) 只做 2.6 与 3.0 的对比，不与其他检索库比较；
     6) 术语按本文件头统一，英文原词首次出现加括号。
   注释怎么用：每节标题下一条 EDITOR 注释，含【本节要写什么】【参考资料】【必须注意】【写完自检】【删除】；
     动笔前先读【参考资料】，写完对着【写完自检】验收，定稿后删掉整块注释（或跑 .temp/strip_editor_notes.py）。
EDITOR:END -->

## 7.1.0 本章导读：3.0 改的不是功能清单，是默认做法

如果你只读过前面 2.x 的内容，3.0 很容易被误解成“又加了一批索引和参数”。实际上它同时改了三件事的默认做法：

1. **数据放在哪**：Storage V3 把底层换成“对象存储 + manifest”，并由此长出了外部集合（数据不动、Milvus 来读）和快照（按段引用、秒级创建）。
2. **一次请求能算什么**：聚合、分面、标量排序、重排从“客户端多次调用”变成“服务端一次下推”。
3. **写入与索引怎么落地**：Woodpecker 取代外部消息队列，SINDI / ADAPT 等新索引把稀疏与标量检索拉回同一套执行链路。

这三条是本章的主线：7.2 讲第 1 条，7.3 讲第 2 条，7.4 讲第 3 条，7.5 把三条串成一条可回滚的链路。

### 三条阅读路线

| 你的目的 | 读这些 | 大约耗时 |
|---|---|---|
| 快速判断“要不要升级” | 7.1.2 → 7.1.3 → 7.2.3 → 7.3.5 → 7.5.6 | 30 分钟 |
| 动手跑一遍新能力 | 7.1.5 → 7.2 → 7.3 | 2 小时 |
| 做上线评估 | 7.1.4 → 7.4.6 → 7.5.5 | 半天 |

### 读之前需要什么

- 已经能建集合、建索引、做一次向量检索（第 2、3 章）；
- 知道 segment / compaction / WAL 大概是什么（第 6 章）；
- 一个能跑 Milvus 3.0.x 的环境。**注意：Storage V3 默认关闭**，外部集合、快照、TEXT、函数派生字段都依赖它；没有对象存储的读者，按 7.1.5.2 的“替代观察方式”做，不要硬跑。

### 版本基线

本章以 **Milvus v3.0.2（2026-09-20）** 为写作基线，首发 GA 为 v3.0.0（2026-07-29）。3.0.1 补充了 REST v2 快照管理、扩展 rerank、TEXT 的 Go/REST 支持；3.0.2 以检索 / 索引 / 外部集合修复为主。正文涉及版本差异处会单独标注。

## 7.1.1 学习目标

<!-- EDITOR:BEGIN 7.1.1
【本节要写什么】动词起头 4–6 条，每条一句。建议覆盖：说出 3.0 的定位转折 / 完成 3.0 部署与 2.6 原地升级 / 正确打开两个默认关闭的开关 / 判断某特性在 Lite·Standalone·集群是否可用。
【必须注意】
・️ 别写成"掌握所有 3.0 新特性"，本章只管版本与环境认知。
【写完自检】4–6 条目标都能在本页找到对应节；读者能据此判断要不要读本章。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.1.2 Milvus 3.0 的定位与演进脉络

### 7.1.2.1 一句话转折

<!-- EDITOR:BEGIN 7.1.2.1
【本节要写什么】3–5 行讲完，不铺陈。落点=3.0 的主叙事不是"更快"，而是"数据不再需要搬进库里"。
【先读这些】https://milvus.io/docs/release_notes.md （v3.0.0 开头两段）｜ https://github.com/milvus-io/milvus/releases/tag/v3.0.0
【必须注意】
・首发 GA 2026-07-29，最新补丁 v3.0.2（2026-09-20，写作基线）；叙事三段——Storage V3 打底座 → 外部集合 + 快照 两种新范式 → schema 在线演进 / 超大规模索引 / WAL 解耦 补齐工程闭环。
・Chapter 1（1.3.1 Milvus 的定位与发展）已讲"Milvus 是什么"，此处只写 2.x→3.0 的转折，不复述发展史。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.2.2 beta 与 GA 的两步节奏

<!-- EDITOR:BEGIN 7.1.2.2
【本节要写什么】一小段 + 一张两列表（beta 立骨架 / GA 补闭环）。
【先读这些】https://milvus.io/docs/release_notes.md#v30-beta ｜ https://github.com/milvus-io/milvus/releases/tag/v3.0.0 （"What's new in 3.0.0 (since 3.0-beta)"）
【必须注意】
・beta 首次出现＝外部集合、快照、TTL、ORDER BY、查询聚合、MinHash、EmbList+DiskANN、Woodpecker、Storage V3；GA 补＝外部集合工作流完整、在线加列/回填/删列、schema 与索引同步演进、重排函数链、聚合上搜索路径（分面）、稀疏索引换代、超大规模优化、索引引擎直通、Woodpecker 独立部署。
・不要在此逐条解释特性做什么（7.2–7.4 各自展开）。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.2.3 四条兼容性红线

<!-- EDITOR:BEGIN 7.1.2.3
【本节要写什么】用 > ⚠️ **注意** 逐条列。这四条是全大章的公共前提，后续子章节会反复回指，务必写准。
【先读这些】https://milvus.io/docs/release_notes.md （"Compatibility and behavior notes"）
【必须注意】
・重点（照此口径，数值勿改）：
・Chapter 1 的"系统要求/硬件建议"不重写。
・四条红线要能被后文直接引用，数值一个字都不能改；第 2 条"回滚窗口永久关闭"是全章最贵的一句，务必写准。
【要点】
・1) Storage V3（Loon）默认关闭，快照、TEXT 依赖它，需手工开 common.storage.useLoonFFI；后续版本才会默认启用；
・2) 2.6 → 3.0 兼容且可回滚，但一旦启用/使用了改变序列化格式的特性（如 Storage V3），回滚窗口永久关闭；
・3) 新引入的索引算法默认不生效，需 dataCoord.targetVecIndexVersion=10、dataCoord.targetScalarIndexVersion=4；
・4) GPU 镜像迁至 CUDA 12.9，不再保留 Ubuntu 20.04 兼容。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.1.3 版本矩阵与客户端门槛

### 7.1.3.1 服务端与 SDK 对照表

<!-- EDITOR:BEGIN 7.1.3.1
【本节要写什么】只放一张表，不写步骤。列＝Milvus / pymilvus / Node.js SDK / Java SDK / Go SDK。
【先读这些】https://github.com/milvus-io/milvus/releases/tag/v3.0.0 ｜ https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/release_notes.md
【必须注意】
・v3.0.0 官方配套＝Milvus 3.0.0 / pymilvus 3.0.1 / Node.js 3.0.3 / Java 3.0.5 / Go 3.0.0；pymilvus 现行 3.0.2。
・️ 关键提醒：服务端 3.0.x 与 Lite 的 3.1/3.2.x 是两套号段，别写成"Milvus Lite 3.0"。
・Chapter 1（1.3.3）已对比过 Lite/Standalone/集群的差异定位，此处只做版本与 SDK 对应。
・配套矩阵以 GitHub v3.0.0 release 页为准：Milvus 3.0.0 / pymilvus 3.0.1 / Node 3.0.3 / Java 3.0.5 / Go 3.0.0；pymilvus 现行 3.0.2。Lite 是另一套号段，禁止写"Milvus Lite 3.0"。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.3.2 Python 版本门槛

<!-- EDITOR:BEGIN 7.1.3.2
【本节要写什么】两三行 + 一条 > 📌 **前置条件**。
【先读这些】https://github.com/milvus-io/pymilvus/releases ｜ https://pypi.org/project/milvus-lite/ ｜ https://milvus.io/docs/install-pymilvus.md
【必须注意】
・pymilvus 3.x 要求 Python ≥3.9（3.8 停支持）；milvus-lite 3.2.x 要求 Python ≥3.10。
・️ 不要改动 Chapter 4 现有 pymilvus==2.5.4 的 pin；建议本章加一句"新旧章节各用独立 venv"。
・版本门槛以 PyPI wheel 元数据与 install-pymilvus.md 现值为准（页面写 3.7+），发布前再核一次，不要凭印象写 3.9/3.10。不要动 Ch4 的 pymilvus==2.5.4 pin，改为建议新旧章节各用独立 venv。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.1.4 环境升级实操

### 7.1.4.1 全新部署 3.0 单机版

<!-- EDITOR:BEGIN 7.1.4.1
【本节要写什么】一段 bash/powershell + 一句"预期输出"。只给**差异片段**，不要重贴整份 compose。
【先读这些】https://milvus.io/docs/install_standalone-docker.md ｜ https://milvus.io/docs/configure-docker.md ｜ https://milvus.io/docs/dynamic_config.md
【必须注意】
・镜像 tag 换 3.0.x；standalone 配置文件与 volume 路径差异项；服务健康检查方式。
・️ 待你浏览器复核：本机打不开 milvus.io，官方 compose 的确切镜像 tag 请以安装页现值为准。
・装 Attu、基础 docker 流程、停止与卸载在 Chapter 1（1.5.2–1.5.4）已写，一律不重复。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.4.2 从 2.6 原地升级

<!-- EDITOR:BEGIN 7.1.4.2
【本节要写什么】写成检查单式步骤（升级前备份 → 升级 → 验证 → 再决定是否开新开关）。
【先读这些】https://milvus.io/docs/upgrade_milvus_standalone-docker.md ｜ https://milvus.io/docs/upgrade_milvus_cluster-helm.md ｜ https://milvus.io/docs/upgrade_milvus_standalone-operator.md ｜ https://milvus.io/docs/upgrade_milvus_cluster-operator.md
【必须注意】
・升级前必看 preupgrade 检查项；2.6→3.0 可回滚的前提是没有启用改序列化格式的特性；升级后索引版本/元数据的处理。
・️ 建议配一张决策表：要用快照/TEXT？→ 要开 Loon？→ 开了就不能回滚 → 先在实验环境演练。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.4.3 打开存储引擎 V3（Loon）

<!-- EDITOR:BEGIN 7.1.4.3
【本节要写什么】milvus.yaml 片段（语言标 yaml）+ 一行验证方式 + 一句"预期输出"。
【先读这些】https://milvus.io/docs/storage-v3.md ｜ https://milvus.io/docs/release_notes.md （Compatibility notes）
【必须注意】
・键名 common.storage.useLoonFFI；默认 false。
・Loon 的原理放到 7.2.2，本章只讲"怎么开、开了意味着什么"。
・键名 common.storage.useLoonFFI，默认 false。本页只写"怎么开、开了意味着什么"，原理留给 7.2.2。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.4.4 打开新版索引开关

<!-- EDITOR:BEGIN 7.1.4.4
【本节要写什么】两个键一起给（yaml 片段），并写明"不开会怎样"——索引建了却仍走老路径，而且**不报错**（高频坑，务必上 > ⚠️ **注意**）。
【先读这些】https://milvus.io/docs/release_notes.md ｜ https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index_selection.md
【必须注意】
・dataCoord.targetVecIndexVersion=10；dataCoord.targetScalarIndexVersion=4；键名仅见 release notes 的 Compatibility notes，未收录进 configure_datacoord 配置手册；按静态配置处理：改后重启，并对已有段重建索引后验证。
・Chapter 2（2.3.3 索引类型 / 2.3.7 索引参数）已讲经典索引，此处不重讲。
・不开不报错，只是仍走老路径——全章最高频的坑，必须单独一个 > ⚠️ 注意，并在 7.4.2.2 回指。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.4.5 客户端 pymilvus 升级

<!-- EDITOR:BEGIN 7.1.4.5
【本节要写什么】一行安装命令 + 一张"后续子章节会用到哪些新客户端能力"预告表（只列名，不讲用法）。
【先读这些】https://milvus.io/docs/install-pymilvus.md ｜ https://milvus.io/docs/using_milvusclient.md ｜ https://github.com/milvus-io/pymilvus/releases
【必须注意】
・pip install -U "pymilvus>=3.0.1,<3.1"。3.0.x 客户端线索：alter_collection_schema()（在线改 schema）、FunctionChain 与 XGBoost 表达式辅助、Bloom/Roaring bitmap 构建器、搜索聚合与 group_by_fields、compaction 生命周期观测、dump_messages / get_replicate_info、DataType.TEXT、alter_role/describe_role。
・️ 每个接口名写前用官方 API 页或 releases 原文核对参数拼写，勿凭记忆。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.1.5 三种形态的支持边界

### 7.1.5.1 特性 × 形态支持矩阵

<!-- EDITOR:BEGIN 7.1.5.1
【本节要写什么】一张表。行＝外部集合 / 快照 / Storage V3 / TEXT / StructArray / 服务端排序聚合与分面 / Function Chain / 新版稀疏索引 / Woodpecker / Force Merge；列＝Lite / Standalone / 集群 / 备注。
【先读这些】https://milvus.io/docs/milvus_lite.md ｜ https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/woodpecker.md ｜ https://pypi.org/project/milvus-lite/
【必须注意】
・Lite 是嵌入式引擎，其 snapshot 概念（钉住不可变数据/增量/索引文件并保存 schema 与 manifest 副本）≠ 服务端快照特性，必须显式区分，否则读者会拿 Lite 去试外部集合。
・️ 官方未逐项明示 Lite 的 3.0 能力边界：不确定的格子写"待验证"，不要猜。
・Chapter 1（1.3.3）与 Chapter 6《Milvus Lite部署与应用》已讲形态定位与 Lite 装法，此处只填这张新表。
・先给表再给替代观察方式；Lite 对本章特性的支持以 milvus_lite.md 现文为准，未写明就标"待验证"，不要替官方下结论。
【写完自检】表能当速查卡用；读者不看正文也能据此做决定。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.1.5.2 本教程的实验环境选择

<!-- EDITOR:BEGIN 7.1.5.2
【本节要写什么】给读者一句"照我做"的环境建议；明确哪些小节必须有对象存储/集群才跑得动（外部集合、快照备份、Woodpecker 独立部署），并给替代观察方式（看配置、看 Attu、看日志）。
【必须注意】
・️ 跑不动的一律标注前置条件，不要假装能跑。
・替代观察方式要具体到"看哪个输出/哪个目录"，否则读者还是跑不动。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.1.6 本章小结

<!-- EDITOR:BEGIN 7.1.6
【本节要写什么】沿用你既有模板——`本章核心掌握：` + 3–5 条 + 一句衔接（"7.2 起在此环境上实操新特性"）。
【必须注意】
・️ 本章只保留三张表（版本矩阵 / SDK 对照 / 支持矩阵）+ 升级操作，特性原理一律不写，防止与 7.2–7.4 重复。
【写完自检】以「本章核心掌握：」起头；只看小结也能复述本章三条主线。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->
