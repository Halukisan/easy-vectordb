# Chapter 7.5 Milvus 3.0 端到端实战：从湖上文件到可回滚的检索服务

<!-- EDITOR:BEGIN 本子章节定位
写什么：把 7.2–7.4 的能力串成一条完整链路，让读者一次跑通并看到"3.0 与 2.6 的体验差"。
不写：任何单点特性的重复教学（外部集合语法见 7.2.3、StructArray 见 7.3.2、稀疏索引见 7.4.2），本章只组装。
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

## 7.5.1 学习目标

<!-- EDITOR:BEGIN 7.5.1
【本节要写什么】4 条：独立搭起一条 3.0 检索链路 / 会用函数在库内派生检索字段 / 会在一次请求里完成检索+排序+聚合+重排 / 会用快照做发布前保护与回滚。
【必须注意】只写本章能证明的目标；"独立搭起"意味着环境准备要自包含，不能默认读者读过前六章。
【写完自检】4–6 条目标都能在本页找到对应节；读者能据此判断要不要读本章。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.5.2 场景与数据准备

<!-- EDITOR:BEGIN 7.5.2
【本节要写什么】给一个贴教程气质的场景（建议：本地文档库→导出为 Parquet 放对象存储/本地目录→做混合检索+去重+版本可回滚）。列清目录结构与样例字段。
【先读这些】https://milvus.io/docs/prepare-source-data.md ｜ https://milvus.io/docs/import-data.md
【必须注意】
・️ 数据放 `data/` 下新建子目录（如 `data/milvus30_demo/`），生成脚本放 `src/` 独立目录，沿用本项目 README + requirements.txt 惯例。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.5.3 搭建步骤

### 7.5.3.1 环境与开关自检

<!-- EDITOR:BEGIN 7.5.3.1
【本节要写什么】三步确认（服务端版本 / Loon 开关 / 索引版本开关），给一段打印版本的代码。回引 7.1.4，不重抄配置细节。
【先读这些】https://milvus.io/docs/install-overview.md ｜ https://milvus.io/docs/dynamic_config.md
【必须注意】
・这是全章"跑不动"的第一道闸，给可复制的检查清单与"哪项没开"的输出。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.2 挂载外部集合

<!-- EDITOR:BEGIN 7.5.3.2
【本节要写什么】8–15 行代码，跑通"湖上文件→外部集合→可检索"。语法回引 7.2.3，此处只给最终形态配置。
【先读这些】https://milvus.io/docs/create-an-external-collection.md
【必须注意】挂载与建索引的写法必须与 7.2.3 完全一致；此处只贴最终配置，参数解释一律回引。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.3 库内派生检索字段

<!-- EDITOR:BEGIN 7.5.3.3
【本节要写什么】一次配好两类函数：BM25 稀疏向量（全文）与 MinHash 签名（去重），说明"原文留外部、检索字段在库里算"。
【先读这些】https://milvus.io/docs/bm25-function.md ｜ https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/embedding-function-overview.md
【必须注意】
・函数原理回引 7.3.4.4，这里只写挂上去、验一下。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.4 用 StructArray 承载切片与多向量

<!-- EDITOR:BEGIN 7.5.3.4
【本节要写什么】给"一文档多切片、每切片一个向量 + 若干标量"的字段设计与写入代码，体现父子同集合。
【先读这些】https://milvus.io/docs/create-structarray-field.md ｜ https://milvus.io/docs/search-with-embedding-lists.md
【必须注意】
・语法细节见 7.3.2，此处不重讲。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.5 索引配置

<!-- EDITOR:BEGIN 7.5.3.5
【本节要写什么】一段代码同时给：稠密向量索引 / 稀疏向量用新算法 / struct 子字段索引 / 标量索引。并给"确认新路径生效"的验证行。
【先读这些】https://milvus.io/docs/index_selection.md ｜ https://milvus.io/docs/sparse-inverted-index.md ｜ https://milvus.io/docs/index-structarray-fields.md
【必须注意】四类索引都要有"打印实际生效配置"的验证行；只贴 create_index 不算验证。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.6 一次请求完成检索 + 排序 + 分面 + 重排

<!-- EDITOR:BEGIN 7.5.3.6
【本节要写什么】本章高潮。给一份完整调用（含 filter、排序、聚合分面、重排阶段），并贴真实返回片段，指出"过去要 3 次请求 + 客户端拼装"。
【先读这些】https://milvus.io/docs/search-aggregation.md ｜ https://milvus.io/docs/release_notes.md#v300（facet） ｜ https://milvus.io/docs/single-vector-search.md （找页内小节 "Sort Search Results by Scalar Fields"） ｜ https://milvus.io/docs/model-ranker-overview.md ｜ https://milvus.io/docs/boost-ranker.md ｜ https://milvus.io/docs/multi-vector-search.md ｜ https://milvus.io/docs/rrf-ranker.md
【必须注意】
・先给"2.6 要几次调用"的对照，再给"3.0 一次"；否则读者感受不到价值。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

### 7.5.3.7 发布前打快照与回滚演练

<!-- EDITOR:BEGIN 7.5.3.7
【本节要写什么】演练一次"故意写坏→从快照恢复→结果一致"，让读者亲眼看到快照创建是秒级、但保留会占存储，以及恢复速度。
【先读这些】https://milvus.io/docs/manage-snapshots.md ｜ https://milvus.io/docs/snapshot-use-cases.md
【必须注意】演练必须记录三个数：快照创建耗时、恢复耗时、恢复后行数与向量数是否一致；缺一个就不算完成。
【写完自检】示例可独立运行；读者按步骤能复现，并说出每一步的预期输出。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.5.4 日常运行：增量与演进

<!-- EDITOR:BEGIN 7.5.4
【本节要写什么】两小节合并写：外部数据刷新（refresh）+ 上线后换嵌入模型时在线加列回填；各 5–10 行代码。
【先读这些】https://milvus.io/docs/create-an-external-collection.md ｜ https://milvus.io/docs/add-fields-to-an-existing-collection.md ｜ https://milvus.io/docs/alter-collection-field.md
【必须注意】
・️ 明确"外部集合只读"，更新靠刷新而非写入。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.5.5 结果观察与代价

<!-- EDITOR:BEGIN 7.5.5
【本节要写什么】给一张"本次链路用了哪些 3.0 新能力 / 2.6 下要怎么做 / 实测差异"的总结表；再诚实列出踩到的限制（引用各节限制清单，不复述）。
【先读这些】https://milvus.io/docs/structarray-limits.md ｜ https://milvus.io/docs/limitations.md
【必须注意】
・️ 所有数字必须是你自己跑出来的，未跑的写"待补测"。
・所有数字必须是作者自己跑的；没跑写"待补测"，禁止搬宣传数字。
【写完自检】读者能用自己的话回答本节标题提出的问题，且不需要翻其他资料。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->

## 7.5.6 本章小结

<!-- EDITOR:BEGIN 7.5.6
【本节要写什么】`本章核心掌握：` + 4 条 + 一句全大章收束（"至此 Milvus 3.0x 的新增能力全部覆盖，未列出的均为与 2.6 相同的行为，见前六章"）。
【必须注意】小结不得引入新事实；收束句要能对上 7.1.0 的三条主线，对不上说明正文跑题。
【写完自检】以「本章核心掌握：」起头；只看小结也能复述本章三条主线。
【删除】本节定稿后，连同本注释块（EDITOR:BEGIN … EDITOR:END）一起删除。
EDITOR:END -->
