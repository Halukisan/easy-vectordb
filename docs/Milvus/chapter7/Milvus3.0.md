# Chapter 7.0 Milvus 3.0：从 2.x 走过来

<!-- EDITOR:BEGIN 
   状态：草稿。未写入 docs/.vitepress/config.js 的 sidebar / nav。
   
   { text: 'Chapter 7 · Milvus 3.0x 新特性', items: [
     { text: '从 2.x 走过来', link: '/Milvus/chapter7/Milvus3.0' },
     { text: '版本演进与环境升级', link: '/Milvus/chapter7/Milvus3.0版本演进与环境升级' },
     { text: '外部集合与快照', link: '/Milvus/chapter7/Milvus3.0外部集合与快照' },
     { text: '数据模型与检索增强', link: '/Milvus/chapter7/Milvus3.0数据模型与检索增强' },
     { text: '索引与写入引擎', link: '/Milvus/chapter7/Milvus3.0索引与写入引擎' },
     { text: '端到端实战', link: '/Milvus/chapter7/Milvus3.0端到端实战' } ] },

EDITOR:END -->

本章将介绍Milvus 3.0 新版本，AI时代，是否还有必要手动编写教程一直都是值得我们DW思考的事情，大可以告诉自己的 AI 助手：“帮我查询Milvus官方文档，告诉我3.0版本都更新了什么”。然后AI会给你输出一大串的内容，详细的告诉你新版本都更新了什么，并且还会附带一张和旧版本对比的表格。下意识的指令和流水似的答案，无论是输入还是输出都没有神经元的信号跳动，答案宛如回家吃饭一样顺其自然的出现，“妈妈会做好饭然后喊我吃饭的”，日复一日的习惯，养成了衣来伸手饭来张口的自己，自己的能力慢慢退化，品尝这些饭菜，思考是如何做的，哪怕和“妈妈”对话，问做这个菜的时候放了些什么，盐放了多少......


本章节会采用分总的形式编写，不会直接像AI一样陈列新版本的变动，静下心来阅读这些内容。



Milvus 3.0 于 2026 年 7 月 29 日发布首个正式版本（v3.0.0），截至本文撰写(2026.10.19)，3.0 线已经走到 v3.0.3。作为一个 major 版本，它带来的不是"多几个参数、多几个索引类型"，而是把几件在 2.x 里**必须由应用在库外完成的事**，变成了数据库内部的一等能力。

本篇不重复后面五篇的操作细节，只解决三个问题：

- 3.0 在整个版本演进里处于什么位置，和 2.6 是什么关系；
- 哪些看起来"新"的能力其实 2.5 / 2.6 就已经有；
- 要让这些新能力真正生效，你必须做些什么事情，哪些是默认就有的。

## 7.0.1 学习目标

- 能说出 Milvus 3.0 三条主线分别解决什么问题：数据在哪里处理、检索逻辑在哪里执行、数据与检索模式怎么扩展
- 能区分"3.0 新增"与"2.x 已有"
- 记住官方明确写出的四条兼容性与默认行为边界
- 能判断自己该马上升级、该观望、还是只读文档不动环境

## 7.0.2：Milvus 3.0 的三条主线

3.0 的新特性看着多，本文把它们归成三条主线，分别回答三个问题：**数据在哪里处理、检索逻辑在哪里执行、系统怎样适应新的数据与检索需求**。

**第一条：Lake-native，让计算靠近数据。** 这里的"湖原生"指直接面向数据湖里的数据，不必先搬进 Milvus。在 **外部集合（External Collection）** 出现之前，湖上的数据要先进入 Milvus 才能查；3.0 的外部集合让 Milvus 直接读 Parquet / Lance / Iceberg / Vortex，**不搬数据**。官方对这条能力的定义只有一句：

> **External Collection** — query lakehouse data (Parquet, Lance, Iceberg, Vortex) in place: zero-copy, read-only, synced through incremental refresh.

零拷贝、**只读**、增量刷新，三个限定词都是官方原话。**只读**本身就说明这类集合不用于写入；**增量刷新**说明数据更新靠刷新动作带进来。至于写入走哪条路径、刷新后多久可见，发布说明没有写，**要看外部集合的专门文档**，本篇不在这里下结论。

**第二条：检索引擎增强，把更多检索与结果处理放进库内。** 这类需求过去很容易落到应用侧：先多取一批结果，再在本地排序、统计或重排。3.0 把这几件事搬进库内：**服务端 ORDER BY**、**查询聚合**、**搜索时分面返回**、**稀疏索引全面换代**（SINDI、Block-Max WAND、Block-Max MaxScore），以及声明式的 **Function Chain 重排**（含原生 XGBoost 打分），细节见 7.3 与 7.4。官方对分面的描述顺带把"为什么要搬进库内"说清了：

> Specify a facet field at search time and Milvus returns the top facet values, each represented by its best-matching member in ANN ranking and annotated with aggregates such as COUNT and AVG — the faceted-search sidebar (brand, price range, attributes) in one request, instead of over-fetching and counting client-side.

> 可以在搜索请求中指定一个分面字段，Milvus 会返回该字段的热门取值；每个取值由 ANN 排名中最匹配的实体代表，并附带 COUNT、AVG 等统计值。这样应用可以在一次请求中生成类似按品牌、价格区间或属性筛选的分面栏，而不必先多取一批结果，再由客户端计算统计值。

**第三条：数据与检索模式扩展，适应持续演进的数据与多向量场景。** 这条主线是三件事：**在线改 schema**（加字段 / 删字段 / 回填）、**TEXT 长文本字段**、**多向量（StructArray）**。官方对第一件（在线改 schema）的原话是：

> 3.0.0 closes the loop: columns can be added, filled, and dropped while serving continues.

> 在 Milvus 提供服务期间，可以在线添加字段、为字段回填数据，并删除字段。

这句要说得准确：它支持的是**在服务持续期间做字段级的增删与回填**，不能理解成"任何 schema 变更都不用停服、不用重建 collection"。具体到操作，有两条限制值得记住。

这两条限制的展开不在本篇：第一条（可空与回填）放在 7.2.5，第二条（StructArray 子字段）放在 7.3.2.6。第一条主线上剩下的快照与 Storage V3，第二条的排序、聚合、稀疏索引与重排，第三条的 TEXT 与多向量，也都从本篇搬去了各自章节——快照在 7.2.4，在线改字段在 7.2.5，排序、聚合与分面在 7.3.5，重排在 7.3.6，稀疏索引在 7.4.2，TEXT 与多向量在 7.3。本篇只留每条主线的开头，故事在后面的章节里继续。

---

## 7.0.3 三条主线收束：落到动手层面

三条主线到此讲完，落到动手层面只有一句话：**第一条改的是数据放在哪里、怎么进库；第二条改的是查询请求里能写什么；第三条改的是集合本身长什么样。** 第二、三条要动应用代码，第一条主要动部署与数据布局。至于哪些能力必须先开开关才真正存在，见 7.0.4。

## 7.0.4 四条兼容性与默认行为边界

3.0 的能力**并不是一升级就全部生效**：官方在 v3.0.0 的 *Compatibility and behavior notes* 里只明确写出了四条默认行为，逐条看原文与含义。

**一、存储引擎 V3 默认关闭。** 原文：

> **Storage V3 (Loon) is disabled by default.** Features that depend on it — such as Snapshot and TEXT fields — require enabling it manually via `common.storage.useLoonFFI`. Storage V3 will be enabled by default in a later release.

含义是：装好 3.0 之后，**快照和 TEXT 长文本字段这些功能实际上并未启用**，必须手动打开 `common.storage.useLoonFFI` 才会出现。

**二、2.6 → 3.0 保证兼容，也能回滚——但有个一次性的开关。** 原文：

> **2.6 → 3.0 compatibility and rollback are guaranteed** — a 3.0 deployment can be rolled back to 2.6. However, once you enable or use features that change the serialized data format (for example Storage V3), rollback is no longer possible.

也就是说升级本身留了后路，可**一旦启用了改变序列化数据格式的特性（例如 Storage V3），回滚窗口就永久关闭**。

**三、新引入的索引算法默认不生效。** 原文：

> **New index versions are opt-in for now.** Newly introduced index algorithms require manually raising the target index version (`dataCoord.targetVecIndexVersion` to 10, `dataCoord.targetScalarIndexVersion` to 4) before they take effect; a later release will enable them by default.

含义是：新引入的索引算法需要手动提高对应目标索引版本后才生效，也就是把 `dataCoord.targetVecIndexVersion` 抬到 **10**、`dataCoord.targetScalarIndexVersion` 抬到 **4**。

**四、GPU 镜像换到 CUDA 12.9。** 原文只有一句：**GPU images move to CUDA 12.9** and no longer preserve Ubuntu 20.04 GPU compatibility。

## 7.0.5 写在最后

三条主线打过了照面，四条边界也交了底。现在轮到你做那个 7.0.1 里预告的决定：马上升级、再观望，还是只读文档、环境先不动。我不替你答，只提醒一句——答案就藏在 7.0.4 的第二条里：升级本身留了后路，可一旦打开了改变数据格式的那几个开关，回滚的门就永久关上了。所以先走 7.1 的路，把版本与环境的事办踏实，再回来动应用代码。

小时候觉得妈妈做饭是理所当然，直到自己站到灶台前，才知道火候两个字怎么写。3.0 把很多"库外的事"收进了库里，排序、聚合、分面、重排，都成了请求里的一行；可火候还是你的：哪条主线该用、开关什么时候开、快照留不留，都得自己掂量。教程到这里，只是把菜谱摊开。下一章 7.1，我们从头一天进厨房开始——版本怎么选、环境怎么升，一步一步来。

