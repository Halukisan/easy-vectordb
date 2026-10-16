# Chapter 7.3 Milvus 3.0 数据模型与检索增强

<!-- EDITOR:BEGIN 本子章节定位
写什么：StructArray、可空向量/默认值/TIMESTAMPTZ/行级 TTL、TEXT 与四类匹配算子、MinHash DIDO、服务端排序/聚合/分面、服务端重排函数链、中文检索增强。
不写：向量/索引/Schema 基础概念（Ch2、Ch3），全文+稀疏混合检索入门（Ch5），客户端 Reranker 用法（Ch6）。
EDITOR:END -->

## 7.3.1 学习目标

<!-- EDITOR:BEGIN 7.3.1
🎯 要求：4–6 条，动词起头。建议：用 StructArray 建模"一文档多切片" / 判断该用可空向量还是默认值 / 用一次请求拿到排序+统计+分面结果 / 把重排从客户端挪到服务端 / 给中文语料配一套分析器与词典。
EDITOR:END -->

## 7.3.2 数组结构体（StructArray）

### 7.3.2.1 它解决什么建模问题

<!-- EDITOR:BEGIN 7.3.2.1
🎯 要求：3–6 行 + 一张表（"一文档多切片"在 2.6 的三种凑合写法 vs 3.0 的 StructArray），点明"父子同集合、一次写入、检索时按子元素返回"。
🔗 参考：https://milvus.io/docs/array-of-structs.md ｜ https://milvus.io/docs/best-practices-for-array-of-structs.md
🚫 去重边界：Ch3（3.3.6 动态字段 / JSON 字段）讲过"弱结构扩展"，此处只对比"嵌套数组对象"这一新类型，不重讲 JSON。
EDITOR:END -->

### 7.3.2.2 定义与写入

<!-- EDITOR:BEGIN 7.3.2.2
🎯 要求：两段可跑代码（建集合含 struct 字段 → 插入嵌套数据），注释写清嵌套层级；给一段"打印返回结构"的观察代码。
🔗 参考：https://milvus.io/docs/create-structarray-field.md ｜ https://milvus.io/docs/insert-data-into-structarray-fields.md
⚠️ 嵌套层数、子字段类型上限等以官方页现值为准，不要凭印象写数字。
EDITOR:END -->

### 7.3.2.3 给子字段建索引

<!-- EDITOR:BEGIN 7.3.2.3
🎯 要求：一段代码 + 一句"哪些子字段值得建索引、哪些不需要"。
🔗 参考：https://milvus.io/docs/index-structarray-fields.md ｜ https://milvus.io/docs/index_selection.md
🚫 去重边界：索引类型选择原理在 Ch2（2.3.3），此处只讲 struct 子字段的索引写法差异。
EDITOR:END -->

### 7.3.2.4 四种检索姿势

<!-- EDITOR:BEGIN 7.3.2.4
🎯 要求：小节内四个 `#### 1. xxx`，每个 5–10 行代码，统一用同一份样例语料，避免每次换数据。
🔗 参考：https://milvus.io/docs/basic-vector-search-with-structarray.md ｜ https://milvus.io/docs/filtered-search-with-structarray.md ｜ https://milvus.io/docs/grouping-search-with-structarray.md ｜ https://milvus.io/docs/hybrid-search-with-structarray.md ｜ https://milvus.io/docs/range-search-with-structarray.md
⭐ 重点：按子元素命中返回、带过滤的 struct 检索、按父文档分组去重、多向量混合。
EDITOR:END -->

### 7.3.2.5 操作符

<!-- EDITOR:BEGIN 7.3.2.5
🎯 要求：一张算子表（算子/作用/示例表达式），正文不超过 15 行。
🔗 参考：https://milvus.io/docs/struct-array-operators.md
EDITOR:END -->

### 7.3.2.6 限制清单

<!-- EDITOR:BEGIN 7.3.2.6
🎯 要求：把官方限制逐条抄成中文清单（数量上限、不支持的操作、与动态字段/分区键的交互限制）。这一节是本子章节**最该存在**的内容，别省略。
🔗 参考：https://milvus.io/docs/structarray-limits.md ｜ https://milvus.io/docs/best-practices-for-array-of-structs.md
⚠️ 用 > ⚠️ **注意** 收口，明确"先查限制再设计 schema"的顺序。
EDITOR:END -->

## 7.3.3 Schema 灵活性：可空向量 · 默认值 · 时区 · 行级过期

### 7.3.3.1 可空向量与默认值

<!-- EDITOR:BEGIN 7.3.3.1
🎯 要求：先讲场景（部分行暂无向量、嵌入模型上线前占位、稀疏向量可选），再给代码。
🔗 参考：https://milvus.io/docs/nullable-and-default.md ｜ https://milvus.io/docs/add-fields-to-an-existing-collection.md
🚫 边界：在线加列本身在 7.2.5 讲过，此处只讲 NULL/default 语义，勿重复 API 流程。
EDITOR:END -->

### 7.3.3.2 TIMESTAMPTZ 字段

<!-- EDITOR:BEGIN 7.3.3.2
🎯 要求：讲清与 Int64 存时间戳的区别（时区语义、按时间过滤/过期的可用性），给一段建字段+过滤查询。
🔗 参考：https://milvus.io/docs/timestamptz-field.md
EDITOR:END -->

### 7.3.3.3 行级 TTL

<!-- EDITOR:BEGIN 7.3.3.3
🎯 要求：一张对比表（集合级 TTL vs 行级 TTL：作用范围/配置位置/依赖字段/清理粒度），这是最容易写混的地方。
🔗 参考：https://milvus.io/docs/set-collection-ttl.md#Set-entity-level-TTL--Milvus-30x ｜ https://milvus.io/docs/timestamptz-field.md
🚫 去重边界：Ch3（3.9.5 数据自动过期 = 集合级 TTL）已写旧机制，**只写差异，不重讲集合级用法**。
⚠️ 必须提醒：过期不等于合规删除，敏感数据另议。
EDITOR:END -->

## 7.3.4 长文本字段 TEXT 与匹配算子

### 7.3.4.1 TEXT 与 VarChar 的分界

<!-- EDITOR:BEGIN 7.3.4.1
🎯 要求：一张表（类型/长度上限/可建什么索引/可用于哪些算子/存储位置）。
🔗 参考：https://milvus.io/docs/text.md ｜ https://milvus.io/docs/string.md
⭐ 重点：TEXT 去掉短文本长度限制，短的内联、超长的走大对象存储且 compaction 只搬引用；这是 RAG 长文场景的关键变化。
🚫 去重边界：VarChar 基础在 Ch2（2.2.4 标量字段）/Ch3（3.3.4），此处只讲新增类型差异。
EDITOR:END -->

### 7.3.4.2 四类匹配算子

<!-- EDITOR:BEGIN 7.3.4.2
🎯 要求：四段各 5–8 行代码，统一同一份中文+英文混合语料。
🔗 参考：https://milvus.io/docs/keyword-match.md ｜ https://milvus.io/docs/phrase-match.md ｜ https://milvus.io/docs/pattern-matching.md ｜ https://milvus.io/docs/membership-match.md
⭐ 重点：关键词 / 短语（带邻近语义）/ 模式匹配 / 集合成员判定（配合 Bloom·Roaring 位图构建器）。
🚫 去重边界：Ch3（3.6.2 字符串匹配与数组字段查询操作符）只讲过 like/==/in 基本式，**算子语义升级部分才写**。
EDITOR:END -->

### 7.3.4.3 MinHash DIDO：文档进·文档出查重

<!-- EDITOR:BEGIN 7.3.4.3
🎯 要求：给一个"原始语料 → 自动签名 → 近重复检索 → 去重后入库"的完整小闭环（15–25 行）。
🔗 参考：https://milvus.io/docs/minhash-function.md ｜ https://milvus.io/docs/minhash-lsh.md
⭐ 重点：DIDO=输入文档、输出签名向量，函数由内核驱动；典型场景=语料清洗、爬取内容去重。
🚫 去重边界：Ch5 的 BM25 稀疏混合检索不重述，只写 MinHash 这条新链路。
EDITOR:END -->

## 7.3.5 服务端排序 · 聚合 · 分面

### 7.3.5.1 检索结果按标量字段排序

<!-- EDITOR:BEGIN 7.3.5.1
🎯 要求：先给"2.6 只能客户端排"的痛点，再给 3.0 写法；给多字段升降序 + 与 topk 交互的说明。
🔗 参考：https://milvus.io/docs/single-vector-search.md#Sort-Search-Results-by-Scalar-Fields--Milvus-30x
🚫 去重边界：Ch3（3.5.1 普通向量检索）不重讲，只写排序参数。
EDITOR:END -->

### 7.3.5.2 查询侧排序与聚合

<!-- EDITOR:BEGIN 7.3.5.2
🎯 要求：一段代码给出 COUNT/SUM/AVG/MIN/MAX + group-by 的返回样例（贴真实输出 JSON）。
🔗 参考：https://milvus.io/docs/get-and-scalar-query.md#Aggregate-Query-Results--Milvus-30x ｜ https://milvus.io/docs/search-aggregation.md
🚫 去重边界：Ch3（3.6.4 统计查询）是客户端聚合/过取老写法——**必须画一张"旧写法 vs 新写法"两步对照**，这是本节最大价值。
⚠️ 提醒读者：统计口径是"命中集合"而非"全表"时，要写清 filter 的作用范围。
EDITOR:END -->

### 7.3.5.3 分面检索

<!-- EDITOR:BEGIN 7.3.5.3
🎯 要求：给"商品搜索页左侧可点筛选"式的最小例子：一次请求同时拿向量 topK + 各分面 Top 值与计数。
🔗 参考：https://milvus.io/docs/search-aggregation.md
⭐ 重点：这是 3.0.0 GA 新增（beta 无），每个分面值由其最优匹配成员代表。
EDITOR:END -->

### 7.3.5.4 分组返回

<!-- EDITOR:BEGIN 7.3.5.4
🎯 要求：与 7.3.2.4 的 struct 分组区分开：这里是普通字段 group_by 透传；5 行代码即可。
🔗 参考：https://milvus.io/docs/grouping-search.md ｜ https://milvus.io/docs/multi-vector-search.md
🚫 去重边界：Ch3 无 group_by 内容，无需回引；不要和 struct 分组混为一谈。
EDITOR:END -->

## 7.3.6 服务端重排函数链

### 7.3.6.1 与客户端 Reranker 的分工

<!-- EDITOR:BEGIN 7.3.6.1
🎯 要求：一张表（重排位置/是否多一跳网络/延迟影响/可组合性/适用场景），用 `> ⛳ **架构演进提示**` 收口。
🔗 参考：https://milvus.io/docs/model-ranker-overview.md ｜ https://milvus.io/docs/rerankers-overview.md
🚫 去重边界：Ch6《Milvus Reranker 重排》讲的是**外部模型/客户端 ranker**，此处只讲服务端函数链，明确"两条路都在，怎么选"。
EDITOR:END -->

### 7.3.6.2 声明式重排实操

<!-- EDITOR:BEGIN 7.3.6.2
🎯 要求：三个小例子递进：融合排名 → 提升/降权某表达式命中 → 外部模型打分融合；每个 8–12 行。
🔗 参考：https://milvus.io/docs/rrf-ranker.md ｜ https://milvus.io/docs/weighted-ranker.md ｜ https://milvus.io/docs/boost-ranker.md ｜ https://milvus.io/docs/decay-ranker-overview.md
⚠️ 客户端函数链 API 名与参数请对照 pymilvus 3.x 与官方页现值核对后再写，勿凭记忆；模型打分需外部服务在跑，正文注明前置条件。
EDITOR:END -->

## 7.3.7 中文检索体验增强

### 7.3.7.1 文件级词典资源

<!-- EDITOR:BEGIN 7.3.7.1
🎯 要求：讲"为什么要在集群级注册词典文件"（多库共用、热更新、免重启），给上传→引用→生效三步。
🔗 参考：https://milvus.io/docs/manage-file-resources.md ｜ https://milvus.io/docs/analyzer-overview.md
EDITOR:END -->

### 7.3.7.2 中文分析器与过滤链

<!-- EDITOR:BEGIN 7.3.7.2
🎯 要求：给一套可直接复用的中文 BM25 配置（分析器 + 拼音 + 同义词 + 数字规范化），并给"改前/改后检索效果对照"两行示例结果。
🔗 参考：https://milvus.io/docs/chinese-analyzer.md ｜ https://milvus.io/docs/pinyin-filter.md ｜ https://milvus.io/docs/synonym-filter.md ｜ https://milvus.io/docs/decimaldigit-filter.md ｜ https://milvus.io/docs/multi-language-analyzers.md ｜ https://milvus.io/docs/choose-the-right-analyzer-for-your-use-case.md
🚫 去重边界：Ch5 全文检索入门已讲 BM25 基本式，此处只写分析器/词典侧增量。
EDITOR:END -->

## 7.3.8 本章小结

<!-- EDITOR:BEGIN 7.3.8
🎯 要求：`本章核心掌握：` + 4–5 条 + 衔接句（"7.4 看索引与写入引擎"）。
⚠️ 自检：StructArray/TEXT/聚合/重排四块是否各只出现一次；是否误重述了 Ch3 的统计查询、Ch3 的集合级 TTL、Ch6 的客户端重排。
EDITOR:END -->
