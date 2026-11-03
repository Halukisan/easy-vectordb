# Chapter 7.4 进阶 · 稀疏倒排索引的遍历算法

对应正文：[7.4.1 稀疏索引换代](./Milvus3.0索引与性能选择)。正文只保留"什么场景用哪个"，这一篇讲五个取值各自怎么工作。

## 先建立一个模型

倒排索引里，每个查询词后面挂着一条"含有这个词的文档 ID"列表，叫倒排链。一次查询 = 拿着几个查询词找到各自的倒排链，从里面找出总分最高的 k 篇文档。五个算法的差别，全在"怎么从这几条链里找得更快"。

## 两个遍历方向：DAAT 与 TAAT

- **DAAT**（Document-at-a-Time）：一篇文档一篇文档地过。对每篇文档，把它在各个查询词上的得分加总，就是这篇文档的总分，同时维护当前前 k 名名单。
- **TAAT**（Term-at-a-Time）：一个词一个词地过。把某个查询词的整条倒排链走完，给碰到的每篇文档累加分数，再走下一个词，最后统一排名。

`TAAT_NAIVE` 就是 TAAT 的朴素实现：不剪枝，全部算完。适合当对照基线，或打分需要跟随全库统计量（比如平均文档长度变化会实时影响各词的贡献）的场景。

## 上界剪枝：MAXSCORE 与 WAND

DAAT 要快，靠"提前算账"：给每个查询词记一个分数上限——这个词最多能给一篇文档加多少分。

- `DAAT_MAXSCORE`：逐文档打分时，一篇文档"当前已得分 + 还没算的词的上限"都进不了前 k 名，剩下的词就不算了，直接看下一篇。
- `DAAT_WAND`：更省一步——一篇文档把所有词的上限全加上都够不到当前第 k 名，这篇文档连打分都跳过。

两者的取舍：MAXSCORE 丢的是"这篇文档后半段的打分"，WAND 丢的是"整篇文档"。所以查询词多、k 大时 MAXSCORE 稳，查询短、k 小时 WAND 省得多——这也是官方参数页给的场景划分。

## Block-Max：把跳读做成整块

`BLOCK_MAX_MAXSCORE` / `BLOCK_MAX_WAND` 在上面两种之上再加一层：把每条倒排链切成块，每块预先存一个"本块最高分"。跳的时候先看块上限——这一块整体够不到阈值，整块跳过；够得到才进块里逐个文档比。

## SINDI：改存法，配 SIMD

参数页原话："A sparse inverted index based on fixed document-id windows, with SIMD acceleration for search."——文档 ID 按固定窗口分组存放，检索时用 SIMD 指令一组一组并行比对。它是 IP 度量的默认算法；发布说明把它列为 3.0 的新搜索算法之一，并提到配套的倒排列表压缩与可配置量化（压缩、量化的收益数字出自发布说明的内部基准，以你自己的数据实测为准）。

## 怎么实测差异

同一批数据，drop 索引后换一个 `inverted_index_algo` 重建，查询语句不变；同一批查询分别跑 QPS 和召回对比。每次只改算法这一个变量。

## 参考文档

- [Sparse Inverted Index](https://milvus.io/docs/zh/sparse-inverted-index.md)（`inverted_index_algo` 参数表与默认值）
- 参数页引用的论文：Query Evaluation: Strategies and Optimizations（MaxScore）· Efficient Query Evaluation using a Two-Level Retrieval Process（WAND）· Faster Top-k Document Retrieval Using Block-Max Indexes（Block-Max）· SINDI paper（参数页内有链接）
- [Milvus 发布说明](https://milvus.io/docs/zh/release_notes.md)
