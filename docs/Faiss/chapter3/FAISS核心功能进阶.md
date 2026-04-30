# Chapter 3：FAISS 核心功能进阶学习教程

通过“理论解析+核心 API+实战案例”的结构，帮助大家掌握复合索引设计、向量归一化、索引持久化及 GPU 加速等关键技能，解决大规模向量检索中的“精度-效率”平衡问题。

**前置准备**：

1. 安装 FAISS（CPU 版本：`pip install faiss-cpu`；GPU 版本：`pip install faiss-gpu`）；
2. 下载 SIFT1M 数据集（含 100 万张图片的 128 维特征向量，获取链接：https://huggingface.co/datasets/fzliu/sift1m/tree/main）；
3. 导入依赖库：`import faiss, numpy as np, time`。

## 1向量归一化与相似度适配：避免检索偏差

FAISS 中相似度计算依赖距离度量，而 COSINE 相似度（衡量向量方向一致性）需通过“L2 归一化”预处理才能正确计算，否则会导致结果偏差。

### 1.1 核心原理：COSINE 相似度与归一化的关系

COSINE 相似度用于衡量两个向量在方向上的一致性，公式如下：
$$
\text{cosine}(a,b) = \frac{a \cdot b}{\|a\| \|b\|}
$$
其中，`a*b`是向量 `a` 和 `b` 的内积，`|a|`和 `|b|` 是向量 `a` 和 `b` 的 L2 范数（即向量的长度）。


#### 1.1.1 L2 归一化对 COSINE 相似度的影响

如果我们对向量 `a` 和 `b` 进行 **L2 归一化**（即将它们的范数归一化为 1），那么：
$$
\|a\| = 1, \quad \|b\| = 1
$$
此时，COSINE 相似度就简化为：
$$
\text{cosine}(a,b) = a \cdot b
$$
也就是说，归一化后的向量之间的 COSINE 相似度直接等于它们的内积。

#### 1.1.2 L2 距离与 COSINE 相似度的关系

L2 距离（即欧几里得距离）可以通过 COSINE 相似度计算得到，具体关系如下：
$$
\|a - b\|^2 = 2(1 - \text{cosine}(a,b))
$$
由此可见，L2 距离与 COSINE 相似度是 **负相关** 的：当 COSINE 相似度越高，L2 距离越小。

#### 1.1.3 结论

1. 在计算 COSINE 相似度时，必须先对向量进行 L2 归一化处理（即将向量的长度规范化为 1）。
2. 使用 COSINE 相似度进行检索时，可以选择 **内积** 或 **L2 距离** 作为度量标准，前提是已对向量进行归一化。

### 1.2 核心 API：faiss.normalize_L2

```python
### 函数作用：对矩阵的每行（向量）做 L2 归一化
faiss.normalize_L2(x)  # x 为 numpy 数组（shape: N×d），原地修改

# 示例
x = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
faiss.normalize_L2(x)
print("归一化后向量：", x)
print("归一化后向量的 L2 范数：", np.linalg.norm(x, axis=1))  # 输出 [1. 1.]
```

> 如果运行后输出不是 `[1. 1.]`，大概率是代码执行时的精度显示问题（如 `[1.0000001 0.9999999]`），属于浮点数计算误差，本质仍满足归一化要求，不影响后续检索逻辑。

### 1.3 实战：归一化前后检索结果对比

**实战目标**：以 COSINE 相似度为目标，对比“未归一化”与“归一化”的检索精度差异。

```python
import faiss
import numpy as np

# 1. 构造测试数据（3 个数据库向量，2 个查询向量，模拟语义相似性）
xb = np.array([[1,2,3], [2,4,6], [3,3,3]], dtype=np.float32)  # 向量0与向量1方向完全一致（COSINE=1），向量2方向独特
xq = np.array([[0.5,1,1.5], [1,1,1]], dtype=np.float32)  # 查询0与向量0/1方向一致，查询1与向量2方向一致
d = xb.shape[1]  # 向量维度：3
k = 1  # 检索Top-1结果（只取最相似的1个）

# 方案 1：未归一化，用内积度量（错误方式）
index_no_norm = faiss.IndexFlatIP(d)  # 内积索引
index_no_norm.add(xb)  # 加入未归一化的数据库向量
D_no_norm, I_no_norm = index_no_norm.search(xq, k)  # 检索

# 方案 2：归一化后，用内积度量（正确方式）
xb_norm = xb.copy()
xq_norm = xq.copy()
faiss.normalize_L2(xb_norm)  # 数据库向量归一化（按行）
faiss.normalize_L2(xq_norm)  # 查询向量必须同步归一化（关键！）

index_norm = faiss.IndexFlatIP(d)
index_norm.add(xb_norm)
D_norm, I_norm = index_norm.search(xq_norm, k)

# 输出结果对比
print("=== 检索结果对比（目标：COSINE 相似度 Top-1）===")
print("数据库向量：\n", xb)
print("查询向量 0：", xq[0], "（与数据库向量0/1方向一致，COSINE=1）")
print("查询向量 1：", xq[1], "（与数据库向量2方向一致）")
print("\n【未归一化+内积】检索结果（错误，受向量长度干扰）：")
print(f"查询0 最相似向量索引：{I_no_norm[0][0]}，内积：{D_no_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_no_norm[1][0]}，内积：{D_no_norm[1][0]:.2f}")  

print("\n【归一化+内积】检索结果（正确，内积=COSINE相似度）：")
print(f"查询0 最相似向量索引：{I_norm[0][0]}，内积（COSINE）：{D_norm[0][0]:.2f}")  
print(f"查询1 最相似向量索引：{I_norm[1][0]}，内积（COSINE）：{D_norm[1][0]:.2f}")  
```

**结果分析**

未归一化时，内积受向量模长影响（向量 2 模长更大，虽与查询 1 方向一致但内积更高），导致检索错误；归一化后，内积直接等价于 COSINE 相似度，结果符合预期。

## 2 索引的保存与加载：适配工程化部署

实际应用中，索引训练耗时较长（百万级数据需分钟级），需将训练好的索引“持久化”保存，在线服务时直接加载使用，避免重复训练。

### 2.1 核心 API 与保存格式

| API 函数                                 | 功能描述                         | 注意事项                                   |
| :--------------------------------------- | :------------------------------- | :----------------------------------------- |
| `faiss.write_index(index, "index.path")` | 将索引序列化保存到本地文件       | 文件格式为二进制，不可直接编辑             |
| `faiss.read_index("index.path")`         | 从本地文件加载索引，返回索引对象 | 加载后可直接调用 search 方法，无需重新训练 |

注意事项：

1. 索引需完成训练和数据添加后再保存；
1. 保存路径需有写入权限，建议用 `.index` 作为后缀标识。

### 2.2 实战：离线训练索引，在线加载检索

**实战目标**：模拟“离线训练-保存索引-在线加载-检索”的工程化流程，验证索引保存与加载的一致性。

**阶段 1：离线训练并保存索引（离线环境）**

```python
import faiss
import numpy as np
import os
import time 
# 1. 加载 SIFT1M 数据库向量（仅需 xb）
def load_sift1m(path):
    # 加载文件（fvecs/ivecs均为二进制格式，按对应 dtype 读取）
    xb = np.fromfile(f"{path}/sift_base.fvecs", dtype=np.float32)
    xq = np.fromfile(f"{path}/sift_query.fvecs", dtype=np.float32)
    gt = np.fromfile(f"{path}/sift_groundtruth.ivecs", dtype=np.int32)
    print(f"向量维度d: {d}")
    print("原始数据形状:", xb.shape, xq.shape, gt.shape)
    
    # ---------------------- 解析 fvecs 格式（数据库/查询向量）----------------------
    # 验证数据长度是否符合格式（总长度必须是 (d+1) 的整数倍）
    assert xb.size % (d + 1) == 0, f"数据库文件损坏：总长度 {xb.size} 不是 (d+1)={d+1} 的整数倍"
    assert xq.size % (d + 1) == 0, f"查询文件损坏：总长度 {xq.size} 不是 (d+1)={d+1} 的整数倍"
    
    # 重构为 [向量数, d+1]，再去掉第一列（维度标识），得到实际向量
    xb = xb.reshape(-1, d + 1)[:, 1:]  # 数据库向量：(1000000, 128)
    xq = xq.reshape(-1, d + 1)[:, 1:]  # 查询向量：(10000, 128)
    
    # ---------------------- 解析 ivecs 格式（真实近邻）----------------------
    # ivecs 格式：每个近邻组 = [4字节近邻数k] + [k个4字节int32近邻ID]
    # 1. 获取每个查询的近邻数k（SIFT1M的groundtruth通常是每个查询100个近邻）
    k = int(gt[0])
    print(f"每个查询的近邻数k: {k}")
    
    # 验证数据长度是否符合格式
    assert gt.size % (k + 1) == 0, f"近邻文件损坏：总长度 {gt.size} 不是 (k+1)={k+1} 的整数倍"
    
    # 重构为 [查询数, k+1]，去掉第一列（近邻数标识），得到实际近邻ID
    gt = gt.reshape(-1, k + 1)[:, 1:]  # 真实近邻：(10000, 100)
    
    print("解析后数据形状:", xb.shape, xq.shape, gt.shape)
    return xb, xq, gt

# 测试加载（注意路径使用 raw 字符串或双反斜杠）
d = 128
nlist = 1024
m = 16 
nbits = 8
nprobe = 50
xb, xq, gt = load_sift1m(r"C:\Users\xiong\Desktop\easy-vecdb\easy-vecdb\data\sift")

# ---------------------- 创建并训练IVF-PQ索引 ----------------------
# 索引格式：IVF{nlist},PQ{m}，距离度量用L2（欧式距离）
index = faiss.index_factory(d, f"IVF{nlist},PQ{m}", faiss.METRIC_L2)

# IVF-PQ必须先训练（聚类中心+PQ量化器），训练数据需足够多（SIFT1M满足）
print("开始训练IVF-PQ索引...")
index.train(xb)
print("训练完成，开始添加数据库向量...")

# 添加向量到索引（IVF会将向量分配到对应的聚类中心）
index.add(xb)
print(f"向量添加完成：共添加 {index.ntotal} 个向量")

# 设置查询参数nprobe（影响查询性能）
index.nprobe = nprobe
print(f"查询参数nprobe设置为：{nprobe}")

# ---------------------- 保存索引 ----------------------
index_path = "IVF_PQ.index"
faiss.write_index(index, index_path)

# 计算索引文件大小
index_size = os.path.getsize(index_path) / 1024 / 1024  # 转为MB
print(f"\nIVF-PQ索引保存成功！")
print(f"索引路径：{index_path}")
print(f"索引大小：{np.round(index_size, 2)} MB")
```

运行结果：

```
向量维度d: 128
原始数据形状: (129000000,) (1290000,) (1010000,)
每个查询的近邻数k: 100
解析后数据形状: (1000000, 128) (10000, 128) (10000, 100)
开始训练IVF-PQ索引...
训练完成，开始添加数据库向量...
向量添加完成：共添加 1000000 个向量
查询参数nprobe设置为：50

IVF-PQ索引保存成功！
索引路径：IVF_PQ.index
索引大小：23.52 MB
```

**阶段 2：在线加载并检索（在线服务环境）**

```python
def calculate_recall(I_pred, I_true, k):
    """
    计算召回率：预测结果中命中真实近邻的比例
    参数：I_pred-模型预测的索引矩阵，I_true-精确结果的索引矩阵，k-近邻数
    返回：平均召回率
    """
    recall_list = []
    for i in range(len(I_pred)):
        pred_set = set(I_pred[i])
        true_set = set(I_true[i])
        hit = len(pred_set & true_set)
        recall = hit / k
        recall_list.append(recall)
    return np.mean(recall_list)


# 计算索引文件大小
index_size = os.path.getsize(index_path) / 1024 / 1024  # 转为MB
print(f"\nIVF-PQ索引保存成功！")
print(f"索引路径：{index_path}")
print(f"索引大小：{np.round(index_size, 2)} MB")

# 2. 加载已保存的索引
index = faiss.read_index(index_path)
print("索引加载成功，是否训练完成：", index.is_trained)  # 输出 True

# 3. 直接检索（无需重新训练）
k = 10
start_time = time.time()
D, I = index.search(xq[:100], k)
search_time = time.time() - start_time

# 验证精度（与离线训练时一致）
recall = calculate_recall(I, gt[:100], k)  # 复用第一章的 calc_recall 函数
print(f"在线检索时间：{search_time:.8f}s，召回率：{recall:.4f}")
```

运行结果

```
索引加载成功，是否训练完成： True
在线检索时间：0.00416613s，召回率：0.9700
```

## 3 GPU 加速 FAISS：实现高性能检索（选学）

FAISS 支持 GPU 加速，通过并行计算提升索引训练和检索速度（百万级数据检索可从秒级降至毫秒级），核心依赖 NVIDIA CUDA 架构。

**Tips:**官方`faiss-gpu`pip wheel 主要面向 Linux，对 Windows系统支持比较差

> 以下内容建议在linux系统上学习~

### 3.1 GPU 版本安装与环境验证

#### 3.1.1 安装步骤

```bash
#使用Conda安装（推荐）
# 创建并激活一个Conda环境（可选但推荐）
conda create -n faiss-gpu-env python=3.10
conda activate faiss-gpu-env

# 安装faiss-gpu包（会自动安装对应的CUDA运行库）
conda install -c pytorch -c nvidia faiss-gpu=1.11.0

#使用Pip安装
#如果你的系统已经配置好了CUDA环境，使用pip安装会更直接。
# 在激活的虚拟环境中，直接使用pip安装
pip install faiss-gpu

# 如果需要指定特定版本，例如
pip install faiss-gpu==1.7.1.post2
```

#### 3.1.2 环境验证

```python
import faiss

# 1. 检查 FAISS 是否支持 GPU
print("是否支持 GPU：", faiss.get_num_gpus() > 0)  # 输出 True 表示支持
print("GPU 数量：", faiss.get_num_gpus())

# 2. 验证 GPU 索引创建
d = 128
index_gpu = faiss.IndexFlatL2(d)
# 将 CPU 索引转换为 GPU 索引
res = faiss.StandardGpuResources()  # 管理 GPU 资源
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_gpu)  # 0 表示使用第 0 块 GPU
print("GPU 索引创建成功：", type(index_gpu))  # 输出 <class 'faiss.swigfaiss.GpuIndexFlatL2'>
```

运行结果

```
是否支持 GPU： True
GPU 数量： 8
GPU 索引创建成功： <class 'faiss.swigfaiss_avx512.GpuIndexFlat'>
```

### 3.2 IndexGPU 使用：单卡与多卡

#### 3.2.1 单卡使用（常用场景）

```python
import faiss
import numpy as np
import time

# ===================== 1. 基础配置与GPU初始化 =====================
d = 128               # 向量维度（需与你的数据一致）
n = 100000            # 基础向量总数（模拟数据）
nq = 100              # 查询向量数
gpu_id = 0            # 指定GPU卡号
k = 10                # Top-k检索结果
nlist = 1000          # IVF分区数（核心参数：建议 n/1000 ~ n/100）

# 初始化GPU资源（可选限制显存，避免OOM）
res = faiss.StandardGpuResources()
# res.setTempMemory(2 * 1024 * 1024 * 1024)  # 限制临时显存2GB

# ===================== 2. 生成模拟数据（替换为真实数据） =====================
np.random.seed(42)
# 训练数据（IVF类索引必须训练，数量≥10*nlist）
train_data = np.random.random((max(10*nlist, 50000), d)).astype('float32')
# 基础向量库（必须float32，FAISS强制要求）
base_data = np.random.random((n, d)).astype('float32')
# 查询向量
query_data = np.random.random((nq, d)).astype('float32')

# ===================== 3. 构建GPU版IVF_FLAT索引（核心） =====================
print("创建CPU版IVF_FLAT索引...")
# 正确语法：IVF{nlist},Flat（Flat表示原始向量存储，无量化）
cpu_index = faiss.index_factory(
    d, 
    f"IVF{nlist},Flat",  # 核心：IVF_FLAT组合
    faiss.METRIC_L2      # 距离度量：L2欧式距离（内积需改METRIC_INNER_PRODUCT）
)

# 转换为GPU索引（IVF_FLAT完美支持GPU）
gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

# 可选：设置IVF检索时的nprobe（探索分区数，越大精度越高、速度越慢，默认1）
gpu_index.nprobe = 10  # 推荐值：nlist/100 ~ nlist/10（如1000分区设10）

# ===================== 4. 训练索引（IVF类必须训练） =====================
print("训练IVF_FLAT索引...")
start = time.time()
gpu_index.train(train_data)  # 训练数据需覆盖数据分布
print(f"训练完成，耗时：{time.time()-start:.2f}s")

# ===================== 5. 添加向量到GPU索引 =====================
print("添加向量到GPU索引...")
start = time.time()
gpu_index.add(base_data)
print(f"添加完成！共{gpu_index.ntotal}个向量，耗时：{time.time()-start:.2f}s")

# ===================== 6. 执行GPU检索（无量化损失，精度最高） =====================
print(f"\n执行Top-{k}检索（IVF_FLAT，nlist={nlist}, nprobe={gpu_index.nprobe}）...")
start = time.time()
# D：距离矩阵（nq×k），I：下标矩阵（nq×k）
D, I = gpu_index.search(query_data, k)
print(f"检索完成！耗时：{(time.time()-start)*1000:.2f}ms，平均每查询：{(time.time()-start)/nq*1000:.2f}ms")

# ===================== 7. 输出检索结果 =====================
print("\n===== 检索结果示例（前5个查询的Top-5） =====")
for i in range(min(5, nq)):
    print(f"\n查询{i}：")
    print(f"  Top-5距离值：{np.round(D[i][:5], 4)}")
    print(f"  Top-5对应下标：{I[i][:5]}")

# ===================== 8. 可选操作：保存/加载索引 =====================
# 保存索引（需转回CPU）
# cpu_index_save = faiss.index_gpu_to_cpu(gpu_index)
# faiss.write_index(cpu_index_save, "ivf_flat_gpu.index")

# 加载索引
# cpu_index_load = faiss.read_index("ivf_flat_gpu.index")
# gpu_index_load = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index_load)
# gpu_index_load.nprobe = 10  # 加载后需重新设置nprobe
```

运行结果

```
创建CPU版IVF_FLAT索引...
训练IVF_FLAT索引...
训练完成，耗时：0.12s
添加向量到GPU索引...
添加完成！共100000个向量，耗时：0.03s

执行Top-10检索（IVF_FLAT，nlist=1000, nprobe=10）...
检索完成！耗时：2.25ms，平均每查询：0.02ms

===== 检索结果示例（前5个查询的Top-5） =====

查询0：
  Top-5距离值：[13.6454 13.842  14.5776 15.5522 15.6321]
  Top-5对应下标：[85484 25642 90796 49942 86371]

查询1：
  Top-5距离值：[14.1259 14.4648 14.8441 14.862  15.0519]
  Top-5对应下标：[54264 27473 63050 59408 34882]

查询2：
  Top-5距离值：[14.1538 14.5383 14.6415 14.9484 15.064 ]
  Top-5对应下标：[ 9141 31778 17320 38208 71374]

查询3：
  Top-5距离值：[14.0641 14.1665 14.6997 15.0549 15.2932]
  Top-5对应下标：[43768 14788 53221 50936 50594]

查询4：
  Top-5距离值：[11.2052 13.4151 14.39   14.4102 14.7487]
  Top-5对应下标：[12642 65914 93744  8293 21985]

```

#### 3.2.2 多卡使用（大规模数据场景）

多卡通过“索引分片”实现并行，将数据库向量分配到多个 GPU 上，检索时联合各卡结果。

```python
import faiss
import numpy as np
import time
import math
import os

# ===================== 1. 核心配置（双GPU+大规模） =====================
# 向量基础配置
d = 128                      # 向量维度
total_base_num = 5_000_000   # 总向量数（500万，可按需调整）
nq = 1000                    # 查询向量数
k = 20                       # Top-k检索
gpu_ids = [0, 1]             # 目标GPU卡号

# IVF_FLAT关键参数
nlist = 5000                 # IVF分区数（500万→5000，建议总向量数/1000）
nprobe = 50                  # 检索探索分区数（nlist/100）
batch_size = 100_000         # 分批添加批次大小（避免显存爆炸）

# ===================== 2. 强制指定双GPU=====================
# 关键：通过环境变量限制仅使用0、1号GPU（兼容所有版本）
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
print(f"已指定使用GPU：{gpu_ids}（CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}）")

# 初始化GPU资源（全局单例即可，多GPU自动共享）
res = faiss.StandardGpuResources()
# 限制GPU总临时显存（双GPU共享，8GB足够500万向量）
res.setTempMemory(8 * 1024 * 1024 * 1024)

# ===================== 3. 生成大规模模拟数据 =====================
np.random.seed(42)
# 训练数据（IVF必须训练，数量≥10*nlist）
train_data = np.random.random((max(10*nlist, 500_000), d)).astype('float32')
# 查询数据
query_data = np.random.random((nq, d)).astype('float32')

# ===================== 4. 构建IVF_FLAT索引 =====================
print("创建CPU版IVF_FLAT索引...")
# 核心：仅保留基础IVF_FLAT配置，避免版本敏感参数
cpu_index = faiss.index_factory(
    d, 
    f"IVF{nlist},Flat",
    faiss.METRIC_L2  # 欧式距离（内积需改METRIC_INNER_PRODUCT+归一化）
)

# 多GPU克隆配置（仅保留核心shard参数，移除所有敏感API）
cloner = faiss.GpuMultipleClonerOptions()
cloner.shard = True  # 按IVF分区分片到双GPU（并行效率最高）
cloner.useFloat16 = False  # 禁用float16，保证IVF_FLAT精度
# 移除所有GpuDeviceParams相关配置（无需手动指定设备）

# 分发索引到双GPU（兼容所有FAISS版本的最简写法）
print("将索引分发到指定双GPU...")
gpu_index = faiss.index_cpu_to_all_gpus(
    cpu_index,
    cloner  # 仅传递分片配置，无需其他参数
)

# 设置检索参数（提升精度）
gpu_index.nprobe = nprobe

# ===================== 5. 训练索引（双GPU自动并行） =====================
print(f"训练IVF_FLAT索引（nlist={nlist}）...")
start = time.time()
gpu_index.train(train_data)
train_time = time.time() - start
print(f"训练完成！耗时：{train_time:.2f}s ({train_time/60:.1f}min)")

# ===================== 6. 分批添加大规模向量 =====================
print(f"分批添加 {total_base_num} 个向量到双GPU索引...")
start = time.time()
total_batches = math.ceil(total_base_num / batch_size)

for batch_idx in range(total_batches):
    # 计算批次范围
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_base_num)
    # 生成当前批次向量（真实场景替换为文件/数据库加载）
    batch_data = np.random.random((end_idx - start_idx, d)).astype('float32')
    
    # 添加到双GPU索引（自动分片）
    gpu_index.add(batch_data)
    
    # 打印进度（每10批/最后一批）
    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        progress = (batch_idx + 1) / total_batches * 100
        elapsed = time.time() - start
        eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
        print(f"  批次 {batch_idx+1}/{total_batches} | 进度 {progress:.1f}% | 已耗时 {elapsed:.2f}s | 剩余 {eta:.2f}s")

add_time = time.time() - start
print(f"\n✅ 所有向量添加完成！索引总向量数：{gpu_index.ntotal}")
print(f"添加耗时：{add_time:.2f}s | 平均速度：{total_base_num/add_time:.0f} 向量/秒")

# ===================== 7. 双GPU并行检索 =====================
print(f"\n执行 {nq} 个查询的Top-{k}检索...")
start = time.time()
D, I = gpu_index.search(query_data, k)  # D=距离，I=下标
search_time = time.time() - start

# 打印检索性能
print(f"✅ 检索完成！")
print(f"总耗时：{search_time:.2f}s | 平均单查询耗时：{search_time/nq*1000:.2f}ms")
print(f"QPS（查询/秒）：{nq/search_time:.0f}")

# ===================== 8. 检索结果示例 =====================
print("\n===== 检索结果示例（前3个查询的Top-5） =====")
for i in range(min(3, nq)):
    print(f"\n查询{i} Top-5结果：")
    print(f"  距离值：{np.round(D[i][:5], 4)}")
    print(f"  向量下标：{I[i][:5]}")

# ===================== 可选：保存/加载索引 =====================
# # 保存索引（需转回CPU）
# cpu_index_save = faiss.index_gpu_to_cpu(gpu_index)
# faiss.write_index(cpu_index_save, "ivf_flat_multi_gpu.index")

# # 加载索引（重新分发到双GPU）
# cpu_index_load = faiss.read_index("ivf_flat_multi_gpu.index")
# gpu_index_load = faiss.index_cpu_to_all_gpus(cpu_index_load, cloner)
# gpu_index_load.nprobe = nprobe
```

运行结果

```
已指定使用GPU：[0, 1]（CUDA_VISIBLE_DEVICES=0,1）
创建CPU版IVF_FLAT索引...
将索引分发到指定双GPU...
训练IVF_FLAT索引（nlist=5000）...
训练完成！耗时：1.18s (0.0min)
分批添加 5000000 个向量到双GPU索引...
  批次 10/50 | 进度 20.0% | 已耗时 1.90s | 剩余 7.61s
  批次 20/50 | 进度 40.0% | 已耗时 3.64s | 剩余 5.46s
  批次 30/50 | 进度 60.0% | 已耗时 5.47s | 剩余 3.65s
  批次 40/50 | 进度 80.0% | 已耗时 6.87s | 剩余 1.72s
  批次 50/50 | 进度 100.0% | 已耗时 8.71s | 剩余 0.00s

✅ 所有向量添加完成！索引总向量数：5000000
添加耗时：8.71s | 平均速度：574213 向量/秒

执行 1000 个查询的Top-20检索...
✅ 检索完成！
总耗时：0.01s | 平均单查询耗时：0.01ms
QPS（查询/秒）：100019

===== 检索结果示例（前3个查询的Top-5） =====

查询0 Top-5结果：
  距离值：[14.6679 14.7715 15.0722 15.1892 15.4013]
  向量下标：[2682022 2900803 3671489 3657717 1205812]

查询1 Top-5结果：
  距离值：[12.7389 13.1389 13.3704 13.5513 13.7122]
  向量下标：[3727640 3850541 4201952 1708469 3119344]

查询2 Top-5结果：
  距离值：[11.8149 13.6595 13.8389 14.2032 14.2623]
  向量下标：[4565741 2941061 4406194 4841445 2768614]

```

### 3.3 实战：CPU VS GPU

为了能充分体现GPU加速的能力，实战部分使用flat算法进行向量检索

```python
import faiss
import numpy as np
import time
import os

# ===================== 1. 单GPU配置（核心简化） =====================
# 基础配置
d = 128                      # 向量维度
nq = 1000                    # 查询向量数（固定，保证对比公平）
k = 20                       # Top-k检索（暴力搜索返回精确结果）
gpu_id = 0                   # 单GPU卡号（固定为0）
test_scales = [10_000, 100_000, 500_000]  # 测试数据规模（1万/10万/50万）
repeat_times = 3             # 重复测试次数（取平均，减少误差）

# 强制指定仅使用0号GPU（避免占用其他卡）
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print(f"当前使用单GPU：{gpu_id}（CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}）")

# 初始化单GPU资源（限制显存，避免OOM）
gpu_res = faiss.StandardGpuResources()
gpu_res.setTempMemory(4 * 1024 * 1024 * 1024)  # 限制4GB临时显存

# ===================== 2. 性能测试核心函数（单GPU专属） =====================
def test_brute_force(
    vec_dim: int,
    base_num: int,
    query_num: int,
    top_k: int,
    device: str  # "cpu" / "gpu"
) -> dict:
    """
    测试单设备暴力搜索性能
    :return: 性能指标+检索结果（用于验证）
    """
    # 固定随机种子，保证CPU/GPU使用完全相同的测试数据
    np.random.seed(42)
    base_data = np.random.random((base_num, vec_dim)).astype('float32')
    query_data = np.random.random((query_num, vec_dim)).astype('float32')

    # 1. 创建暴力搜索索引（单GPU极简写法）
    if device == "cpu":
        index = faiss.IndexFlatL2(vec_dim)  # CPU暴力索引（L2欧式距离）
    elif device == "gpu":
        # 单GPU：CPU索引直接转GPU（核心简化点）
        cpu_index = faiss.IndexFlatL2(vec_dim)
        index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, cpu_index)
    else:
        raise ValueError(f"仅支持cpu/gpu，当前输入：{device}")

    # 2. 添加向量到索引
    index.add(base_data)

    # 3. 预热（避免首次初始化耗时干扰）
    if device == "gpu":
        index.search(query_data[:10], top_k)  # GPU预热

    # 4. 正式测试（多次运行取平均）
    total_time = 0.0
    final_D, final_I = None, None
    for _ in range(repeat_times):
        start = time.time()
        D, I = index.search(query_data, top_k)
        total_time += (time.time() - start)
        final_D, final_I = D, I  # 保留最后一次结果用于验证

    # 5. 计算核心性能指标
    avg_time = total_time / repeat_times  # 平均总耗时
    qps = query_num / avg_time            # 每秒查询数（QPS）
    per_query_ms = (avg_time / query_num) * 1000  # 单查询耗时（毫秒）

    return {
        "avg_total_time": round(avg_time, 4),
        "qps": round(qps, 0),
        "per_query_ms": round(per_query_ms, 4),
        "D": final_D,  # 距离矩阵（验证用）
        "I": final_I   # 下标矩阵（验证用）
    }

# ===================== 4. 主测试流程（单GPU vs CPU） =====================
if __name__ == "__main__":
    print("="*80)
    print("单GPU（0号卡） vs CPU 暴力搜索（Flat）性能对比")
    print(f"配置：维度={d} | 查询数={nq} | Top-k={k} | 重复测试={repeat_times}次")
    print(f"测试规模：{[f'{s:,}' for s in test_scales]} 条基础向量")
    print("="*80)

    # 逐规模测试
    for scale in test_scales:
        print(f"\n📊 测试规模：{scale:,} 条基础向量")
        print("-" * 60)

        # 1. CPU测试
        print("🔵 CPU暴力搜索：")
        cpu_perf = test_brute_force(d, scale, nq, k, "cpu")
        print(f"  平均总耗时：{cpu_perf['avg_total_time']}s")
        print(f"  QPS：{cpu_perf['qps']} 次/秒")
        print(f"  单查询耗时：{cpu_perf['per_query_ms']}ms")

        # 2. 单GPU测试
        print("🟢 单GPU暴力搜索：")
        gpu_perf = test_brute_force(d, scale, nq, k, "gpu")
        print(f"  平均总耗时：{gpu_perf['avg_total_time']}s")
        print(f"  QPS：{gpu_perf['qps']} 次/秒")
        print(f"  单查询耗时：{gpu_perf['per_query_ms']}ms")

        # 3. 性能对比
        print("📈 性能对比：")
        speedup = round(cpu_perf['avg_total_time'] / gpu_perf['avg_total_time'], 2)
        qps_ratio = round(gpu_perf['qps'] / cpu_perf['qps'], 2)
        print(f"  GPU相对CPU提速：{speedup}倍")
        print(f"  GPU QPS是CPU的：{qps_ratio}倍")

```

运行结果

```
当前使用单GPU：0（CUDA_VISIBLE_DEVICES=0）
================================================================================
单GPU（0号卡） vs CPU 暴力搜索（Flat）性能对比
配置：维度=128 | 查询数=1000 | Top-k=20 | 重复测试=3次
测试规模：['10,000', '100,000', '500,000'] 条基础向量
================================================================================

📊 测试规模：10,000 条基础向量
------------------------------------------------------------
🔵 CPU暴力搜索：
  平均总耗时：0.1028s
  QPS：9727.0 次/秒
  单查询耗时：0.1028ms
🟢 单GPU暴力搜索：
  平均总耗时：0.0007s
  QPS：1410166.0 次/秒
  单查询耗时：0.0007ms
📈 性能对比：
  GPU相对CPU提速：146.86倍
  GPU QPS是CPU的：144.97倍

📊 测试规模：100,000 条基础向量
------------------------------------------------------------
🔵 CPU暴力搜索：
  平均总耗时：0.7115s
  QPS：1405.0 次/秒
  单查询耗时：0.7115ms
🟢 单GPU暴力搜索：
  平均总耗时：0.0027s
  QPS：364500.0 次/秒
  单查询耗时：0.0027ms
📈 性能对比：
  GPU相对CPU提速：263.52倍
  GPU QPS是CPU的：259.43倍

📊 测试规模：500,000 条基础向量
------------------------------------------------------------
🔵 CPU暴力搜索：
  平均总耗时：8.6476s
  QPS：116.0 次/秒
  单查询耗时：8.6476ms
🟢 单GPU暴力搜索：
  平均总耗时：0.0124s
  QPS：80845.0 次/秒
  单查询耗时：0.0124ms
📈 性能对比：
  GPU相对CPU提速：697.39倍
  GPU QPS是CPU的：696.94倍

```

GPU 加速核心逻辑：通过并行计算同时处理多个查询和向量距离计算，突破 CPU 单线程瓶颈。

## 4 学习总结与常见问题

### 4.1 核心能力地图

- 复合索引：用 `index_factory` 组合 IVF/HNSW/PQ，通过 `nprobe` 调优精度与效率；
- 归一化：COSINE 相似度必做 L2 归一化，调用 `faiss.normalize_L2` 实现；
- 持久化：`write_index/read_index` 实现索引保存与加载，适配工程化；
- GPU 加速：通过 `index_cpu_to_gpu` 转换索引，多卡用 `index_cpu_to_all_gpus`。

### 4.2 常见问题解答

- **Q1：索引训练时报错“index not trained”？** A：IVF、PQ、HNSW 等索引需先调用 `train(xb)` 方法，且训练数据量需大于 IVF 分区数。
- **Q2：GPU 版本安装失败？** A：检查 CUDA 版本与 FAISS 版本兼容性，建议用 conda 安装或参考 [FAISS 官方安装指南](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)。
- **Q3：检索结果为空？** A：确认索引已添加数据（`index.ntotal > 0`），且 `nprobe` 不小于 1。