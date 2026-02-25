# CPU 微架构与流水线深度解析

> 本文是 SimSIMD 高性能编程讲解系列第十三篇。从 CPU 硬件层面解释为什么代码快或慢——延迟、吞吐量、端口压力、乱序执行，这些概念直接决定了 SimSIMD 每一个优化决策。

---

## 一、两个最重要的性能指标

学习 CPU 微架构，必须先掌握两个概念。它们经常被混淆，但含义截然不同：

### 1.1 延迟（Latency）：等待时间

**延迟**是一条指令从开始到**产生结果**所需的时钟周期数。

```
示例：FMA 指令（_mm256_fmadd_ps）
  延迟 = 4 个周期

含义：
  周期1：指令开始执行
  周期2：还在执行...
  周期3：还在执行...
  周期4：结果就绪，后续指令才能使用这个结果
```

延迟决定了**依赖链**的速度。如果指令 B 依赖指令 A 的结果，必须等 A 完成后 B 才能开始。

### 1.2 吞吐量（Throughput）：并发能力

**吞吐量**是每个周期 CPU 能发出多少条同类型指令（也常用"倒数吞吐量"表示，即两次发射之间的最小间隔）。

```
示例：FMA 指令（在 Intel Skylake 上）
  吞吐量 = 每 0.5 周期 1 条（即每周期 2 条）

含义：
  周期1：发射 FMA_1
  周期1：同时发射 FMA_2（不等 FMA_1 完成！）
  周期1：甚至发射 FMA_3...
  只要 FMA_1 和 FMA_2 不互相依赖，CPU 可以同时处理多条！
```

吞吐量决定了**无依赖操作**的最大速度。

### 1.3 二者的关系

```
单累加器（性能受限于延迟）：
  acc = fmadd(a, b, acc)  ← 依赖上次的 acc
  // 即使 FMA 吞吐量=2/周期，因为依赖链，实际只有 1/4 周期
  // 有效吞吐量 = 1/延迟 = 1/4 周期

四个独立累加器（性能接近吞吐量上限）：
  acc0 = fmadd(a0, b0, acc0)  ← 依赖 acc0
  acc1 = fmadd(a1, b1, acc1)  ← 依赖 acc1（独立！）
  acc2 = fmadd(a2, b2, acc2)  ← 独立
  acc3 = fmadd(a3, b3, acc3)  ← 独立
  // 四个链并行，每个链延迟=4周期，但互相不等待
  // 有效吞吐量接近 4/4 = 1/周期（比单累加器快4倍）
```

**SimSIMD 用多个独立累加器（ab_vec, a2_vec, b2_vec）隐藏 FMA 延迟，正是利用了这一原理。**

---

## 二、超标量与乱序执行

### 2.1 超标量（Superscalar）

现代 CPU 每个周期可以**同时发射多条指令**——不是一次一条，而是多条并行。

```
Intel Skylake 每周期最多可以发射 6 条 µop（微操作）：
  Port 0: ALU, FP multiply, ...
  Port 1: ALU, FP add, FMA, ...
  Port 2: Load
  Port 3: Load
  Port 4: Store
  Port 5: Branch, Shuffle

假设一个循环迭代包含：
  2× FMA（走 Port 0 和 Port 1）
  2× Load（走 Port 2 和 Port 3）
  1× Store（走 Port 4）

理论上这 5 条指令可以在一个周期内全部发射！
```

### 2.2 端口压力（Port Pressure）

每种指令只能走特定的执行端口。如果循环中某类指令太多，对应端口会成为瓶颈。

```
示例（AVX512 Skylake）：
  FMA 指令走 Port 0 和 Port 1
  每周期最多 2 条 FMA

如果写了：
  acc = fma(a, b, acc)
  acc = fma(c, d, acc)  ← 同一个 acc，有依赖！

即使端口有空闲，依赖链也阻止了并发。
```

**SimSIMD 的余弦距离用 3 个累加器（ab, a2, b2），天然适配超标量架构**：
- ab 的 FMA 走 Port 0
- a2 的 FMA 走 Port 1
- b2 的 FMA 可能走 Port 0（如果 Port 0 空闲）
- 三者之间没有依赖，乱序执行引擎可以自由调度

### 2.3 乱序执行（Out-of-Order Execution）

CPU 不按程序顺序执行指令，而是动态分析依赖关系，把**没有依赖**的指令提前执行。

```
程序顺序：
  1. a_vec = load(a + i)        // 可能 cache miss，需要等待
  2. b_vec = load(b + i)        // 独立，可以同步执行
  3. acc = fmadd(a_vec, b_vec, acc)  // 依赖 1 和 2
  4. c_vec = load(c + i)        // 独立！不等 3 完成
  5. ...

乱序引擎的实际执行顺序：
  周期1：发射指令 1、2、4（三个独立的 load）
  周期3：load 1 和 2 完成，发射指令 3（FMA）
  周期4：load 4 完成，发射下一轮计算
  // CPU 自动利用了 load 的延迟"窗口"
```

**这就是为什么 SimSIMD 的循环要把 load 和 compute 写在一起**——编译器和 CPU 会自动把独立的 load 提前，隐藏内存延迟。

---

## 三、SimSIMD 中的微架构决策解析

### 3.1 为什么余弦距离比点积"快"？

看起来余弦距离比点积计算量更大（需要 3 个 FMA 而不是 1 个），但实际上吞吐量可能更高：

```c
// 点积（单累加器，受延迟限制）
ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
// 每次迭代必须等 4 周期，吞吐量上限 = 1/4

// 余弦距离（三个独立累加器，接近吞吐量上限）
ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);  // Port 0
a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);  // Port 1（同时进行）
b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);  // Port 0（或延迟后执行）
// 吞吐量上限 = 3/4 (FMA 每周期 2 条) → 1.5/周期
// 相比单累加器，每周期有效工作量提升了 6 倍！
```

**结论**：在 FMA 延迟受限的场景下，增加更多独立操作比减少操作更能提升吞吐量。

### 3.2 hadd 指令的端口分析

```c
// 水平加法（horizontal add）
sum = _mm_hadd_epi32(sum, sum);

// Intel 文档：
// hadd 走 Port 5（Shuffle 单元）
// 吞吐量：1 周期/指令（只有一个 Port 5）

// 普通 add 走 Port 0 或 Port 1
// 吞吐量：0.33 周期/指令（三个端口）

// 所以树形规约（用 add + shuffle）不一定比 hadd 慢！
// hadd 代码更简洁，但树形规约在某些情况下更快
```

SimSIMD 在规约中使用 `_mm_hadd_epi32` 是为了代码简洁性——两次 hadd 搞定，比树形规约少几行代码，且对小规模规约性能差异不显著。

### 3.3 `_mm256_lddqu_si128` vs `_mm256_loadu_si128`

```c
// dot.h 第 1157 行：
__m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx));
//                  ↑ 用 lddqu 而不是 loadu
```

**原因**：`LDDQU`（Load Unaligned Integer）是专为非对齐加载优化的指令。

```
在早期 Intel CPU（Core 2, Nehalem）：
  MOVDQU（loadu）：跨缓存行时可能需要 2 次内存访问
  LDDQU：专门优化了跨缓存行的情况，可能快 10-20%

在现代 CPU（Haswell+）：
  两者性能基本相同（硬件已优化非对齐访问）
  但 lddqu 保留了：向后兼容保证、语义更清晰

结论：写新代码用 loadu 即可，lddqu 是历史遗留优化
```

### 3.4 `_mm512_alignr_epi32`：旋转而不是 Shuffle

在 `sparse.h` 的稀疏向量交集实现中：

```c
// sparse.h（Ice Lake u16 交集）
__m512i a1 = _mm512_alignr_epi32(a, a, 4);   // a 旋转 4 个 i32 = 16 字节
__m512i a2 = _mm512_alignr_epi32(a, a, 8);   // a 旋转 8 个 i32
__m512i a3 = _mm512_alignr_epi32(a, a, 12);  // a 旋转 12 个 i32
```

**为什么用 `alignr` 而不是 `permute`？**

```
_mm512_alignr_epi32：1 个周期延迟（Haswell/Ice Lake）
_mm512_permutexvar_epi32：4-6 个周期延迟

对于简单旋转操作，alignr 快 4-6 倍！
alignr(a, a, n) 等价于把 a 和 a 拼接成 1024 位，右移 n×4 字节后取低 512 位
= 把 a 循环旋转 n 个 i32 位置
```

这里体现的原则：**在功能等价的前提下，选延迟最低的指令**。

---

## 四、分支预测的量化代价

### 4.1 分支预测失败的代价

```
现代 CPU 流水线深度：约 14-20 级（Skylake 约 14 级）

分支预测失败时：
  1. 清空流水线（flush pipeline）
  2. 丢弃已预取和部分执行的指令
  3. 从正确地址重新取指

代价：约 15-20 个时钟周期

数字化影响：
  如果循环每次迭代 1ns（1GHz），预测失败 = 额外 15-20ns
  对于一个 1000 次迭代的循环，如果有 5% 的预测失败：
    额外代价 = 50次 × 20周期 = 1000 周期
    比循环本身（1000周期）多了 100%！
```

### 4.2 什么时候分支预测会失败？

```
预测成功率高（CPU 能学会规律）：
  - 循环结束条件（几乎总是继续 → 最后一次才退出）
  - 固定模式的条件（每次都相同）

预测成功率低（难以预测）：
  - 数据相关的分支（a[i] > 0 在随机数据中）
  - 交替的条件（奇偶交替）
  - 非常短的循环（CPU 还没学会就结束了）
```

### 4.3 SimSIMD 的无分支策略

```c
// sparse.h：无分支的集合交集（第一篇已介绍）
intersection_size += (ai == bj);  // 用算术替代条件分支
i += (ai <= bj);
j += (ai >= bj);

// 用 SIMD 掩码替代标量分支（probability.h）
__mmask16 mask = _mm512_cmp_ps_mask(a_vec, epsilon_vec, _CMP_GE_OQ);
sum_vec = _mm512_mask3_fmadd_ps(a_vec, log_ratio_vec, sum_vec, mask);
// 等价于：if (a > epsilon) sum += a * log_ratio;
// 但完全无分支！
```

---

## 五、指令级并行（ILP）的极限

### 5.1 理论 ILP 上限

```
在 Skylake（6发射宽度）上：
  每周期最多发射 6 条 µop

  FMA 每周期 2 条（Port 0 + Port 1）
  Load 每周期 2 条（Port 2 + Port 3）
  Store 每周期 1 条（Port 4）
  ALU 每周期 3 条（Port 0, 1, 5）

对于一个理想的 SIMD 点积循环：
  每次迭代：2× Load + 1× FMA = 3 µop
  理论峰值：每 1 周期 1 次迭代
  制约因素：FMA 延迟（4周期）→ 需要 4 个独立累加器才能充分利用
```

### 5.2 实际 ILP：以 AVX2 f32 余弦距离为例

```c
// spatial.h NEON cosine（等价 AVX2 逻辑）
for (; i + 4 <= n; i += 4) {
    float32x4_t a_vec = vld1q_f32(a + i);    // Load  → 1 µop
    float32x4_t b_vec = vld1q_f32(b + i);    // Load  → 1 µop
    ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec); // FMA   → 1 µop (dep: ab)
    a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec); // FMA   → 1 µop (dep: a2, 独立于 ab)
    b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec); // FMA   → 1 µop (dep: b2, 独立于 ab/a2)
}

// 依赖图：
//   Load_a ────→ FMA_ab
//   Load_b ────→ FMA_ab
//   Load_a ────→ FMA_a2
//   Load_b ────→ FMA_b2
//
//   FMA_ab 只依赖 Load（延迟2-4周期），不依赖上次 FMA_ab（延迟4周期）
//   但实际上 FMA_ab 依赖上次的 ab_vec！这是 SimSIMD 唯一的串行依赖链。

// 分析：
//   三个 FMA 的依赖链各自独立（ab, a2, b2 互不相关）
//   每个链的延迟上限 = FMA 延迟 = 4 周期
//   但三个链可以交错，CPU 每 4/3 ≈ 1.33 周期完成一次循环迭代
```

### 5.3 Roofline 模型：计算密集 vs 内存密集

```
Roofline 模型：性能受限于两个"屋顶"
  1. 算术强度（Arithmetic Intensity）= FLOP / 内存字节
  2. 实际性能 = min(峰值 FLOP/s, 内存带宽 × 算术强度)

点积的算术强度：
  每次迭代：2 个 f32 load（8字节）+ 1 次乘法 + 1 次加法（2 FLOP）
  算术强度 = 2 FLOP / 8 字节 = 0.25 FLOP/byte
  （很低！内存密集型）

AVX2 f32 FMA 峰值：
  2 FMA/周期 × 8 f32/FMA × 2 FLOP/FMA × 3GHz ≈ 96 GFLOP/s

内存带宽（DDR4-3200 双通道）：
  ≈ 50 GB/s

实际性能上限 = min(96, 50×0.25) = min(96, 12.5) ≈ 12.5 GFLOP/s

→ 点积是内存受限的！即使 CPU 的 FMA 单元大量空闲。
→ 优化方向：减少内存访问（量化、使用 i8/f16）而不是减少 FLOP。
```

**这就是为什么 i8 量化能把性能提升 4 倍**：
- 数据量缩小 4 倍（f32 → i8）
- 内存带宽消耗减少 4 倍
- 在内存受限场景下，直接提升 4 倍性能

---

## 六、写代码时如何考虑微架构

### 6.1 查询指令延迟/吞吐量

**推荐资源**：
- [Agner Fog's Instruction Tables](https://agner.org/optimize/)：最全面的指令时序数据库
- [uops.info](https://uops.info/)：每条 x86 指令在各代 CPU 上的详细端口分配

```
如何读取 Agner Fog 表格：
  指令名    | Latency | Throughput | µops | Ports
  ─────────────────────────────────────────────────
  VFMADD    |    4    |    0.5     |  1   | FP01
  VMOVDQU   |    5    |    0.5     |  1   | MEM
  VPSHUFB   |    1    |    1.0     |  1   | 5
  VPCMPEQW  |    1    |    0.33    |  1   | 015

  Latency = 延迟（周期）
  Throughput = 倒数吞吐量（每条指令最少间隔周期）
  Ports = 可以执行的端口
```

### 6.2 快速估算循环性能

```
步骤1：列出循环体内的所有指令类型和数量
步骤2：按端口分类，计算每个端口的压力
步骤3：瓶颈 = max(各端口压力, 最长依赖链)

示例：AVX2 f32 点积
  2× VMOVAPS（Load，Port 2/3）→ Port 2+3 压力 = 1/次迭代
  1× VFMADD（FMA，Port 0/1） → Port 0+1 压力 = 0.5/次迭代

  最长依赖链：FMA 链，延迟 4 周期

  瓶颈 = max(Load 1次/迭代, FMA 4周期/次) = 4 周期/迭代（依赖链受限）

  用 4 个独立累加器后：
  最长依赖链变为 4 周期（4 个并行链，每链 4 周期，但交错执行）
  Load 2次/迭代 → Port 2+3 = 1 周期/迭代
  FMA 4次/迭代 → Port 0+1 = 2 周期/迭代（4 FMA / 2FMA每周期）
  瓶颈 = max(1, 2) = 2 周期/迭代（接近理论上限）
```

### 6.3 SIMD 代码性能检查清单

```
□ 累加器独立性：
    - 循环中是否有多个独立的 FMA 累加器？
    - 独立累加器数量 ≥ FMA 延迟（至少 4 个）？

□ 指令混合：
    - Load 和 FMA 是否交替出现（利用不同端口）？
    - 是否有不必要的 Shuffle（占用 Port 5）？

□ 内存访问：
    - 是否顺序访问（便于硬件预取）？
    - 是否有非对齐访问（考虑改用 loadu/storeu）？
    - 是否为内存密集型（算术强度 < 1）？如果是，优先减少数据量

□ 尾部处理：
    - 尾部循环是否对 SIMD 主循环有影响？
    - 是否可以用 AVX512 掩码消除尾部循环？

□ 分支：
    - 循环内是否有数据相关的分支？
    - 是否可以用 cmov 或 SIMD 掩码替代？
```

---

## 七、各代 CPU 的关键差异

| 代次 | 关键新增 | FMA 吞吐量 | Load 带宽 | 备注 |
|------|---------|-----------|-----------|------|
| Haswell (2013) | AVX2, FMA | 2条/周期（256位） | 2×256位/周期 | 首个真正的 AVX2 |
| Skylake (2015) | AVX512（某些SKU） | 2条/周期（512位） | 2×512位/周期 | AVX512 首发 |
| Ice Lake (2019) | VNNI, VPOPCNTDQ | 2条/周期（512位） | 同上 | i8 点积硬件加速 |
| Genoa (AMD 2022) | AVX512 BF16 | 4条/周期（512位） | 同上 | AMD 大幅提升 FMA |
| Sapphire (Intel 2023) | AMX, FP16 | 2条/周期（512位） | 同上 | 原生 f16 FMA |

**SimSIMD 的后端选择**正是基于这些代次的差异：每一代都有对应的实现，利用该代新增的指令集。

---

## 八、小结：微架构优化的核心方程

```
实际性能 = min(
    峰值算术吞吐量,
    内存带宽 × 算术强度
) / 流水线效率折损

其中：
  峰值算术吞吐量 = FMA数量 × 宽度 × 频率
  流水线效率折损 = 1 / (1 + 依赖等待/吞吐量)
  算术强度 = FLOP / 内存字节

提升性能的三条路径：
  1. 降低分子（减少依赖等待）→ 多累加器、Horner法
  2. 提升算术强度（减少内存访问）→ 量化、数据复用
  3. 使用更宽/更快的指令→ AVX512、VNNI、AMX
```

下一篇将讲解**内存系统与缓存行对齐优化**，进入"内存受限"场景的专项优化。
