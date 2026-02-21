# 补充课程3：二进制向量与POPCNT优化深度剖析

## 课程目标
- 理解二进制向量在AI/搜索场景中的价值
- 深入掌握 POPCOUNT 的硬件演进与软件实现
- 逐行分析 `binary.h` 中的四种实现（Serial/NEON/SVE/AVX-512）
- 理解"31周期溢出防护"技巧的数学本质
- 掌握 Hamming 距离与 Jaccard 系数的位运算优化

**源码路径**: `include/simsimd/binary.h`（485行）

---

## 第一部分：为什么需要二进制向量？

### 1.1 应用场景

```
密集向量（Dense Vector）     二进制向量（Binary Vector）
┌───────────────────────┐   ┌─────────────────────────┐
│ [0.12, -0.87, 0.43,   │   │ 01101100 10011010 ...   │
│  0.31, 0.09, -0.55 …] │   │  8 bits/word            │
│  4 bytes/element      │   │                         │
│  1536 × 4 = 6144 B    │   │  1536 / 8 = 192 B       │
└───────────────────────┘   └─────────────────────────┘
                               32× 内存节省！
```

**典型用途**：
- **局部敏感哈希（LSH）**：将高维向量哈希为二进制码
- **学习型哈希（Learning to Hash）**：BERT/ResNet 等网络输出的紧凑表示
- **布隆过滤器（Bloom Filter）**：集合成员快速判断
- **特征图（Feature Maps）**：图像指纹（如 pHash、dHash）
- **遗传算法**：染色体表示

### 1.2 核心问题：比较两个二进制向量

给定两个二进制向量 A 和 B（每个 n_words 字节），需要高效计算：

```
Hamming 距离：differing bits = popcount(A XOR B)
Jaccard 系数：1 - popcount(A AND B) / popcount(A OR B)
```

性能关键：**POPCOUNT**（population count，计算1-bit的数量）

---

## 第二部分：POPCOUNT 的演进史

### 2.1 从软件到硬件

```
1950s-1990s：纯软件实现
│
├── 朴素循环（最慢）
│   unsigned count(unsigned x) {
│       int n = 0;
│       while (x) { n += x & 1; x >>= 1; }
│       return n;
│   }
│   代价：O(bits) = 64次迭代
│
├── Brian Kernighan 技巧（1988）
│   while (x) { x &= (x-1); n++; }
│   技巧：x & (x-1) 清除最低位的1
│   代价：O(popcount(x))，稀疏时快
│
├── 查找表法（SimSIMD实际使用）
│   见 simsimd_popcount_b8()
│   代价：1次内存访问
│
├── 2003年：首个硬件 POPCNT
│   AMD Barcelona (Phenom) - SSE4a 扩展
│
├── 2008年：Intel Nehalem
│   引入 POPCNT 指令（SSE4.2）
│   延迟：3周期，吞吐量：1/周期
│
└── 2019年：AVX-512 VPOPCNTDQ（Ice Lake）
    _mm512_popcnt_epi64：向量化 POPCNT
    1个指令处理8个64位整数的 POPCNT
    吞吐量：1/3 周期（vs 串行的 8/3 = 2.67 周期）
```

### 2.2 字节级查找表（SimSIMD 串行实现）

```c
// binary.h:79-90
SIMSIMD_PUBLIC unsigned char simsimd_popcount_b8(simsimd_b8_t x) {
    static unsigned char lookup_table[] = {
        // 256个条目，lookup_table[i] = popcount(i)
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,  // 0x00-0x0F
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,  // 0x10-0x1F
        // ... 共 256 项
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8   // 0xF0-0xFF
    };
    return lookup_table[x];
}
```

**表的构造原理**：
```
位置  二进制    popcount
0     00000000    0      ← 0个1
1     00000001    1      ← 1个1
2     00000010    1      ← 1个1
3     00000011    2      ← 2个1
4     00000100    1      ← 1个1
5     00000101    2      ← 2个1
...
255   11111111    8      ← 8个1
```

**性能特性**：
```
代价：~1-4周期（取决于缓存状态）
表大小：256字节 = 4个缓存行（64字节/行）
优点：简单、无分支、可预测
缺点：16位以上需要多次查询
```

**与4位（nibble）查找表对比**：

```
Nibble表（16项，常用于SIMD加速POPCNT）：
static const uint8_t nibble_table[16] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
};

// 对一个字节：
uint8_t hi = (byte >> 4) & 0x0F;
uint8_t lo = byte & 0x0F;
return nibble_table[hi] + nibble_table[lo];

优点：表只有16字节，完全在寄存器中（SIMD pshufb/tbl可直接实现）
缺点：需要两次查询和一次加法

// ARM NEON 利用 vcntq_u8 直接完成（硬件支持）
// x86 用 _mm_shuffle_epi8 (pshufb) 模拟 nibble 查表
```

---

## 第三部分：串行实现逐行分析

### 3.1 Hamming 距离串行版

```c
// binary.h:92-96
SIMSIMD_PUBLIC void simsimd_hamming_b8_serial(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,    // 字节数（不是位数）
    simsimd_distance_t *result)
{
    simsimd_i32_t differences = 0;
    for (simsimd_size_t i = 0; i != n_words; ++i)
        differences += simsimd_popcount_b8(a[i] ^ b[i]);
    //                                     ^^^^^^^^^^^
    //                                     XOR：不同位 → 1
    //                                     POPCOUNT：统计差异位数
    *result = differences;
}
```

**数学含义**：
```
a = 10110101
b = 11001100
      -------
XOR = 01111001  ← 有5个1 → Hamming距离 = 5
```

**为什么返回 f64 (simsimd_distance_t)**：

```c
// types.h
typedef simsimd_f64_t simsimd_distance_t;  // 统一接口：所有距离都是 f64

// 设计哲学：
// 1. Python/Rust 上层代码不需要关心具体类型
// 2. f64 精度足够表示 0~(n_words*8) 的整数（最大 64000+）
// 3. 统一函数指针签名（simsimd_metric_dense_punned_t）
```

### 3.2 Jaccard 系数串行版

```c
// binary.h:99-104
SIMSIMD_PUBLIC void simsimd_jaccard_b8_serial(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    simsimd_i32_t intersection = 0, union_ = 0;
    //                     ^^^^^
    //                     C 关键字冲突：union_ 而非 union
    for (simsimd_size_t i = 0; i != n_words; ++i)
        intersection += simsimd_popcount_b8(a[i] & b[i]),
        // AND：两者都为1 → 交集位
        union_ += simsimd_popcount_b8(a[i] | b[i]);
        // OR：至少一个为1 → 并集位

    *result = (union_ != 0) ? 1 - (simsimd_f64_t)intersection / (simsimd_f64_t)union_ : 1;
    //         ^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //         防止除零           Jaccard distance = 1 - Jaccard similarity
}
```

**数学关系**：
```
Jaccard Similarity = |A ∩ B| / |A ∪ B|
Jaccard Distance   = 1 - Jaccard Similarity

注意：两个全零向量 → union = 0 → 定义距离为 1（最远）
这是一种设计选择，也可以定义为 0（最近），但通常选择 1 以避免"空集相同"的歧义
```

---

## 第四部分：ARM NEON 实现深度分析

### 4.1 核心技巧：31周期溢出防护

这是 `binary.h` 中最精妙的设计之一。

```c
// binary.h:128-148
SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    simsimd_i32_t differences = 0;
    simsimd_size_t i = 0;

    while (i + 16 <= n_words) {
        uint8x16_t differences_cycle_vec = vdupq_n_u8(0);
        //         ^^^^^^^^^^^^^^^^^^^
        //         每个 lane 是 uint8，最大值 255

        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
        //                             ^^^^^^^^^^^^
        //                             关键限制！为什么是 31？

            uint8x16_t a_vec = vld1q_u8(a + i);   // 加载 16 字节
            uint8x16_t b_vec = vld1q_u8(b + i);   // 加载 16 字节
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            //                         ^^^^^^^   ^^^^^^^^
            //                         vcntq_u8: ARM NEON 原生 POPCOUNT
            //                         veorq_u8: XOR 运算
            differences_cycle_vec = vaddq_u8(differences_cycle_vec, xor_count_vec);
            //                      ^^^^^^^^
            //                      uint8 累加（每个 lane 最多累加 255）
        }
        differences += _simsimd_reduce_u8x16_neon(differences_cycle_vec);
        // 将 uint8x16 向量水平求和到 i32
    }

    // 处理尾部（不足16字节）
    for (; i != n_words; ++i)
        differences += simsimd_popcount_b8(a[i] ^ b[i]);

    *result = differences;
}
```

**31周期的数学证明**：

```
问题：uint8x16_t 向量，每个 lane 最大 255

每轮循环：
  - 处理 16 字节 = 16 × 8 = 128 位
  - 最坏情况：全部不同 → xor_count_vec 每个 lane 为 8
  - 累加到 differences_cycle_vec

31轮后每个 lane 的最大值：
  31 × 8 = 248 ≤ 255 ✓  （不溢出）

32轮后：
  32 × 8 = 256 > 255 ✗  （溢出！）

所以选择 31 是精确的溢出防护阈值。

┌─────────────────────────────────────────────────┐
│  cycle = 31 时最坏情况                           │
│  每个 lane 值 = 248（1111_1000 in binary）       │
│  uint8 最大值 = 255（1111_1111 in binary）       │
│  248 ≤ 255 ✓ 安全！                             │
└─────────────────────────────────────────────────┘
```

**31 × 16 = 496 字节 = 3968 位** 是每次内循环处理的最大向量大小。

### 4.2 辅助函数：uint8x16 水平归约

```c
// binary.h:113-126
SIMSIMD_INTERNAL simsimd_u32_t _simsimd_reduce_u8x16_neon(uint8x16_t vec) {

    // 步骤1：拆分为两半，扩展到 uint16
    uint16x8_t low_half  = vmovl_u8(vget_low_u8(vec));   // 低8个 u8 → 8个 u16
    uint16x8_t high_half = vmovl_u8(vget_high_u8(vec));  // 高8个 u8 → 8个 u16
    //         ^^^^^^^                                      零扩展，防溢出

    // 步骤2：合并（两个 u16x8 相加）
    uint16x8_t sum16 = vaddq_u16(low_half, high_half);
    // 此时每个 lane 最大：248 + 248 = 496 ≤ 65535 ✓

    // 步骤3-4：逐级归约到 u32
    uint32x4_t sum32 = vpaddlq_u16(sum16);  // 相邻配对相加，u16→u32
    uint64x2_t sum64 = vpaddlq_u32(sum32);  // 相邻配对相加，u32→u64
    simsimd_u32_t final_sum = vaddvq_u64(sum64);  // 水平求和
    return final_sum;
}
```

**归约过程可视化**：

```
输入（uint8x16）: [8, 8, 8, 8, 8, 8, 8, 8 | 8, 8, 8, 8, 8, 8, 8, 8]
                          × 31 次累加 = 248
               = [248, 248, 248, 248, 248, 248, 248, 248 | 248, 248, 248, 248, 248, 248, 248, 248]

vmovl + vaddq  = [496, 496, 496, 496, 496, 496, 496, 496]   uint16x8

vpaddlq_u16    = [992, 992, 992, 992]                        uint32x4

vpaddlq_u32    = [1984, 1984]                                uint64x2

vaddvq_u64     = 3968                                        u32
                 ^^^^
                 31 轮 × 16 字节 × 8位/字节 = 3968 位

```

### 4.3 vcntq_u8：ARM 的杀手锏

```
vcntq_u8（Count bits in 8-bit words）：

ARM Neon 硬件指令，原生支持 8 位 POPCOUNT

输入：  uint8x16_t [0b10110101, 0b11001100, ...]
输出：  uint8x16_t [        5,          4, ...]
        每个 lane 独立计算 bit count

指令特性（ARMv8.2-A）：
  汇编：  cnt v0.16b, v1.16b
  延迟：  2周期
  吞吐量：2/周期（2个执行单元）
  吞吐量（每元素）：1/4 周期

x86 没有等价指令（直到 Ice Lake 的 VPOPCNTDQ）！
这是 ARM 在二进制向量上的显著优势。
```

---

## 第五部分：ARM SVE 实现 - 自适应寄存器

```c
// binary.h:188-218
SIMSIMD_PUBLIC void simsimd_hamming_b8_sve(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    // SVE 策略：根据寄存器大小自适应
    simsimd_size_t const words_per_register = svcntb();
    //                                        ^^^^^^^^
    //                                        运行时查询寄存器字节数
    //                                        可能是 16, 32, 64, ... 字节

    if (words_per_register <= 32) {
        // 小寄存器：SVE 没有优势，退回 NEON
        simsimd_hamming_b8_neon(a, b, n_words, result);
        return;
    }

    // 大寄存器（>32字节）：SVE 更高效
    simsimd_size_t i = 0, cycle = 0;
    simsimd_i32_t differences = 0;
    svuint8_t differences_cycle_vec = svdup_n_u8(0);
    svbool_t const all_vec = svptrue_b8();  // 所有 lane 激活

    while (i < n_words) {
        do {
            // SVE 掩码加载：自动处理尾部
            svbool_t pg_vec = svwhilelt_b8((unsigned int)i, (unsigned int)n_words);
            //                ^^^^^^^^^^^^ 生成掩码：i < n_words 的位为1
            svuint8_t a_vec = svld1_u8(pg_vec, a + i);
            svuint8_t b_vec = svld1_u8(pg_vec, b + i);

            // XOR → POPCOUNT → 累加
            svuint8_t xor_vec = sveor_u8_m(all_vec, a_vec, b_vec);
            svuint8_t cnt_vec = svcnt_u8_x(all_vec, xor_vec);
            //         ^^^^^^^
            //         svcnt：SVE 中的 POPCOUNT（等价 NEON vcntq_u8）
            differences_cycle_vec = svadd_u8_z(all_vec, differences_cycle_vec, cnt_vec);

            i += words_per_register;
            ++cycle;
        } while (i < n_words && cycle < 31);
        //                       ^^^^^^^^^
        //                       同样的31周期溢出防护！

        differences += svaddv_u8(all_vec, differences_cycle_vec);
        //             ^^^^^^^^^
        //             SVE 水平归约（一步完成）
        differences_cycle_vec = svdup_n_u8(0);
        cycle = 0;
    }
    *result = differences;
}
```

**SVE 的优势**：
```
NEON（固定128位）：每轮处理 16 字节
SVE（512位）：     每轮处理 64 字节 → 同样31轮，处理 31×64 = 1984 字节

更大的向量 = 更少的循环迭代 = 更低的分支开销
```

---

## 第六部分：x86 AVX-512 Ice Lake 实现

### 6.1 VPOPCNTDQ 指令

```
Ice Lake (2019) 引入 AVX-512VPOPCNTDQ 扩展：

_mm512_popcnt_epi64(a)：
  ┌─────────────┐      ┌───────────────┐
  │  512位输入  │  →   │  8×64位输出   │
  │ [u64×8]     │      │  [u64×8]      │
  │             │      │  每个存放      │
  │             │      │  对应元素的    │
  │             │      │  bit count    │
  └─────────────┘      └───────────────┘

  汇编：VPOPCNTQ zmm0, zmm1
  延迟：3周期（Ice Lake），2周期（Zen4 Genoa）
  端口：1×p5（Ice Lake），1×FP01（Genoa）
```

### 6.2 分段展开优化

```c
// binary.h:271-330（ICE 版本）
SIMSIMD_PUBLIC void simsimd_hamming_b8_ice(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    simsimd_size_t xor_count;

    // 分段展开：根据向量大小选择不同的处理路径
    // 避免通用循环的分支预测开销

    if (n_words <= 64) {  // ≤ 512位
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        //                          ^^^^^^^^^ BMI2指令：清除高位
        //                          生成低 n_words 位为1的掩码

        __m512i a_vec = _mm512_maskz_loadu_epi8(mask, a);
        //              ^^^^^^^^^^^^^^^^^^^^^^^^
        //              掩码加载：只加载有效字节，其余置0
        __m512i b_vec = _mm512_maskz_loadu_epi8(mask, b);

        __m512i xor_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a_vec, b_vec));
        //                      ^^^^^^^^^^^^^^^^^^ VPOPCNTQ：向量化POPCOUNT
        //                                         ^^^^^^^^^^^^^^^^
        //                                         先XOR，再POPCOUNT
        xor_count = _mm512_reduce_add_epi64(xor_count_vec);
        //          ^^^^^^^^^^^^^^^^^^^^^^^^ 将8个u64求和

    } else if (n_words <= 128) {  // ≤ 1024位，双寄存器展开
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words - 64);
        __m512i a1_vec = _mm512_loadu_epi8(a);          // 完整加载第一块
        __m512i b1_vec = _mm512_loadu_epi8(b);
        __m512i a2_vec = _mm512_maskz_loadu_epi8(mask, a + 64);  // 掩码加载尾部
        __m512i b2_vec = _mm512_maskz_loadu_epi8(mask, b + 64);

        __m512i xor1_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a1_vec, b1_vec));
        __m512i xor2_count_vec = _mm512_popcnt_epi64(_mm512_xor_si512(a2_vec, b2_vec));
        xor_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(xor2_count_vec, xor1_count_vec));
        //  ^^^^^^^^^^^^^^^
        //  合并两块的结果

    } // ... 以此类推到 n_words <= 256
    else { /* 通用大向量循环 */ }

    *result = xor_count;
}
```

### 6.3 `_bzhi_u64` - 生成任意位掩码

```
_bzhi_u64(x, index)：
  = x & ((1 << index) - 1)
  = 保留 x 的低 index 位，清除高位

示例：_bzhi_u64(0xFFFFFFFFFFFFFFFF, 5)
  = 0xFFFFFFFFFFFFFFFF & ((1<<5) - 1)
  = 0xFFFFFFFFFFFFFFFF & 0x1F
  = 0x000000000000001F
  = 0b...0011111  ← 低5位为1

用于处理不是64字节倍数的向量时，生成精确掩码
```

### 6.4 分段展开的性能考量

```
策略：针对 n_words ≤ 64/128/192/256 分别展开

为什么这么设计？
1. 避免循环控制的开销（对短向量很显著）
2. 编译器可以直接分配固定数量的寄存器
3. 消除每轮的边界检查

代价：代码膨胀（但在 icache 中很可能全部命中）

对比通用循环版本（用于大向量）：
while (i + 64 <= n_words) {
    // 处理64字节块
    i += 64;
}
// 尾部用掩码处理
```

---

## 第七部分：Jaccard NEON 实现

```c
// binary.h:151-177
SIMSIMD_PUBLIC void simsimd_jaccard_b8_neon(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    simsimd_i32_t intersection = 0, union_ = 0;
    simsimd_size_t i = 0;

    while (i + 16 <= n_words) {
        uint8x16_t intersections_cycle_vec = vdupq_n_u8(0);
        uint8x16_t unions_cycle_vec = vdupq_n_u8(0);
        // 两个累加器：同时追踪交集和并集

        for (simsimd_size_t cycle = 0; cycle < 31 && i + 16 <= n_words; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(a + i);
            uint8x16_t b_vec = vld1q_u8(b + i);

            uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
            //                         ^^^^^^^   ^^^^^^^
            //                         POPCOUNT   AND：两者为1→交集
            uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
            //                                  ^^^^^^
            //                                  OR：至少一个为1→并集

            intersections_cycle_vec = vaddq_u8(intersections_cycle_vec, and_count_vec);
            unions_cycle_vec = vaddq_u8(unions_cycle_vec, or_count_vec);
            // 两个累加器独立累加（不相互依赖 → 更好的指令级并行）
        }

        intersection += _simsimd_reduce_u8x16_neon(intersections_cycle_vec);
        union_ += _simsimd_reduce_u8x16_neon(unions_cycle_vec);
    }

    for (; i != n_words; ++i)
        intersection += simsimd_popcount_b8(a[i] & b[i]),
        union_ += simsimd_popcount_b8(a[i] | b[i]);

    *result = (union_ != 0) ? 1 - (simsimd_f64_t)intersection / (simsimd_f64_t)union_ : 1;
}
```

**两路累加器的并行优势**：
```
and_count_vec 和 or_count_vec 的计算是独立的：
  - vandq_u8 使用 Port 0（NEON 整数单元）
  - vorrq_u8 使用 Port 0（NEON 整数单元，可能不同流水线）
  - 两个 vcntq_u8 可以乱序执行

最佳情况：两个管道同时计算，实际吞吐量接近 2 倍
```

---

## 第八部分：Haswell 实现的特殊设计

```c
// binary.h:对应 Haswell 版本
// 注意：Haswell 没有 AVX-512，也没有 VPOPCNTDQ
// 策略：使用标量 POPCNT 指令 + 循环展开

SIMSIMD_PUBLIC void simsimd_hamming_b8_haswell(
    simsimd_b8_t const *a,
    simsimd_b8_t const *b,
    simsimd_size_t n_words,
    simsimd_distance_t *result)
{
    // Haswell 策略：64位 POPCNT 指令一次处理 8 字节
    simsimd_size_t xor_count = 0;
    simsimd_size_t i = 0;

    // 每次处理 8 字节（64位）
    for (; i + 8 <= n_words; i += 8) {
        simsimd_u64_t a64, b64;
        memcpy(&a64, a + i, 8);
        memcpy(&b64, b + i, 8);
        xor_count += __builtin_popcountll(a64 ^ b64);
        //           ^^^^^^^^^^^^^^^^^^^^ GCC/Clang 内置
        //           映射到 POPCNT 指令（64位版本）
    }

    // 处理剩余字节（逐字节）
    for (; i != n_words; ++i)
        xor_count += simsimd_popcount_b8(a[i] ^ b[i]);

    *result = xor_count;
}
```

**POPCNT 指令特性（x86 Haswell）**：
```
POPCNT r64, r/m64：
  延迟：3周期
  吞吐量：1/周期（一个执行单元）
  端口：Port 1

vs AVX-512 VPOPCNTQ（Ice Lake）：
  VPOPCNTQ zmm, zmm：
  延迟：3周期
  吞吐量：1/3周期（一次处理8个64位值）
  速度提升：8x（理论峰值）
```

---

## 第九部分：四种实现性能对比

```
测试场景：1536维二进制向量（192字节 = 1536位）

Hamming 距离性能对比：
┌───────────────┬────────────┬──────────┬──────────────────────────┐
│ 实现          │ 吞吐量      │ 延迟      │ 备注                     │
├───────────────┼────────────┼──────────┼──────────────────────────┤
│ Serial        │ ~1 GB/s    │ ~192 ns  │ 字节查找表，无并行       │
│ Haswell       │ ~8 GB/s    │ ~24 ns   │ 64位POPCNT，8x并行       │
│ NEON          │ ~16 GB/s   │ ~12 ns   │ 128位向量，vcntq_u8      │
│ SVE(256位)    │ ~32 GB/s   │ ~6 ns    │ 256位向量，2x NEON       │
│ Ice Lake      │ ~64 GB/s   │ ~3 ns    │ 512位VPOPCNTQ，最快      │
└───────────────┴────────────┴──────────┴──────────────────────────┘

注：实际性能受内存带宽限制，峰值理论值
```

---

## 第十部分：实践练习

### 练习1：验证31周期溢出防护

```c
// 编写代码验证最坏情况下的溢出不发生
#include <stdint.h>
#include <assert.h>

void verify_overflow_protection(void) {
    // 最坏情况：31轮，每字节全部不同（XOR全1）
    uint8_t max_val = 0;
    for (int cycle = 0; cycle < 31; cycle++) {
        // 每个 lane，每轮最多增加 8（一个字节 8 位全部不同）
        max_val += 8;
    }
    // 31 × 8 = 248 ≤ 255
    assert(max_val <= 255);  // 不溢出
    // max_val += 8;  // 32轮：256 > 255，溢出！

    printf("最大值：%d，uint8最大：%d，安全：%s\n",
           max_val, 255, max_val <= 255 ? "是" : "否");
}
```

### 练习2：实现Nibble查找表版本

```c
// 比较：字节查找表 vs nibble查找表的L1缓存行为
#include <stdint.h>

// 方案A：256字节查找表（需要4个缓存行）
static uint8_t byte_table[256];  // 需初始化

// 方案B：16字节nibble查找表（1个缓存行内！）
static const uint8_t nibble_table[16] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
};

uint8_t popcount_nibble(uint8_t x) {
    return nibble_table[x & 0xF] + nibble_table[x >> 4];
}

// 讨论：什么场景下 nibble 表更快？
// 当处理大量不同字节时（256字节表会有缓存缺失）
// 当处理局部性差的数据时
```

### 练习3：理解 VPOPCNTQ 和 VPOPCNTB 的区别

```c
// Ice Lake 提供多种粒度的向量 POPCNT：
// _mm512_popcnt_epi64：对每个 64 位整数计算 popcount，输出 u64
// _mm512_popcnt_epi32：对每个 32 位整数计算 popcount，输出 u32
// _mm512_popcnt_epi16：对每个 16 位整数计算 popcount，输出 u16
// _mm512_popcnt_epi8：对每个  8 位整数计算 popcount，输出 u8

// 问题：为什么 binary.h 使用 _mm512_popcnt_epi64 而不是 _mm512_popcnt_epi8？
// 提示：累加器溢出 vs 指令吞吐量
// 答案见下：
//   epi8 版本：每个 lane 最大 8，31轮后最大 248 ≤ 255（不需要溢出防护）
//   epi64 版本：每个 lane 最大 64×8=512，一次处理更多数据
//   Ice Lake 上两者延迟相同，但 epi64 每次处理 8× 更多数据
```

---

## 总结

### 核心设计模式

1. **31周期技巧**：`uint8` 累加器，31轮后最大值 `31×8=248 ≤ 255`，精确防溢出
2. **两路累加器**：Jaccard 同时维护交集和并集，允许指令级并行
3. **分段展开**：ICE 版本对小向量（≤64/128/192字节）展开，消除循环开销
4. **SVE 自适应**：根据 `svcntb()` 决定是否使用 SVE 或退回 NEON
5. **掩码处理尾部**：`_bzhi_u64` + `_mm512_maskz_loadu_epi8` 消除标量尾部循环

### SIMD POPCOUNT 能力矩阵

```
指令集    │ POPCOUNT 指令      │ 粒度   │ 延迟  │ 吞吐量
──────────┼────────────────────┼────────┼───────┼────────
标量x86   │ POPCNT r64,r64     │ 64位   │ 3周期 │ 1/周期
NEON      │ VCNT.8B v0,v0      │ 8位×8  │ 2周期 │ 2/周期
SVE       │ CNT z0.b,p0/m,z0.b │ 8位×N  │ 2周期 │ 2/周期
AVX-512   │ VPOPCNTQ zmm,zmm   │ 64位×8 │ 3周期 │ 1/3周期
（Ice Lake）
```

---

**下一课程**: [补充4：运行时CPU检测与动态分发机制](./补充4_运行时CPU检测与动态分发.md)
