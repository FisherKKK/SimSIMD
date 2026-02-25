# SimSIMD SIMD 指令集原理与向量化策略

> 本文是 SimSIMD 讲解系列第五篇，系统讲解 SIMD 的工作原理、各代 CPU 的 SIMD 特性，以及 SimSIMD 的向量化设计策略。

---

## 一、SIMD 是什么？为什么要用它？

### 1.1 标量（Scalar）vs 向量（Vector）计算

**标量计算**（普通方式）：
```
a = [1.0, 2.0, 3.0, 4.0]
b = [5.0, 6.0, 7.0, 8.0]

// 加法：一次一个
result[0] = a[0] + b[0] = 6.0   ← 第1个周期
result[1] = a[1] + b[1] = 8.0   ← 第2个周期
result[2] = a[2] + b[2] = 10.0  ← 第3个周期
result[3] = a[3] + b[3] = 12.0  ← 第4个周期
// 共4个时钟周期
```

**SIMD 计算**（向量化方式）：
```
SIMD 寄存器（256位，可放8个f32）：
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

// 向量加法（一条指令）：
[6.0, 8.0, 10.0, 12.0, ...]  ← 第1个周期，全部完成！
// 只需1个时钟周期！
```

**SIMD = Single Instruction, Multiple Data（单指令，多数据）**

### 1.2 SIMD 寄存器宽度

不同指令集的寄存器宽度：

```
SSE（2000年）：128位 = 4个f32 或 2个f64
AVX（2011年）：256位 = 8个f32 或 4个f64
AVX2（2013年）：256位 + 整数指令增强
AVX512（2017年）：512位 = 16个f32 或 8个f64

ARM NEON：128位 = 4个f32
ARM SVE：可变（128~2048位），同一代码适配所有宽度
```

---

## 二、x86 SIMD 指令集代际详解

### 2.1 Haswell（AVX2，2013年）—— SimSIMD 的 x86 基准线

**关键能力**：
- AVX2：256位整数运算（不只是浮点）
- FMA：融合乘加 `a×b+c` 一条指令完成（比分开两条精度更高）
- F16C：f16 ↔ f32 转换指令

**SimSIMD 中的体现**：

```c
// types.h / simsimd.h
simsimd_cap_haswell_k = 1 << 10  // Haswell 的能力标志

// spatial.h：Haswell 特化函数
SIMSIMD_PUBLIC void simsimd_l2_i8_haswell(...)
SIMSIMD_PUBLIC void simsimd_cos_i8_haswell(...)
SIMSIMD_PUBLIC void simsimd_l2_f16_haswell(...)
```

**为什么从 Haswell 开始？**
- Haswell 于 2013 年推出，截止2024年，绝大多数 x86 机器都支持 AVX2
- 注释说："至少有一款 Haswell 处理器（Pentium G3420）直到2022年还在销售"

### 2.2 Skylake（AVX512 基础，2015年服务器）

**关键能力**：
- AVX512F：512位浮点运算
- **掩码操作（Masked Operations）**：用 k 寄存器控制哪些元素参与计算
- 完全消除尾部循环（对齐问题）

**掩码操作演示**：

```c
// 传统方式：向量16个元素，数组只有11个元素
for (i = 0; i < 16; i += 16) {
    __m512 v = _mm512_loadu_ps(a + i);  // 危险：可能越界读！
}
// 尾部（剩余3个）：
for (i = 16; i < 19; i++) { sum += a[i] * b[i]; }  // 标量处理

// Skylake 掩码方式：
__mmask16 mask = _cvtu32_mask16((1u << n) - 1);  // n=11 → 低11位为1
__m512 v = _mm512_maskz_loadu_ps(mask, a);  // 只加载有效位，其余为0
// 一次循环搞定，无需尾部处理！
```

### 2.3 Ice Lake（AVX512 + VNNI，2019年）

**关键新指令**：
- `VNNI`（Vector Neural Network Instructions）：`_mm512_dpbusd_epi32`
  - 一条指令：对 16 组 4 字节计算 `uint8×int8` 的点积，累加到 32 位
  - 专门为神经网络整数量化优化
- `VPOPCNTDQ`：向量化 popcount，同时对 8 个 64 位整数计数 1

```
VNNI 原理：
输入：4个uint8 × 4个int8 → 32位整数
[a0,a1,a2,a3] × [b0,b1,b2,b3] = a0*b0 + a1*b1 + a2*b2 + a3*b3

一条 dpbusd 指令处理16组，等效于64次乘法和累加！
```

### 2.4 Genoa（AMD Zen4，BF16，2023年）

**关键新指令**：
- BF16 算术：`_mm512_dpbf16_ps`
  - 从 BF16 输入直接计算到 f32 累加器
  - 无需手动转换！

```c
// Genoa 后端中的 BF16 点积（dot.h 中）
// _mm512_dpbf16_ps：BF16 点积到 f32
// 一次处理 32 个 BF16（16 对），累加到 16 个 f32
```

### 2.5 Sapphire Rapids（Intel，FP16，2023年）

**关键新指令**：
- FP16 算术：`_mm512_fmadd_ph`、`_mm512_add_ph` 等
- 原生 16 位浮点运算：不再需要先转换到 f32！
- 同时处理 32 个 f16（一个 512 位寄存器）

```
对比：
Haswell f16 处理：f16→f32→计算→结果（需要转换）
Sapphire f16 处理：f16 直接参与运算，速度提升约2倍
```

### 2.6 Sierra Forest（现代 Intel 效率核，AVX2+VNNI）

```c
simsimd_cap_sierra_k = 1 << 16

// 专门为使用 VNNI 的 i8 量化计算设计
// AVX2（256位）+ VNNI 整数点积指令
```

---

## 三、ARM SIMD 指令集详解

### 3.1 NEON（ARM 的 SSE/AVX2 等价物）

- 128 位固定宽度寄存器
- ARM v8+ 的**标配**（所有现代 ARM 手机、服务器）
- 32 个 128 位向量寄存器（v0~v31）

**常用 NEON 类型名称**：

```c
float32x4_t  // 4个f32的向量
float64x2_t  // 2个f64的向量
int8x16_t    // 16个i8的向量
uint8x16_t   // 16个u8的向量
int16x8_t    // 8个i16的向量
int32x4_t    // 4个i32的向量
```

**命名规律**：`<类型><位宽>x<个数>_t`

**常用 NEON 操作**：

| 函数 | 操作 | 类比 C 操作 |
|------|------|-----------|
| `vdupq_n_f32(x)` | 广播：将 x 复制到4个位置 | `{x,x,x,x}` |
| `vld1q_f32(ptr)` | 加载4个f32 | `memcpy` |
| `vst1q_f32(ptr,v)` | 存储4个f32 | `memcpy` |
| `vaddq_f32(a,b)` | 向量加法 | `a+b` |
| `vsubq_f32(a,b)` | 向量减法 | `a-b` |
| `vmulq_f32(a,b)` | 向量乘法 | `a*b` |
| `vfmaq_f32(acc,a,b)` | 融合乘加 acc+a×b | `acc+a*b` |
| `vaddvq_f32(v)` | 水平求和 | `v[0]+v[1]+v[2]+v[3]` |
| `vmaxq_f32(a,b)` | 逐元素最大值 | `max(a,b)` |
| `vabdq_s8(a,b)` | 绝对差值 | `abs(a-b)` |
| `vcntq_u8(v)` | 逐字节 popcount | bitcount |
| `vdotq_s32(acc,a,b)` | i8 点积到 i32 | 4个i8的乘积和 |

### 3.2 i8 点积的特殊优化：vdotq

对于 i8（8位整数）计算，有一个专门的指令 `vdotq_s32`（需要 dotprod 扩展）：

```c
// spatial.h 第 712-733 行
// i8 余弦距离（NEON 实现）
for (; i + 16 <= n; i += 16) {
    int8x16_t a_vec = vld1q_s8(a + i);  // 加载16个i8
    int8x16_t b_vec = vld1q_s8(b + i);  // 加载16个i8

    // vdotq_s32：把16个i8看成4组（每组4个），
    // 每组计算 4 个 i8 的点积，累加到对应的 i32 通道
    // 结果：4个i32的累加器
    ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
    b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
}
```

**为什么要用 vdotq？**

注释中明确说明了性能对比（Graviton 4）：
```
朴素方法（i8→i16→i32）：17 GB/s 吞吐量
vabdq_s8 方法：33 GB/s 吞吐量
vdotq 方法：39 GB/s 吞吐量（最快）
```

`vdotq_s32` 一条指令完成 16 个 i8 的乘法累加（等效于 16 次乘法和 16 次加法），利用率极高。

### 3.3 绝对差值技巧：vabdq_s8

```c
// i8 L2 距离（spatial.h 第 589-614 行）
for (; i + 16 <= n; i += 16) {
    int8x16_t a_vec = vld1q_s8(a + i);
    int8x16_t b_vec = vld1q_s8(b + i);

    // vabdq_s8：同时计算 16 个 abs(a-b)，结果是 uint8
    // 注意：差的绝对值在 [0,255]，可以用 uint8 存
    uint8x16_t d_vec = vreinterpretq_u8_s8(vabdq_s8(a_vec, b_vec));

    // vdotq_u32：把每组4个 uint8 的平方累加到 uint32
    // d_vec * d_vec = |a-b|² 的累积
    d2_vec = vdotq_u32(d2_vec, d_vec, d_vec);
}
```

**关键洞察**：`vabdq_s8` 计算的是 `|a-b|`，虽然结果应该是 `uint8`，但通过 `vreinterpretq_u8_s8` 重新解释（不做实际转换，只改类型），就能直接用于 `vdotq_u32`。这省去了一次数据类型转换。

### 3.4 SVE（可变长度 SIMD）

```c
// SVE 的核心特性：svcntb() 在运行时返回寄存器宽度（字节数）
simsimd_size_t const vl = svcntb();  // 可以是 16, 32, 64, ... 字节

// SVE 的 predicate（谓词）寄存器：控制哪些元素参与计算
svbool_t pg = svwhilelt_b8_u64(i, n);  // 创建掩码：前 min(vl, n-i) 个元素有效

// SVE 加法（自动处理不同宽度）
svfloat32_t a_vec = svld1_f32(pg, a + i);  // 带掩码加载
svfloat32_t b_vec = svld1_f32(pg, b + i);
```

**SVE 的革命性设计**：
- 同一份代码在 128 位到 2048 位的 CPU 上都能运行
- CPU 自动决定每次处理多少元素
- 写一次，所有 SVE CPU 自动优化

---

## 四、编译目标隔离：#pragma push/pop

SimSIMD 的一个关键技术是**在同一文件中为不同函数指定不同的编译目标**：

```c
// 方式1：GCC 的 push_options / pop_options
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")

// 这里的函数使用 dotprod + i8mm 指令编译
void simsimd_cos_i8_neon(...) { ... }

#pragma GCC pop_options  // 恢复之前的编译选项

// 方式2：Clang 的 attribute push/pop
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

// 这里的函数使用 fp16 指令编译
void simsimd_dot_f16_neon(...) { ... }

#pragma clang attribute pop
```

**这意味着**：

```
同一个编译单元（.h 文件）中：

simsimd_dot_f32_serial   → 不使用任何 SIMD 指令（基础 C）
simsimd_dot_f32_neon     → 使用 NEON 指令（+simd）
simsimd_dot_f16_neon     → 使用 NEON + FP16（+simd+fp16）
simsimd_dot_i8_neon      → 使用 NEON + dotprod（+dotprod）
simsimd_dot_f32_haswell  → 使用 AVX2（-mavx2 -mfma）
simsimd_dot_f32_skylake  → 使用 AVX512（-mavx512f）
```

所有这些在同一个文件中，用不同的 `#pragma target` 隔离，不会互相干扰。

---

## 五、融合乘加（FMA）：精度与性能的双重收益

### 5.1 数学背景

```
传统两步：r = a * b + c
  1. temp = a * b  → 中间结果可能因精度限制被截断
  2. r = temp + c  → 再次截断

FMA一步：r = fma(a, b, c) = a*b+c（无中间截断）
```

**精度提升**：FMA 使用无限精度的中间结果，最终只截断一次。对于余弦距离的计算，这减少了累积误差。

### 5.2 FMA 在 SimSIMD 中的应用

```c
// spatial.h 第 331 行（NEON f32 L2）
sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
//        ↑ 等价于 sum_vec = sum_vec + diff_vec * diff_vec
//        使用单条 FMLA 指令，精度更高，且不需要两条指令

// x86 AVX2 中：
sum = _mm256_fmadd_ps(diff, diff, sum);
//    等价于 sum = diff * diff + sum
```

**性能优势**：FMA 是一条指令，占用一个周期（在配备 FMA 单元的 CPU 上）；分开的乘法和加法是两条指令，需要两个周期，且有数据依赖延迟。

---

## 六、静态断言：编译期的正确性保证

```c
// types.h 第 400-415 行
#define SIMSIMD_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]

SIMSIMD_STATIC_ASSERT(sizeof(simsimd_f32_t) == 4, simsimd_f32_t_must_be_4_bytes);
SIMSIMD_STATIC_ASSERT(sizeof(simsimd_f16_t) == 2, simsimd_f16_t_must_be_2_bytes);
// ...
```

**工作原理**：
- 如果 `cond` 为真（比如 `sizeof(float)==4`）：创建大小为1的数组（合法）
- 如果 `cond` 为假（比如在某个奇怪平台 `float` 是8字节）：创建大小为 -1 的数组（**编译错误**！）

这在**编译时**就能捕获类型大小不正确的问题，不需要运行程序才发现。

---

## 七、类型转换的精细处理

### 7.1 f16 ↔ f32 转换

```c
// types.h 第 424-431 行

// 如果平台支持原生 f16：
#if SIMSIMD_NATIVE_F16
#define SIMSIMD_F16_TO_F32(x) (*(x))  // 直接解引用，编译器知道怎么转

// 如果不支持（大多数平台）：
#else
#define SIMSIMD_F16_TO_F32(x) (simsimd_f16_to_f32(x))  // 调用软件实现

// 软件实现的核心（位操作）：
// IEEE 754 f16：s(1) + exp(5) + mantissa(10)
// IEEE 754 f32：s(1) + exp(8) + mantissa(23)
// 只需把 exp 的偏置从15调整到127，扩展尾数
#endif
```

### 7.2 f32 → i8 转换（量化）

```c
// types.h 第 448-451 行
#define SIMSIMD_F32_TO_I8(x, y) \
    *(y) = (simsimd_i8_t)(               \
        (x) > 127  ? 127  :              \  // 饱和上限
        (x) < -128 ? -128 :              \  // 饱和下限
        (int)((x) + ((x) < 0 ? -0.5f : 0.5f)) // 四舍五入
    )
```

这个宏实现了带饱和的四舍五入量化：
- 超出范围的值被"夹紧"（saturation）到边界
- 正数加 0.5 再取整 = 四舍五入
- 负数减 0.5 再取整 = 四舍五入（向负方向）

---

## 八、各代硬件的最大向量化吞吐量

| 平台 | 寄存器宽度 | f32 并行度 | i8 并行度 |
|------|-----------|-----------|---------|
| 标量 | 32位 | 1 | 1 |
| ARM NEON | 128位 | 4 | 16 |
| x86 AVX2 | 256位 | 8 | 32 |
| ARM SVE (256位) | 256位 | 8 | 32 |
| x86 AVX512 | 512位 | 16 | 64 |
| ARM SVE (512位) | 512位 | 16 | 64 |
| x86 AVX512+VNNI | 512位 | — | 256（i8点积）|

**理论加速比（相对标量）**：

- NEON f32：4x
- AVX2 f32：8x
- AVX512 f32：16x
- AVX512 i8（VNNI）：最高 256x（每周期处理256个乘法）

---

## 九、内存访问模式对性能的影响

SIMD 指令要求数据连续存储在内存中（对齐或未对齐加载）：

```c
// 对齐加载（要求地址是 16/32/64 字节对齐）：
_mm256_load_ps(ptr)    // AVX2 对齐加载（违反则崩溃）

// 非对齐加载（SimSIMD 默认用这个，更安全）：
_mm256_loadu_ps(ptr)   // AVX2 非对齐加载（在现代CPU上代价极小）
vld1q_f32(ptr)         // ARM NEON 非对齐加载（NEON 本身支持非对齐）
```

SimSIMD 使用 `u`（unaligned，非对齐）版本的加载指令，因为：
- 用户数据可能不保证对齐
- 现代 CPU 上非对齐加载的开销可以忽略不计（L1 缓存命中时）

---

## 十、小结：SimSIMD 的向量化层次

```
硬件能力层（越往下越快）：
  ┌─────────────────────────────────────┐
  │ Serial（标量）                       │ 所有 CPU
  ├─────────────────────────────────────┤
  │ NEON（128位）                        │ ARM v8+
  ├─────────────────────────────────────┤
  │ Haswell（AVX2，256位）               │ Intel/AMD 2013+
  ├─────────────────────────────────────┤
  │ SVE（可变宽度）                      │ ARM Graviton3+
  ├─────────────────────────────────────┤
  │ Skylake（AVX512，512位）             │ Intel 2015+ 服务器
  ├─────────────────────────────────────┤
  │ Ice Lake（AVX512+VNNI，整数加速）    │ Intel 2019+
  ├─────────────────────────────────────┤
  │ Genoa（AVX512+BF16）                │ AMD 2023+
  ├─────────────────────────────────────┤
  │ Sapphire（AVX512+FP16，最新）        │ Intel 2023+
  └─────────────────────────────────────┘

关键技术组合：
  向量化 × FMA × 掩码操作 × 点积指令 = 最大性能
```

下一篇将讲解 **Python、JavaScript 等多语言绑定的实现机制**。
