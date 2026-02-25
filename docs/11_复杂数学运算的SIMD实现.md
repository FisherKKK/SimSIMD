# 复杂数学运算的 SIMD 实现：Shuffle、对数与复数

> 本文是 SimSIMD 高性能编程讲解系列第十一篇。深入讲解三个高级主题：用 Shuffle 指令重排数据、用位操作近似 log₂、以及复数点积的两种实现方式对比。

---

## 一、数据重排：Shuffle 指令族

### 1.1 为什么需要 Shuffle？

SIMD 寄存器里的数据是固定顺序存储的。但很多算法要求把数据重新排列，比如：

- 复数运算：需要分离实部和虚部
- 矩阵转置：行列互换
- 对称操作：需要把元素"交叉"处理

**Shuffle 类指令**就是 SIMD 中的"搬运工"：在寄存器内部，或两个寄存器之间，按指定方式重排元素。

### 1.2 `_mm256_shuffle_epi8`：字节级随机重排

这是 AVX2 中最灵活的重排指令，可以按字节粒度任意重排：

```c
// dot.h 第 929-938 行（复数点积）
// swap_adjacent_vec 定义了重排规则
__m256i swap_adjacent_vec = _mm256_set_epi8(
    11, 10,  9,  8,   // 第4个f32 → 位置3
    15, 14, 13, 12,   // 第3个f32 → 位置4（交换！）
     3,  2,  1,  0,   // 第2个f32 → 位置1
     7,  6,  5,  4,   // 第1个f32 → 位置2（交换！）
    // 低128位和高128位各自重复以上规则
    11, 10,  9,  8,
    15, 14, 13, 12,
     3,  2,  1,  0,
     7,  6,  5,  4
);
```

**工作原理**：

```
输入寄存器（256位 = 8个f32 = [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]）：
  字节位置：[0..3]=ar0, [4..7]=ai0, [8..11]=ar1, [12..15]=ai1, ...（低128位）

swap_adjacent_vec 中的数字是"源字节位置"：
  目标字节0  ← 源字节7   (ai0的第4字节)
  目标字节1  ← 源字节6
  目标字节2  ← 源字节5
  目标字节3  ← 源字节4   → 目标0..3 = ai0（原来的第5-8字节）

  目标字节4  ← 源字节11
  ...                   → 目标4..7 = ar0（原来的第1-4字节）

效果：[ar0, ai0] → [ai0, ar0]（相邻f32交换了！）
```

**注意限制**：`_mm256_shuffle_epi8` 的 Shuffle 只能在 128 位 lane 内部操作，不能跨 lane（256 位寄存器分成两个 128 位 lane，字节索引 ≥16 会访问 lane 内的相对地址）。

### 1.3 `_mm256_permutevar8x32_ps`：f32 粒度跨 Lane 重排

```c
// dot.h 第 903-908 行（朴素方案的注释，已废弃）
__m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

// 输入：[ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]（4对复数）
// permute_vec 指定目标位置 i 从哪个源位置取：
//   目标位置 0 ← 源位置 0（ar0）
//   目标位置 1 ← 源位置 2（ar1）
//   目标位置 2 ← 源位置 4（ar2）
//   目标位置 3 ← 源位置 6（ar3）
//   目标位置 4 ← 源位置 1（ai0）
//   目标位置 5 ← 源位置 3（ai1）
//   ...

// 结果：[ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3]（实部/虚部分离）
__m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
```

**这个操作的代价**：
- `_mm256_permutevar8x32_ps` 延迟约 3 个周期
- 需要额外的 `_mm256_extractf128_ps` 分离高低 128 位
- 每次迭代需要 2 次 shuffle，代价很高

这就是为什么作者放弃了这个方案，改用 XOR 技巧。

---

## 二、复数点积：两种 SIMD 方案深度对比

### 2.1 复数乘法的数学基础

```
(ar + i·ai) × (br + i·bi) = (ar·br - ai·bi) + i(ar·bi + ai·br)
```

所以对多个复数对的点积（求和）：

```
Σ (ar·br - ai·bi)  →  实部和
Σ (ar·bi + ai·br)  →  虚部和
```

### 2.2 朴素 Shuffle 方案（已废弃，2.5 GB/s）

```c
// 注意：以下代码来自 dot.h 第 901-918 行的注释，是被废弃的方案

// 步骤1：用 permute 分离实虚部
__m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
// a_shuffled = [ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3]

// 步骤2：提取实部和虚部到单独寄存器
__m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0); // [ar0..ar3]
__m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1); // [ai0..ai3]
__m128 b_real_vec = ...;
__m128 b_imag_vec = ...;

// 步骤3：FMA + FMS
ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec); // ar*br
ab_real_vec = _mm_fnmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec); // -= ai*bi
ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);  // ar*bi
ab_imag_vec = _mm_fmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);  // += ai*br
```

**瓶颈分析**：

```
每次迭代的指令序列：
  2 × _mm256_permutevar8x32_ps (a和b各一次)  → 2条，延迟3周期
  4 × _mm256_extractf128_ps                  → 4条，延迟3周期
  4 × FMA/FMS                               → 4条，延迟4-5周期

总延迟：非常长，且 extract 指令吞吐量低（1条/周期）
结果：2.5 GB/s
```

### 2.3 XOR + Shuffle 优化方案（当前实现，5 GB/s）

**核心洞察**：不用提前分离实虚部，而是直接在交织存储的数据上做计算。

```c
// dot.h 第 926-950 行（simsimd_dot_f32c_haswell）

// 初始化
__m256 ab_real_vec = _mm256_setzero_ps();
__m256 ab_imag_vec = _mm256_setzero_ps();
__m256i sign_flip_vec = _mm256_set1_epi64x(0x8000000000000000ull);
// sign_flip_vec 用于翻转每个 64 位对中第二个 f32 的符号位
// 布局：[+, -, +, -, +, -, +, -]（+/-表示保持/翻转符号）

__m256i swap_adjacent_vec = _mm256_set_epi8(...);  // 相邻f32对调

// 主循环
for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
    // a_vec = [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]
    // b_vec = [br0, bi0, br1, bi1, br2, bi2, br3, bi3]
    __m256 a_vec = _mm256_loadu_ps((float *)(a_pairs + idx_pairs));
    __m256 b_vec = _mm256_loadu_ps((float *)(b_pairs + idx_pairs));

    // b_swapped: 相邻f32对调
    // b_swapped_vec = [bi0, br0, bi1, br1, bi2, br2, bi3, br3]
    __m256 b_swapped_vec = _mm256_castsi256_ps(
        _mm256_shuffle_epi8(_mm256_castps_si256(b_vec), swap_adjacent_vec)
    );

    // 计算实部点积的中间结果（不减，先全加）
    // ab_real += [ar0*br0, ai0*bi0, ar1*br1, ai1*bi1, ...]
    ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);

    // 计算虚部点积的中间结果
    // ab_imag += [ar0*bi0, ai0*br0, ar1*bi1, ai1*br1, ...]
    ab_imag_vec = _mm256_fmadd_ps(a_vec, b_swapped_vec, ab_imag_vec);
}

// 循环结束后，翻转 ab_real_vec 中的 ai*bi 项（奇数位置）
// sign_flip_vec = [+MSB, -MSB, +MSB, -MSB, ...]（每个64位对的高位为-）
// 对于 [ar0*br0, ai0*bi0, ...]，翻转第二个 → [ar0*br0, -(ai0*bi0), ...]
ab_real_vec = _mm256_castsi256_ps(
    _mm256_xor_si256(_mm256_castps_si256(ab_real_vec), sign_flip_vec)
);

// 规约：水平求和
simsimd_distance_t ab_real = _simsimd_reduce_f32x8_haswell(ab_real_vec);
simsimd_distance_t ab_imag = _simsimd_reduce_f32x8_haswell(ab_imag_vec);
```

### 2.4 关键步骤图解

**存储布局**：

```
交织存储：[ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]
                                                  (8个f32，256位)

a_vec: [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]
b_vec: [br0, bi0, br1, bi1, br2, bi2, br3, bi3]
```

**FMA 步骤1（实部中间值）**：

```
a_vec * b_vec = [ar0*br0, ai0*bi0, ar1*br1, ai1*bi1, ar2*br2, ai2*bi2, ar3*br3, ai3*bi3]
                 ↑正确项    ↑需要减   ↑正确项   ↑需要减  ...
```

**FMA 步骤2（虚部中间值）**：

```
b_swapped_vec = [bi0, br0, bi1, br1, ...]（实虚对调）

a_vec * b_swapped = [ar0*bi0, ai0*br0, ar1*bi1, ai1*br1, ...]
                     ↑正确项   ↑正确项  ...
```

**XOR 翻转**：

```
ab_real_vec（累加后）= [Σar*br, Σai*bi, Σar*br, Σai*bi, ...]
                        ↑每对的第1个f32    ↑每对的第2个f32（64位里）

sign_flip_vec       = [+, -, +, -, ...]（每个f64位置上的第2个f32取反）
// 实际上是：每个64位对里，低32位不变，高32位翻转符号

XOR 后               = [Σar*br, -Σai*bi, Σar*br, -Σai*bi, ...]
```

**水平规约（reduce_f32x8_haswell）**：

```
结果会把8个f32全部相加：
Σar*br + (-Σai*bi) + Σar*br + (-Σai*bi) + ...
= Σ(ar*br - ai*bi)  ← 正是实部的定义！✓
```

**性能对比**：

```
方案        每迭代关键操作                          吞吐量
朴素方案    2×permute + 4×extract + 4×FMA           2.5 GB/s
XOR方案    1×shuffle_epi8 + 2×FMA（+ 1次循环外XOR）  5.0 GB/s
```

### 2.5 共轭点积的变体（vdot）

`simsimd_vdot_f32c_haswell` 计算共轭点积（conjugate dot product）：

```
Σ conj(a) · b = Σ (ar - i·ai)(br + i·bi) = Σ(ar·br + ai·bi) + i·Σ(ar·bi - ai·br)
```

与普通点积的差别：实部是相加（不是相减），虚部是减法：

```c
// dot.h 第 988-994 行（vdot 版本）
for (...) {
    ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);         // ar*br + ai*bi
    b_vec = _mm256_castsi256_ps(_mm256_shuffle_epi8(...));            // 交换b的实虚
    ab_imag_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_imag_vec);         // ar*bi + ai*br
}

// 翻转 ab_imag 中的 ai*br 项（奇数位置）
ab_imag_vec = _mm256_castsi256_ps(
    _mm256_xor_si256(_mm256_castps_si256(ab_imag_vec), sign_flip_vec)
);
// → ab_imag = Σ(ar*bi - ai*br) ✓
```

---

## 三、NEON 复数点积：`vld2_s16` 的妙用

ARM NEON 版本采用了完全不同的策略：**用 `vld2` 加载时直接解交织**。

```c
// dot.h 第 465-476 行（simsimd_dot_f16c_neon）

while (count_pairs >= 4) {
    // vld2_s16：加载4对 f16，自动解交织成两个寄存器
    // a_vec.val[0] = [ar0, ar1, ar2, ar3]（实部）
    // a_vec.val[1] = [ai0, ai1, ai2, ai3]（虚部）
    int16x4x2_t a_vec = vld2_s16((short *)a_pairs);
    int16x4x2_t b_vec = vld2_s16((short *)b_pairs);

    // 类型转换：i16 重解释为 f16，再转 f32
    float32x4_t a_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[0]));
    float32x4_t a_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[1]));
    float32x4_t b_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[0]));
    float32x4_t b_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[1]));

    // 直接计算：实虚部已经分离，不需要 shuffle
    ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);   // ar*br
    ab_real_vec = vfmsq_f32(ab_real_vec, a_imag_vec, b_imag_vec);   // -= ai*bi
    ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);   // ar*bi
    ab_imag_vec = vfmaq_f32(ab_imag_vec, a_imag_vec, b_real_vec);   // += ai*br

    count_pairs -= 4, a_pairs += 4, b_pairs += 4;
}
```

**`vld2_s16` 的工作原理**：

```
内存中的 f16c 数据（交织存储）：
  [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]（8个 short）

vld2_s16 加载后自动解交织：
  .val[0] = [ar0, ar1, ar2, ar3]
  .val[1] = [ai0, ai1, ai2, ai3]
```

**ARM 的优势**：NEON 的 `vld2/vld3/vld4` 系列指令专为交织数据设计，在加载的同时完成解交织，代价极低（硬件层面完成，类似零开销）。

**x86 没有等价指令**：这就是 x86 版本需要用 Shuffle 重排的原因。

---

## 四、IEEE 754 位操作：提取指数和尾数

NEON 实现的 `log₂` 利用 IEEE 754 浮点数的位格式直接提取指数，无需任何复杂运算。

### 4.1 IEEE 754 f32 的位布局

```
f32 = [符号位(1)] [指数(8)] [尾数(23)]
       31         30..23    22..0

存储值 = (-1)^sign × 2^(exponent-127) × (1 + mantissa/2^23)
```

举例：
```
1.0 = 0_01111111_00000000000000000000000
        指数=127，存储指数=127，实际指数=127-127=0 → 1.0

2.0 = 0_10000000_00000000000000000000000
        指数=128，实际指数=128-127=1 → 2^1=2 ✓

0.5 = 0_01111110_00000000000000000000000
        指数=126，实际指数=126-127=-1 → 2^(-1)=0.5 ✓
```

### 4.2 NEON log₂ 实现（`probability.h` 第 142-165 行）

```c
SIMSIMD_PUBLIC float32x4_t _simsimd_log2_f32_neon(float32x4_t x) {

    // === 步骤1：提取指数 ===
    int32x4_t i = vreinterpretq_s32_f32(x);  // 把f32位模式解释为int32（零代价！）

    // 提取指数字段（位23-30）：
    //   与 0x7F800000 掩码 → 保留指数位
    //   右移23位 → 得到0-255的整数
    //   减127 → 得到实际指数（即 log₂ 的整数部分）
    int32x4_t e = vsubq_s32(
        vshrq_n_s32(vandq_s32(i, vdupq_n_s32(0x7F800000)), 23),
        vdupq_n_s32(127)
    );
    float32x4_t e_float = vcvtq_f32_s32(e);  // 整数 → 浮点

    // === 步骤2：提取尾数（归一化到 [1, 2) 区间）===
    // 把原始指数替换为127（即让实际指数=0），保留尾数
    // 结果：m ∈ [1.0, 2.0)
    float32x4_t m = vreinterpretq_f32_s32(
        vorrq_s32(
            vandq_s32(i, vdupq_n_s32(0x007FFFFF)),  // 只保留尾数位
            vdupq_n_s32(0x3F800000)                  // 设置指数=127（即值=1）
        )
    );

    // === 步骤3：Horner 多项式计算 log₂(m)，m∈[1,2) ===
    // log₂(m) = log₂(1 + (m-1)) ≈ 多项式近似
    // 系数由最小二乘拟合得到
    float32x4_t p = vdupq_n_f32(-3.4436006e-2f);      // 最高次系数
    p = vmlaq_f32(vdupq_n_f32(3.1821337e-1f),  m, p); // p = 0.318 + m*p
    p = vmlaq_f32(vdupq_n_f32(-1.2315303f),    m, p); // p = -1.23 + m*p
    p = vmlaq_f32(vdupq_n_f32(2.5988452f),     m, p); // p = 2.60 + m*p
    p = vmlaq_f32(vdupq_n_f32(-3.3241990f),    m, p); // p = -3.32 + m*p
    p = vmlaq_f32(vdupq_n_f32(3.1157899f),     m, p); // p = 3.12 + m*p

    // === 步骤4：最终结果 ===
    // log₂(x) = log₂(2^e × m) = e + log₂(m)
    // log₂(m) 用多项式近似：p * (m - 1)
    float32x4_t result = vaddq_f32(vmulq_f32(p, vsubq_f32(m, vdupq_n_f32(1.0f))), e_float);
    return result;
}
```

### 4.3 Horner 求值法详解

**朴素多项式求值**（效率低）：

```
p(x) = a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0

需要：5次乘幂 + 5次乘法 + 5次加法 = 大量运算
```

**Horner 法（嵌套乘法）**：

```
p(x) = a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5))))

从内到外展开，每步只需一次乘法一次加法！
总计：5次乘法 + 5次加法 = 10次运算（比朴素少5次乘法）
```

**代码中 Horner 法的执行顺序**：

```
初始：p = a5 = -3.4436006e-2

步骤1：p = a4 + m*p = 3.1821337e-1 + m*(-3.44e-2)
步骤2：p = a3 + m*p = -1.2315303 + m*(上一步结果)
步骤3：p = a2 + m*p = 2.5988452 + m*(...)
步骤4：p = a1 + m*p = -3.3241990 + m*(...)
步骤5：p = a0 + m*p = 3.1157899 + m*(...)

最终 p(m) 即为 log₂(m) 的多项式近似值
```

### 4.4 KL 散度的 log₂ 使用

```c
// probability.h 第 186-193 行
float32x4_t ratio_vec = vdivq_f32(
    vaddq_f32(a_vec, epsilon_vec),   // a + ε（防止除以0）
    vaddq_f32(b_vec, epsilon_vec)    // b + ε
);

// log₂(ratio)
float32x4_t log_ratio_vec = _simsimd_log2_f32_neon(ratio_vec);

// KL散度：Σ a * log(a/b)
// log(a/b) = log₂(a/b) * ln(2)，所以最后乘 ln(2)=0.693147
float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
sum_vec = vaddq_f32(sum_vec, prod_vec);

// ...循环结束后
simsimd_f32_t log2_normalizer = 0.693147181f;  // ln(2)
simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
```

**设计决策**：为什么用 log₂ 而不是直接用 ln？

```
log₂ 可以通过 IEEE 754 指数字段直接近似（整数部分精确！）
ln(x) = log₂(x) × ln(2)

在最后乘一次 ln(2)（常数，只做一次）比在循环里用 ln 更快。
```

---

## 五、SIMD 中的类型双关（Type Punning）

SimSIMD 大量使用"类型双关"——不改变位模式，只改变类型解释：

### 5.1 float ↔ integer 互转（零代价）

```c
// 把 f32 的位模式当 i32 用（不是数值转换！）
int32x4_t i = vreinterpretq_s32_f32(x);  // ARM
__m256i   i = _mm256_castps_si256(x);    // x86

// 把 i32 的位模式当 f32 用
float32x4_t f = vreinterpretq_f32_s32(i);  // ARM
__m256      f = _mm256_castsi256_ps(i);     // x86
```

**代价**：**零代价**。这些指令在大多数 CPU 上不生成任何实际汇编指令，只是告诉编译器如何解释寄存器内容。

### 5.2 类型双关的用途

```
用途                           代码                               目的
──────────────────────────────────────────────────────────────────────────
指数提取                       castps_si256 → and → shift          无需 FPU 参与
符号位翻转                     castps_si256 → xor → castsi256_ps   比乘以-1.0更快
掩码生成                       比较结果直接当整数掩码用             避免额外转换
bf16→f32                       cvtepu16_epi32 → slli → castsi256_ps 只需2条指令
```

### 5.3 f32 类型双关的安全性

```c
// 安全的类型双关（通过 union）
typedef union {
    simsimd_f32_t f;
    simsimd_u32_t i;
} simsimd_f32i32_t;

// types.h 第 488-492 行：快速平方根倒数
simsimd_f32i32_t conv;
conv.f = number;
conv.i = 0x5F1FFFF9 - (conv.i >> 1);  // 整数操作！
conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
```

**为什么用 union 而不是指针转换？**

C 语言中，通过不同类型的指针访问同一内存是**未定义行为（UB）**。但通过 union 的类型双关是 C 标准明确允许的（C99 §6.5.2.3）。

---

## 六、综合案例：log₂ 精度与性能权衡

```
精度需求        实现选择                    误差范围
──────────────────────────────────────────────────
最高精度        libm 的 logf()（精确）       < 0.5 ULP
中等精度        IEEE 指数+Horner 5次多项式   < 1%
低精度          Mercator 级数（3项）         < 5%（|x-1|<0.5时）
仅整数部分      直接取 IEEE 指数字段         ±1（整数误差）
```

**SimSIMD 的选择依据**：

```
KL/JS 散度用于相似度计算：
  - 最终结果用于排名，不需要绝对精确
  - 1% 的对数误差对相似度排名几乎无影响
  - 使用 IEEE 指数+Horner，速度是 logf 的 5-10 倍

串行（备用）实现：
  - 使用 Mercator 级数（x - x²/2 + x³/3），避免 LibC 依赖
  - 适用于 x ≈ 1 的情况（概率分布中常见）
```

---

## 七、小结

| 技术 | 指令 | 用途 | 性能影响 |
|------|------|------|---------|
| Shuffle 字节重排 | `_mm256_shuffle_epi8` | 交换相邻f32对（复数） | 1周期/次 |
| 跨Lane重排 | `_mm256_permutevar8x32_ps` | 分离实虚部（被废弃） | 3周期/次 |
| NEON 解交织加载 | `vld2_s16` | 自动分离实虚部 | 极低（硬件完成） |
| 类型双关 | `vreinterpretq_*` / `_mm256_cast*` | f32↔int32 | 零代价 |
| XOR 符号翻转 | `_mm256_xor_si256` | 取消减法 | 1周期/次 |
| 指数提取 | AND + SHR | 提取 IEEE 754 指数 | 2条指令 |
| Horner 法 | `vmlaq_f32` 链 | 多项式求值 | 最优乘法次数 |

核心设计哲学：**理解数据的位级表示，在位的层面做最少的工作**。

下一篇将是系列总结：**高性能 C 代码实战指南**，把所有技术汇总成可直接应用的方法论。
