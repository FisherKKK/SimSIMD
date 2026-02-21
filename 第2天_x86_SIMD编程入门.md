# 第2天：x86 SIMD编程入门 - AVX2与内积运算

## 课程目标
- 掌握x86 SIMD intrinsics的基本使用
- 理解AVX2指令集和寄存器
- 实现第一个SIMD优化函数：float32点积
- 学习SimSIMD中dot.h模块的实现细节
- 性能测试与分析

---

## 第一部分：x86 SIMD Intrinsics基础

### 1.1 什么是Intrinsics？

**Intrinsics（内联函数）** 是编译器提供的特殊函数，直接映射到CPU指令。

```c
// 汇编代码（难以编写和维护）
__asm__ {
    movups xmm0, [rax]
    movups xmm1, [rbx]
    addps  xmm0, xmm1
    movups [rcx], xmm0
}

// Intrinsics（C风格，易读易写）
__m128 a = _mm_loadu_ps(ptr_a);  // movups xmm0, [rax]
__m128 b = _mm_loadu_ps(ptr_b);  // movups xmm1, [rbx]
__m128 c = _mm_add_ps(a, b);     // addps xmm0, xmm1
_mm_storeu_ps(ptr_c, c);         // movups [rcx], xmm0
```

### 1.2 Intrinsics命名规则

所有x86 intrinsics遵循统一的命名模式：

```
_mm<width>_<operation>_<type>
  │     │        │         └─ 数据类型
  │     │        └─────────── 操作名称
  │     └──────────────────── 向量宽度（可选）
  └────────────────────────── 前缀（表示SIMD系列）

示例：
_mm_add_ps       - SSE:  128-bit, add, packed single
_mm256_add_ps    - AVX:  256-bit, add, packed single
_mm512_add_ps    - AVX-512: 512-bit, add, packed single
```

**类型后缀含义**：
- `ps` = Packed Single (4× float32)
- `pd` = Packed Double (2× float64)
- `epi32` = Extended Packed Integer 32-bit (4× int32)
- `epi8` = Extended Packed Integer 8-bit (16× int8)

### 1.3 AVX2数据类型

```c
// SSE (128 bits)
__m128  vec_f32;  // 4× float32
__m128d vec_f64;  // 2× float64
__m128i vec_i32;  // 4× int32, 8× int16, 16× int8

// AVX/AVX2 (256 bits)
__m256  vec_f32;  // 8× float32
__m256d vec_f64;  // 4× float64
__m256i vec_i32;  // 8× int32, 16× int16, 32× int8

// AVX-512 (512 bits)
__m512  vec_f32;  // 16× float32
__m512d vec_f64;  // 8× float64
__m512i vec_i32;  // 16× int32, 32× int16, 64× int8
```

### 1.4 常用Intrinsics分类

#### 加载/存储
```c
// 对齐加载（地址必须16字节对齐）
__m256 _mm256_load_ps(float const* mem_addr);

// 非对齐加载（任意地址）
__m256 _mm256_loadu_ps(float const* mem_addr);

// 存储
void _mm256_store_ps(float* mem_addr, __m256 a);
void _mm256_storeu_ps(float* mem_addr, __m256 a);

// 广播（将一个值复制到所有元素）
__m256 _mm256_set1_ps(float a);  // {a, a, a, a, a, a, a, a}
```

#### 算术运算
```c
// 加法
__m256 _mm256_add_ps(__m256 a, __m256 b);

// 减法
__m256 _mm256_sub_ps(__m256 a, __m256 b);

// 乘法
__m256 _mm256_mul_ps(__m256 a, __m256 b);

// 除法
__m256 _mm256_div_ps(__m256 a, __m256 b);

// FMA (Fused Multiply-Add): a*b+c
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);
```

#### 水平归约（Horizontal Reduction）
```c
// 把向量中的所有元素加起来
float horizontal_sum_avx(__m256 vec) {
    // vec = {v0, v1, v2, v3, v4, v5, v6, v7}

    // 1. 把高128位和低128位相加
    __m128 low = _mm256_castps256_ps128(vec);      // {v0, v1, v2, v3}
    __m128 high = _mm256_extractf128_ps(vec, 1);   // {v4, v5, v6, v7}
    __m128 sum128 = _mm_add_ps(low, high);         // {v0+v4, v1+v5, v2+v6, v3+v7}

    // 2. 继续归约128位向量
    __m128 shuf = _mm_movehdup_ps(sum128);         // {v1+v5, v1+v5, v3+v7, v3+v7}
    __m128 sums = _mm_add_ps(sum128, shuf);        // {v0+v1+v4+v5, ...}
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}
```

---

## 第二部分：实现第一个SIMD函数 - Dot Product

### 2.1 标量版本（Serial实现）

```c
// 文件位置：include/simsimd/dot.h

SIMSIMD_PUBLIC void simsimd_dot_f32_serial(
    simsimd_f32_t const* a,  // 第一个向量
    simsimd_f32_t const* b,  // 第二个向量
    simsimd_size_t n,        // 向量长度
    simsimd_distance_t* result  // 输出：点积结果
) {
    simsimd_f32_t ab = 0;

    // 简单的循环累加
    for (simsimd_size_t i = 0; i != n; ++i) {
        ab += a[i] * b[i];
    }

    *result = ab;
}
```

**时间复杂度**：O(n)
**性能**：约 1-2 GFLOPS（浮点运算/秒）

### 2.2 查看SimSIMD中的Serial实现

```bash
# 搜索serial版本的dot实现
grep -A 15 "simsimd_dot_f32_serial" /home/dev/SimSIMD/include/simsimd/dot.h | head -20
```

### 2.3 AVX2版本（Haswell实现）

让我们查看SimSIMD中的Haswell实现：

```bash
# 查找Haswell后端的dot函数
grep -A 50 "simsimd_dot_f32_haswell" /home/dev/SimSIMD/include/simsimd/dot.h | head -60
```

**关键优化点**：

1. **向量化计算**：一次处理8个float32
2. **循环展开**：减少分支预测开销
3. **FMA指令**：融合乘加，减少舍入误差
4. **水平归约**：高效求和

典型实现结构：
```c
SIMSIMD_PUBLIC void simsimd_dot_f32_haswell(
    simsimd_f32_t const* a, simsimd_f32_t const* b,
    simsimd_size_t n, simsimd_distance_t* result) {

    // 初始化累加器（256位向量，可容纳8个float）
    __m256 ab_vec = _mm256_setzero_ps();

    simsimd_size_t i = 0;
    // 主循环：每次处理8个元素
    for (; i + 8 <= n; i += 8) {
        // 加载数据
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);

        // FMA: ab_vec = ab_vec + a_vec * b_vec
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    }

    // 处理尾部元素（剩余 < 8 个）
    simsimd_f32_t ab_tail = 0;
    for (; i < n; ++i) {
        ab_tail += a[i] * b[i];
    }

    // 水平归约：把8个累加值加起来
    simsimd_f32_t ab = _simsimd_reduce_f32x8_haswell(ab_vec);

    *result = ab + ab_tail;
}
```

### 2.4 FMA指令的优势

**传统方法（2条指令）**：
```c
tmp = _mm256_mul_ps(a, b);     // 乘法
sum = _mm256_add_ps(sum, tmp);  // 加法
```

**FMA（1条指令）**：
```c
sum = _mm256_fmadd_ps(a, b, sum);  // 一次完成：sum = a*b + sum
```

**优势**：
1. **性能提升**：减少50%的指令数
2. **精度提升**：只舍入一次（vs 传统方法舍入两次）
3. **延迟降低**：单指令流水线

### 2.5 水平归约实现

SimSIMD中的高效归约：

```c
// 将__m256向量的8个float求和
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_f32x8_haswell(__m256 vec) {
    // vec = {v0, v1, v2, v3, v4, v5, v6, v7}

    // 第一步：高128位 + 低128位
    // 结果：{v0+v4, v1+v5, v2+v6, v3+v7}
    __m128 low = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 sum = _mm_add_ps(low, high);

    // 第二步：交换并相加
    // hadd: 水平相加指令
    sum = _mm_hadd_ps(sum, sum);  // {v0+v1+v4+v5, v2+v3+v6+v7, ...}
    sum = _mm_hadd_ps(sum, sum);  // {v0+v1+v2+v3+v4+v5+v6+v7, ...}

    return _mm_cvtss_f32(sum);
}
```

---

## 第三部分：实战 - 编译并测试

### 3.1 创建测试程序

创建文件 `test_dot_product.c`：

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // AVX intrinsics

#define N 1536  // OpenAI embedding维度

// Serial实现
float dot_serial(float const* a, float const* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// AVX2实现
float dot_avx2(float const* a, float const* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // 归约
    __m128 low = _mm256_castps256_ps128(sum_vec);
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // 尾部
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// 性能测试
double benchmark(float (*func)(float const*, float const*, size_t),
                 float const* a, float const* b, size_t n,
                 int iterations) {
    clock_t start = clock();
    volatile float result;  // 防止编译器优化掉

    for (int i = 0; i < iterations; ++i) {
        result = func(a, b, n);
    }

    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    // 分配对齐内存
    float* a = (float*)aligned_alloc(32, N * sizeof(float));
    float* b = (float*)aligned_alloc(32, N * sizeof(float));

    // 初始化随机数据
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // 验证正确性
    float result_serial = dot_serial(a, b, N);
    float result_avx2 = dot_avx2(a, b, N);
    printf("结果验证：\n");
    printf("  Serial: %.6f\n", result_serial);
    printf("  AVX2:   %.6f\n", result_avx2);
    printf("  误差:   %.6e\n\n", result_serial - result_avx2);

    // 性能测试
    int iterations = 1000000;
    printf("性能测试 (%d次迭代)：\n", iterations);

    double time_serial = benchmark(dot_serial, a, b, N, iterations);
    printf("  Serial: %.3f秒 (%.2f GFLOPS)\n",
           time_serial, (2.0 * N * iterations / time_serial) / 1e9);

    double time_avx2 = benchmark(dot_avx2, a, b, N, iterations);
    printf("  AVX2:   %.3f秒 (%.2f GFLOPS)\n",
           time_avx2, (2.0 * N * iterations / time_avx2) / 1e9);

    printf("  加速比: %.2fx\n", time_serial / time_avx2);

    free(a);
    free(b);
    return 0;
}
```

### 3.2 编译并运行

```bash
# 编译Serial版本
gcc -O3 -o test_dot_serial test_dot_product.c -lm
./test_dot_serial

# 编译AVX2版本（需要Haswell或更新的CPU）
gcc -O3 -march=haswell -mavx2 -mfma -o test_dot_avx2 test_dot_product.c -lm
./test_dot_avx2

# 查看生成的汇编代码
objdump -d test_dot_avx2 | grep -A 20 "dot_avx2"
```

**预期结果**：
- Serial版本：~2-5 GFLOPS
- AVX2版本：~15-30 GFLOPS
- 加速比：~4-8x

---

## 第四部分：深入SimSIMD的dot.h实现

### 4.1 模块结构分析

```bash
# 查看dot.h的整体结构
head -200 /home/dev/SimSIMD/include/simsimd/dot.h
```

**关键部分**：
1. **宏生成器**：`SIMSIMD_MAKE_DOT`
2. **Serial后端**：所有数据类型的串行实现
3. **各平台后端**：NEON, SVE, Haswell, Skylake等

### 4.2 宏生成模式

SimSIMD使用宏来减少代码重复：

```c
// 宏定义（伪代码）
#define SIMSIMD_MAKE_DOT(name, input_type, accumulator_type, converter)  \
    SIMSIMD_PUBLIC void simsimd_dot_##input_type##_##name(               \
        simsimd_##input_type##_t const* a,                                \
        simsimd_##input_type##_t const* b,                                \
        simsimd_size_t n,                                                 \
        simsimd_distance_t* result) {                                     \
                                                                          \
        simsimd_##accumulator_type##_t ab = 0;                           \
        for (simsimd_size_t i = 0; i != n; ++i) {                        \
            simsimd_##accumulator_type##_t ai = converter(a + i);        \
            simsimd_##accumulator_type##_t bi = converter(b + i);        \
            ab += ai * bi;                                                \
        }                                                                 \
        *result = ab;                                                     \
    }

// 使用示例
SIMSIMD_MAKE_DOT(serial, f32, f32, SIMSIMD_DEREFERENCE)
SIMSIMD_MAKE_DOT(serial, f16, f32, SIMSIMD_F16_TO_F32)
SIMSIMD_MAKE_DOT(serial, bf16, f32, SIMSIMD_BF16_TO_F32)

// 生成的函数：
// - simsimd_dot_f32_serial
// - simsimd_dot_f16_serial
// - simsimd_dot_bf16_serial
```

### 4.3 混合精度计算

SimSIMD的一个重要特性是**混合精度**：
- 输入：低精度（f16, bf16, i8）
- 累加器：高精度（f32, i32）
- 输出：双精度（f64）

```c
// F16输入，F32累加
SIMSIMD_PUBLIC void simsimd_dot_f16_haswell(
    simsimd_f16_t const* a,
    simsimd_f16_t const* b,
    simsimd_size_t n,
    simsimd_distance_t* result) {

    __m256 ab_vec = _mm256_setzero_ps();  // F32累加器

    for (simsimd_size_t i = 0; i + 8 <= n; i += 8) {
        // 加载F16
        __m128i a_f16 = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i b_f16 = _mm_loadu_si128((__m128i const*)(b + i));

        // 转换为F32（使用F16C指令）
        __m256 a_f32 = _mm256_cvtph_ps(a_f16);
        __m256 b_f32 = _mm256_cvtph_ps(b_f16);

        // FMA累加（F32精度）
        ab_vec = _mm256_fmadd_ps(a_f32, b_f32, ab_vec);
    }

    // ... 归约和尾部处理
}
```

**为什么混合精度很重要？**
1. **内存带宽**：F16只需F32一半的带宽
2. **精度保证**：累加器使用高精度，避免溢出
3. **AI/ML应用**：模型权重常用F16/BF16存储

### 4.4 准确性变体

SimSIMD提供两个版本：

```c
// 标准版本：使用与输入相同或稍高的精度累加
simsimd_dot_f32_haswell()  // F32累加

// 准确版本：使用F64累加器
simsimd_dot_f32_accurate_haswell()  // F64累加

// 对于大规模计算，准确版本可避免累积误差
```

---

## 第五部分：性能分析与优化技巧

### 5.1 性能瓶颈识别

**理论峰值计算**：
```
Intel Skylake @ 3.0 GHz
- AVX2: 8× FMA/cycle
- 理论峰值 = 3.0 GHz × 8 × 2 (FMA算2次) = 48 GFLOPS (单核)

实际性能：
- Serial: ~2-5 GFLOPS (5-10%峰值)
- AVX2:   ~20-30 GFLOPS (40-60%峰值)
```

**为什么达不到100%？**
1. **内存带宽限制**：从RAM加载数据的速度
2. **指令延迟**：流水线停顿
3. **缓存未命中**：L1/L2/L3缓存访问延迟
4. **分支预测失败**：条件跳转开销

### 5.2 内存对齐优化

```c
// 坏例子：未对齐
float* a = (float*)malloc(n * sizeof(float));

// 好例子：32字节对齐（AVX2要求）
float* a = (float*)aligned_alloc(32, n * sizeof(float));

// 使用对齐加载（更快）
__m256 vec = _mm256_load_ps(a);  // 要求32字节对齐

// vs 非对齐加载（较慢，但更灵活）
__m256 vec = _mm256_loadu_ps(a);  // 任意地址
```

### 5.3 循环展开

```c
// 基础版本
for (size_t i = 0; i + 8 <= n; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(a + i);
    __m256 b_vec = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
}

// 优化：展开2次（减少循环开销）
for (size_t i = 0; i + 16 <= n; i += 16) {
    __m256 a0 = _mm256_loadu_ps(a + i);
    __m256 b0 = _mm256_loadu_ps(b + i);
    sum0 = _mm256_fmadd_ps(a0, b0, sum0);

    __m256 a1 = _mm256_loadu_ps(a + i + 8);
    __m256 b1 = _mm256_loadu_ps(b + i + 8);
    sum1 = _mm256_fmadd_ps(a1, b1, sum1);
}
// 最后合并sum0和sum1
```

### 5.4 编译器优化标志

```bash
# 基础优化
gcc -O2 program.c

# 激进优化
gcc -O3 -march=native program.c

# 特定目标
gcc -O3 -march=haswell -mavx2 -mfma -mf16c program.c

# 查看优化报告
gcc -O3 -march=haswell -fopt-info-vec-optimized program.c

# 生成汇编查看
gcc -S -O3 -march=haswell program.c
cat program.s
```

---

## 第六部分：今日总结与作业

### 今日学习要点

1. ✅ **Intrinsics基础**：命名规则、数据类型、常用函数
2. ✅ **FMA指令**：融合乘加的性能优势
3. ✅ **水平归约**：高效求和技巧
4. ✅ **混合精度**：F16输入、F32累加
5. ✅ **性能优化**：对齐、循环展开

### 作业

#### 作业1：实现并测试
运行上面的test_dot_product.c程序，记录你机器上的性能数据。

#### 作业2：扩展实现
修改test_dot_product.c，添加：
1. int8点积（提示：使用`__m256i`和`_mm256_madd_epi16`）
2. 对比Serial vs AVX2的精度差异

#### 作业3：阅读源码
深入阅读SimSIMD的dot.h：
```bash
# 查看完整的Haswell实现
grep -A 100 "simsimd_dot_f32_haswell" /home/dev/SimSIMD/include/simsimd/dot.h

# 对比不同后端
grep "simsimd_dot_f32_" /home/dev/SimSIMD/include/simsimd/dot.h | head -20
```

#### 作业4：性能分析
使用perf工具分析性能：
```bash
# Linux性能分析
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses ./test_dot_avx2

# 计算IPC（每周期指令数）
# IPC = instructions / cycles
# 理想值：2-4
```

### 预习第3天内容

明天我们将学习：
- **AVX-512编程**（Skylake, Ice Lake）
- **掩码操作**：处理尾部元素
- **复数点积**：vdot实现
- **空间距离**：Cosine和L2距离

---

## 参考资源

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [FMA指令详解](https://en.wikipedia.org/wiki/FMA_instruction_set)

---

**下一课**：[第3天：AVX-512与高级向量操作](./第3天_AVX512与高级向量操作.md)
