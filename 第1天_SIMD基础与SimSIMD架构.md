# 第1天：SIMD基础与SimSIMD架构

## 课程目标
- 理解SIMD（Single Instruction Multiple Data）编程的核心概念
- 掌握SimSIMD的整体架构设计
- 学习头文件库的设计模式
- 了解跨平台SIMD编程的挑战

---

## 第一部分：SIMD编程基础（理论）

### 1.1 什么是SIMD？

**SIMD（单指令多数据流）** 是一种并行计算技术，允许单条指令同时处理多个数据元素。

#### 传统标量运算 vs SIMD向量运算

```c
// 标量运算（一次处理一个元素）
float a[4] = {1.0, 2.0, 3.0, 4.0};
float b[4] = {5.0, 6.0, 7.0, 8.0};
float c[4];

for (int i = 0; i < 4; i++) {
    c[i] = a[i] + b[i];  // 4次加法操作
}

// SIMD向量运算（一次处理4个元素）
__m128 a_vec = _mm_loadu_ps(a);  // 加载4个float
__m128 b_vec = _mm_loadu_ps(b);
__m128 c_vec = _mm_add_ps(a_vec, b_vec);  // 1条指令完成4次加法
_mm_storeu_ps(c, c_vec);
```

**性能提升**：理论上可达到 4x（SSE）、8x（AVX）、16x（AVX-512）

### 1.2 主流SIMD指令集对比

| 架构 | 指令集 | 向量宽度 | 浮点元素数 | 发布年份 |
|------|--------|----------|------------|----------|
| **x86** | SSE | 128 bits | 4× float32 | 1999 |
| | AVX | 256 bits | 8× float32 | 2011 |
| | AVX2 | 256 bits | + 整数支持 | 2013 |
| | AVX-512 | 512 bits | 16× float32 | 2015 |
| **ARM** | NEON | 128 bits | 4× float32 | 2005 |
| | SVE | 128-2048 bits | 可变长度 | 2016 |
| | SVE2 | 同SVE | + 更多指令 | 2019 |

### 1.3 SIMD编程的三大挑战

#### 挑战1：指令集碎片化
```
x86处理器：
  - Intel Haswell (2013)  → AVX2 + FMA + F16C
  - Intel Skylake (2015)  → AVX-512 基础
  - Intel Ice Lake (2019) → + VNNI, VBMI
  - AMD Genoa (2022)      → + BF16
  - Intel Sapphire (2023) → + FP16

每一代都增加新指令，导致数十种可能的组合！
```

#### 挑战2：数据类型支持不均
```c
// 不同数据类型在不同平台的支持情况
float32   - 所有平台都支持 ✓
float64   - 大部分支持 ✓
float16   - AVX-512FP16 (2023+), ARM FP16 ✓
bfloat16  - AVX-512BF16 (2020+), ARM BF16 (2021+) ✓
int8      - AVX-512VNNI (2019+), ARM i8mm (2021+) ✓
```

#### 挑战3：编写和维护成本
- 每个算法需要为每个平台、每种数据类型编写专门的实现
- 1个算法 × 5个平台 × 6种数据类型 = 30个函数！

---

## 第二部分：SimSIMD架构设计

### 2.1 整体架构概览

```
SimSIMD架构分层：
┌─────────────────────────────────────┐
│     Python/Rust/JS 语言绑定层      │
├─────────────────────────────────────┤
│    公共API层 (simsimd.h)            │
│  - 度量枚举 (dot, cos, l2...)       │
│  - 动态分发机制                      │
│  - CPU功能检测                       │
├─────────────────────────────────────┤
│      算法模块层 (独立头文件)         │
│  ├─ dot.h         (内积)            │
│  ├─ spatial.h     (空间距离)        │
│  ├─ binary.h      (二进制)          │
│  ├─ sparse.h      (稀疏向量)        │
│  ├─ probability.h (概率散度)        │
│  └─ curved.h      (曲线空间)        │
├─────────────────────────────────────┤
│   后端实现层 (每个模块多后端)       │
│  Serial, NEON, SVE, Haswell,        │
│  Skylake, Ice Lake, Genoa, Sapphire │
├─────────────────────────────────────┤
│    基础设施层 (types.h)             │
│  - 类型定义                         │
│  - 平台检测                         │
│  - 编译宏                           │
└─────────────────────────────────────┘
```

### 2.2 核心文件结构

查看实际文件大小分布：
```bash
cd /home/dev/SimSIMD/include/simsimd
ls -lh *.h
```

**文件说明**：
- `types.h` (25KB) - 基础类型、平台检测、编译配置
- `simsimd.h` (123KB) - 主入口、动态分发、CPU检测
- `dot.h` (102KB) - 最复杂的模块，支持最多后端
- `spatial.h` (126KB) - 空间距离，最常用的算法
- `elementwise.h` (113KB) - FMA和加权和操作
- `curved.h` (85KB) - 矩阵-向量操作
- `sparse.h` (72KB) - 稀疏向量优化算法
- `probability.h` (30KB) - 概率分布相关
- `binary.h` (25KB) - 二进制向量操作

### 2.3 设计哲学与原则

SimSIMD遵循以下设计原则（出自README）：

```c
// 1. 避免循环展开 - 让编译器决定
for (size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
}
// 不要手动写成：
// for (size_t i = 0; i < n; i += 4) {
//     sum0 += a[i+0] * b[i+0];
//     sum1 += a[i+1] * b[i+1];
//     ...
// }

// 2. 永不分配内存
void simsimd_dot_f32(float const* a, float const* b,
                     size_t n, float* result) {
    // 只使用栈内存和寄存器
    // 无 malloc/new
}

// 3. 永不抛异常或设置errno
// - 使用返回值或输出参数报告错误
// - 保持函数为 noexcept

// 4. 所有函数参数都是指针大小
// - 便于寄存器传参，减少栈操作

// 5. 使用输出参数而非返回值
void simsimd_dot_f32(..., float* result);  // ✓ 推荐
float simsimd_dot_f32(...);                // ✗ 不推荐

// 6. 优先优化新指令和混合精度
// - 不过度优化float32/float64（编译器已很好）
// - 重点放在 float16, bfloat16, int8 等
```

---

## 第三部分：types.h - 基础设施层

### 3.1 查看 types.h 的结构

```bash
head -100 /home/dev/SimSIMD/include/simsimd/types.h
```

### 3.2 平台检测机制

```c
// 检测目标架构
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define _SIMSIMD_TARGET_X86 1
#else
    #define _SIMSIMD_TARGET_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define _SIMSIMD_TARGET_ARM 1
#else
    #define _SIMSIMD_TARGET_ARM 0
#endif

// 检测编译器
#if defined(__GNUC__) && !defined(__clang__)
    #define _SIMSIMD_DEFINED_GCC
#elif defined(__clang__)
    #define _SIMSIMD_DEFINED_CLANG
#elif defined(_MSC_VER)
    #define _SIMSIMD_DEFINED_MSVC
#endif

// 检测操作系统
#if defined(__APPLE__)
    #define _SIMSIMD_DEFINED_APPLE
#elif defined(__linux__)
    #define _SIMSIMD_DEFINED_LINUX
#elif defined(_WIN32)
    #define _SIMSIMD_DEFINED_WINDOWS
#endif
```

### 3.3 基础类型定义

```c
// 标准整数类型（不依赖 stdint.h）
typedef signed char simsimd_i8_t;
typedef unsigned char simsimd_u8_t;
typedef unsigned short simsimd_u16_t;
typedef unsigned int simsimd_u32_t;
typedef unsigned long long simsimd_u64_t;

// 浮点类型
typedef float simsimd_f32_t;
typedef double simsimd_f64_t;

// 特殊类型（条件编译）
#if defined(_SIMSIMD_DEFINED_GCC) || defined(_SIMSIMD_DEFINED_CLANG)
    typedef _Float16 simsimd_f16_t;
    typedef __bf16 simsimd_bf16_t;
#else
    // MSVC等编译器使用自定义实现
    typedef struct { simsimd_u16_t value; } simsimd_f16_t;
    typedef struct { simsimd_u16_t value; } simsimd_bf16_t;
#endif

// 复数类型
typedef struct { simsimd_f32_t real, imag; } simsimd_f32c_t;
typedef struct { simsimd_f64_t real, imag; } simsimd_f64c_t;

// 距离结果类型
typedef double simsimd_distance_t;

// 大小类型
typedef unsigned long long simsimd_size_t;
```

### 3.4 编译时特性检测

```c
// 检测AVX2支持（Haswell）
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    #define SIMSIMD_TARGET_HASWELL 1
#else
    #define SIMSIMD_TARGET_HASWELL 0
#endif

// 检测AVX-512基础支持（Skylake）
#if defined(__AVX512F__) && defined(__AVX512CD__) && \
    defined(__AVX512VL__) && defined(__AVX512DQ__) && \
    defined(__AVX512BW__)
    #define SIMSIMD_TARGET_SKYLAKE 1
#else
    #define SIMSIMD_TARGET_SKYLAKE 0
#endif

// 检测NEON支持（ARM）
#if defined(__ARM_NEON)
    #define SIMSIMD_TARGET_NEON _SIMSIMD_TARGET_ARM
#else
    #define SIMSIMD_TARGET_NEON 0
#endif

// 检测SVE支持（ARM可变长度向量）
#if defined(__ARM_FEATURE_SVE)
    #define SIMSIMD_TARGET_SVE _SIMSIMD_TARGET_ARM
#else
    #define SIMSIMD_TARGET_SVE 0
#endif
```

---

## 第四部分：实战练习 - 理解类型转换

### 练习1：查看float16的实现

```bash
# 在types.h中搜索float16转换函数
grep -A 20 "simsimd_f16_to_f32" /home/dev/SimSIMD/include/simsimd/types.h
```

**任务**：理解IEEE 754半精度浮点格式
- 符号位：1 bit
- 指数：5 bits (偏移15)
- 尾数：10 bits

```c
// F16转F32的位级操作
SIMSIMD_INTERNAL simsimd_f32_t simsimd_f16_to_f32(simsimd_f16_t const* h) {
    simsimd_u16_t h_bits = *(simsimd_u16_t const*)h;

    // 提取组件
    simsimd_u32_t sign = (h_bits & 0x8000) >> 15;      // 符号位
    simsimd_u32_t exponent = (h_bits & 0x7C00) >> 10;  // 指数 (5 bits)
    simsimd_u32_t mantissa = (h_bits & 0x03FF);        // 尾数 (10 bits)

    // 转换到F32格式
    // F32: 1 sign + 8 exponent + 23 mantissa
    simsimd_u32_t f32_bits;

    if (exponent == 0) {
        // 次正规数或零
        if (mantissa == 0) {
            f32_bits = sign << 31;  // ±0
        } else {
            // 次正规数转正规数
            // ...
        }
    } else if (exponent == 0x1F) {
        // 无穷大或NaN
        f32_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        // 正规数
        f32_bits = (sign << 31) |
                   ((exponent - 15 + 127) << 23) |  // 重新偏移指数
                   (mantissa << 13);                 // 扩展尾数
    }

    return *(simsimd_f32_t*)&f32_bits;
}
```

### 练习2：理解bfloat16

**BF16格式**（Brain Float 16）：
- 符号：1 bit
- 指数：8 bits（与F32相同）
- 尾数：7 bits（截断F32的23 bits）

```c
// BF16转F32非常简单（零扩展）
SIMSIMD_INTERNAL simsimd_f32_t simsimd_bf16_to_f32(simsimd_bf16_t const* b) {
    simsimd_u16_t bf16_bits = *(simsimd_u16_t const*)b;
    simsimd_u32_t f32_bits = (simsimd_u32_t)bf16_bits << 16;  // 左移16位
    return *(simsimd_f32_t*)&f32_bits;
}

// F32转BF16（简单截断，可选舍入）
SIMSIMD_INTERNAL void simsimd_f32_to_bf16(simsimd_f32_t f, simsimd_bf16_t* b) {
    simsimd_u32_t f32_bits = *(simsimd_u32_t*)&f;
    // 可选：舍入
    // f32_bits += 0x8000;  // 四舍五入
    simsimd_u16_t bf16_bits = (simsimd_u16_t)(f32_bits >> 16);
    *(simsimd_u16_t*)b = bf16_bits;
}
```

### 练习3：编写一个简单的类型转换测试

```c
#include <stdio.h>
#include "include/simsimd/types.h"

int main() {
    // 测试F32
    simsimd_f32_t f32_val = 3.14159f;
    printf("F32: %.6f\n", f32_val);

    // 转换为BF16并回转
    simsimd_bf16_t bf16_val;
    simsimd_f32_to_bf16(f32_val, &bf16_val);
    simsimd_f32_t f32_from_bf16 = simsimd_bf16_to_f32(&bf16_val);
    printf("F32 -> BF16 -> F32: %.6f (error: %.6f)\n",
           f32_from_bf16, f32_val - f32_from_bf16);

    // 显示精度损失
    printf("精度损失: %.2e\n", fabs(f32_val - f32_from_bf16) / f32_val);

    return 0;
}
```

---

## 第五部分：今日总结与作业

### 今日学习要点

1. ✅ **SIMD概念**：单指令多数据，并行处理
2. ✅ **指令集演进**：从SSE到AVX-512，从NEON到SVE
3. ✅ **SimSIMD架构**：分层设计，模块化，动态分发
4. ✅ **类型系统**：支持多种精度，位级转换

### 作业

#### 作业1：探索代码库
```bash
# 统计每个模块的后端实现数量
cd /home/dev/SimSIMD/include/simsimd
for file in *.h; do
    echo "=== $file ==="
    grep -c "SIMSIMD_PUBLIC void simsimd_" $file
done
```

#### 作业2：理解编译宏
查看你的机器支持哪些SIMD特性：
```bash
# Linux x86
gcc -march=native -dM -E - < /dev/null | grep -E "AVX|SSE|FMA"

# Linux ARM
gcc -march=native -dM -E - < /dev/null | grep -E "NEON|SVE|FP16"
```

#### 作业3：绘制架构图
使用任意工具（纸笔/绘图软件）绘制：
1. SimSIMD的五层架构图
2. 标注每层的主要职责
3. 标注数据流向

#### 作业4：阅读源码
阅读并理解以下代码段：
```bash
# 查看simsimd.h的前200行
head -200 /home/dev/SimSIMD/include/simsimd/simsimd.h

# 特别关注：
# - SIMSIMD_VERSION宏定义
# - SIMSIMD_DYNAMIC_DISPATCH设置
# - 各个模块头文件的包含顺序
```

### 预习第2天内容

明天我们将深入学习：
- **x86 SIMD intrinsics（内联函数）**
- **AVX2编程基础**
- **实现第一个SIMD函数：dot product**
- **性能对比：Serial vs Haswell**

准备环境：
```bash
# 确保可以编译AVX2代码
gcc -march=haswell -mavx2 -mfma -mf16c --version
```

---

## 参考资源

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide)
- [ARM Intrinsics Search](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [IEEE 754浮点标准](https://en.wikipedia.org/wiki/IEEE_754)
- [SIMD Tutorial (x86)](https://www.intel.com/content/www/us/en/developer/articles/technical/a-guide-to-vectorization-with-intel-c-compilers.html)

---

**下一课**：[第2天：x86 SIMD编程入门 - AVX2与内积运算](./第2天_x86_SIMD编程入门.md)
