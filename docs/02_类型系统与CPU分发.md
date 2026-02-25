# SimSIMD 类型系统与 CPU 能力检测

> 本文是 SimSIMD 讲解系列第二篇，深入解析 `types.h` 和 `simsimd.h` 中的类型定义、平台检测宏、以及运行时 CPU 能力探测机制。

---

## 一、为什么需要类型系统？

在 C 语言中，`int` 的大小在不同平台上可能是 16 位、32 位或 64 位。SimSIMD 需要处理 8 位整数、16 位浮点等特定位宽的数值，**必须确保类型大小准确无误**，否则 SIMD 指令会读错数据。

这就是 `types.h` 存在的原因：**给所有数值类型起一个明确的别名**。

---

## 二、类型别名定义

```c
// include/simsimd/types.h

// 浮点类型
typedef double  simsimd_f64_t;   // 64位双精度
typedef float   simsimd_f32_t;   // 32位单精度
typedef unsigned short simsimd_f16_t;  // 16位半精度（存储为无符号短整型）
typedef unsigned short simsimd_bf16_t; // 脑浮点（同样16位）

// 整数类型
typedef signed char    simsimd_i8_t;   // 有符号8位
typedef unsigned char  simsimd_u8_t;   // 无符号8位
typedef signed short   simsimd_i16_t;  // 有符号16位
typedef unsigned short simsimd_u16_t;  // 无符号16位
typedef unsigned int   simsimd_u32_t;  // 无符号32位
typedef unsigned long long simsimd_u64_t; // 无符号64位

// 特殊类型
typedef unsigned long long simsimd_size_t;    // 数组长度
typedef double             simsimd_distance_t; // 结果距离值（统一用 f64）
```

### 关键细节：f16 和 bf16 为什么用 `unsigned short` 存储？

因为 C 语言标准（在大多数编译器）并不内置 16 位浮点类型。SimSIMD 用 `unsigned short`（16位无符号整数）来"承载"16位浮点的二进制内容，需要时再通过位操作转换成 `float`。

**打比方**：就像用一个16格的收纳盒存放某种零件，盒子本身不知道里面是什么，但你知道怎么把它取出来使用。

---

## 三、平台检测宏

`types.h` 的第一项任务是检测**运行在什么操作系统**和**什么 CPU 架构**上：

```c
// types.h 第 15-22 行：操作系统检测
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define _SIMSIMD_DEFINED_WINDOWS 1      // Windows
#elif defined(__APPLE__) && defined(__MACH__)
#define _SIMSIMD_DEFINED_APPLE 1        // macOS / iOS
#elif defined(__linux__)
#define _SIMSIMD_DEFINED_LINUX 1        // Linux
#endif

// CPU 架构检测
#if defined(__aarch64__) || defined(_M_ARM64)
#define _SIMSIMD_TARGET_ARM 1           // ARM 64位
#else
#define _SIMSIMD_TARGET_ARM 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define _SIMSIMD_TARGET_X86 1           // x86 64位
#else
#define _SIMSIMD_TARGET_X86 0
#endif
```

这些宏使用了编译器**预定义的宏**（不同编译器有不同约定）：

| 宏名称 | 含义 | 定义来源 |
|--------|------|----------|
| `__WIN32__` | Windows 平台 | MSVC / GCC on Windows |
| `__APPLE__` | Apple 平台 | Clang on macOS |
| `__linux__` | Linux 平台 | GCC / Clang on Linux |
| `__aarch64__` | ARM 64位 | GCC / Clang on ARM |
| `__x86_64__` | x86 64位 | GCC / Clang on x86 |

---

## 四、SIMD 能力宏：编译时的"开关"

这是 `types.h` 最复杂的部分。对于每种 SIMD 指令集，库都有一个宏来控制是否编译对应的代码：

```c
// 以 NEON（ARM 最基础的 SIMD）为例
#if !defined(SIMSIMD_TARGET_NEON) || (SIMSIMD_TARGET_NEON && !_SIMSIMD_TARGET_ARM)
#if defined(__ARM_NEON)               // 编译器自动设置的宏，表示支持 NEON
    #define SIMSIMD_TARGET_NEON _SIMSIMD_TARGET_ARM  // 如果是 ARM，就启用
#else
    #undef SIMSIMD_TARGET_NEON
    #define SIMSIMD_TARGET_NEON 0     // 否则禁用
#endif
#endif
```

### 全部 SIMD 目标宏列表

**ARM 平台：**
```
SIMSIMD_TARGET_NEON       → ARM NEON 基础（128位）
SIMSIMD_TARGET_NEON_F16   → NEON + 半精度浮点
SIMSIMD_TARGET_NEON_BF16  → NEON + 脑浮点
SIMSIMD_TARGET_NEON_I8    → NEON + INT8 点积
SIMSIMD_TARGET_SVE        → ARM SVE（可变宽度）
SIMSIMD_TARGET_SVE_F16    → SVE + 半精度
SIMSIMD_TARGET_SVE_BF16   → SVE + 脑浮点
SIMSIMD_TARGET_SVE_I8     → SVE + INT8
SIMSIMD_TARGET_SVE2       → SVE2（增强版）
```

**x86 平台：**
```
SIMSIMD_TARGET_HASWELL    → AVX2 + FMA + F16C
SIMSIMD_TARGET_SKYLAKE    → AVX512 基础集
SIMSIMD_TARGET_ICE        → AVX512 + VNNI
SIMSIMD_TARGET_GENOA      → AVX512 + BF16
SIMSIMD_TARGET_SAPPHIRE   → AVX512 + FP16
SIMSIMD_TARGET_TURIN      → AVX512 + VP2INTERSECT
SIMSIMD_TARGET_SIERRA     → AVX2 + VNNI
```

### 用户可以手动覆盖这些宏

```c
// 在包含头文件之前，强制禁用 AVX512
#define SIMSIMD_TARGET_SKYLAKE 0
#define SIMSIMD_TARGET_ICE 0
#include "simsimd.h"
```

这给用户提供了精细控制，比如在调试时强制使用串行代码。

---

## 五、函数符号可见性宏

```c
// types.h 第 32-44 行

// Windows DLL 导出符号
#if defined(_WIN32) || defined(__CYGWIN__)
#define SIMSIMD_DYNAMIC __declspec(dllexport)
#define SIMSIMD_PUBLIC  inline static
#define SIMSIMD_INTERNAL inline static

// GCC / Clang：使用 GCC 属性
#elif defined(__GNUC__) || defined(__clang__)
#define SIMSIMD_DYNAMIC  __attribute__((visibility("default"))) __attribute__((nonnull))
#define SIMSIMD_PUBLIC   __attribute__((unused, nonnull)) inline static
#define SIMSIMD_INTERNAL __attribute__((always_inline)) inline static
```

三个修饰符的含义：

| 修饰符 | 用途 | 解释 |
|--------|------|------|
| `SIMSIMD_DYNAMIC` | 运行时分发函数 | 导出为共享库公开符号 |
| `SIMSIMD_PUBLIC` | 公开 API 函数 | 内联，不可为 NULL 参数 |
| `SIMSIMD_INTERNAL` | 内部辅助函数 | **强制内联**，不暴露给用户 |

`__attribute__((nonnull))` 告诉编译器：指针参数绝对不会是 NULL。这允许编译器省略 NULL 检查，提升性能。

`__attribute__((always_inline))` 强制内联：即使编译器认为不值得内联，也必须内联。内部辅助函数（如"水平求和"）被调用无数次，强制内联可以消除函数调用开销。

---

## 六、能力枚举：CPU 能力的数字编码

`simsimd.h` 定义了一个非常重要的枚举，用位标志（bit flags）表示 CPU 能力：

```c
// simsimd.h 第 195-218 行
typedef enum {
    simsimd_cap_serial_k   = 1,        // 串行（无SIMD）
    simsimd_cap_any_k      = 0x7FFFFFFF, // 所有能力的掩码

    // x86
    simsimd_cap_haswell_k  = 1 << 10,  // 第10位 = 1024
    simsimd_cap_skylake_k  = 1 << 11,  // 第11位 = 2048
    simsimd_cap_ice_k      = 1 << 12,  // 第12位 = 4096
    simsimd_cap_genoa_k    = 1 << 13,
    simsimd_cap_sapphire_k = 1 << 14,
    simsimd_cap_turin_k    = 1 << 15,
    simsimd_cap_sierra_k   = 1 << 16,

    // ARM
    simsimd_cap_neon_k     = 1 << 20,  // 第20位
    simsimd_cap_neon_f16_k = 1 << 21,
    simsimd_cap_neon_bf16_k= 1 << 22,
    simsimd_cap_neon_i8_k  = 1 << 23,
    simsimd_cap_sve_k      = 1 << 24,
    simsimd_cap_sve_f16_k  = 1 << 25,
    simsimd_cap_sve_bf16_k = 1 << 26,
    simsimd_cap_sve_i8_k   = 1 << 27,
    simsimd_cap_sve2_k     = 1 << 28,
    simsimd_cap_sve2p1_k   = 1 << 29,
} simsimd_capability_t;
```

**为什么用位标志？**

使用位标志可以用一个整数表示多种能力的组合：

```c
// 如果 CPU 支持 AVX512 和 NEON（比如 Apple M4）
int caps = simsimd_cap_haswell_k | simsimd_cap_neon_k;
//       = 1024 | 1048576 = 1049600（二进制中第10位和第20位都是1）

// 检查是否支持 NEON
if (caps & simsimd_cap_neon_k) { /* 支持 NEON */ }

// 检查是否支持所有 AVX512 变体
if (caps & (simsimd_cap_skylake_k | simsimd_cap_ice_k)) { /* ... */ }
```

---

## 七、运行时 CPU 能力检测

这是整个库最"技术性"的部分。SimSIMD 如何在**程序运行时**判断 CPU 支持哪些 SIMD 指令？

### 7.1 x86 的方法：CPUID 指令

x86 CPU 提供了一个特殊指令 `CPUID`，程序可以用它查询 CPU 能力：

```c
// 伪代码
void check_x86_caps() {
    // 调用 CPUID，读取功能位
    // 检查 EBX 寄存器第5位 → 是否支持 AVX2
    // 检查 EBX 寄存器第16位 → 是否支持 AVX512F
    // 等等...
}
```

在实际代码中，GCC/Clang 提供了 `__builtin_cpu_supports()` 函数简化这个过程：
```c
if (__builtin_cpu_supports("avx2")) { /* 支持 AVX2 */ }
if (__builtin_cpu_supports("avx512f")) { /* 支持 AVX512 */ }
```

### 7.2 ARM Linux 的方法：信号陷阱

ARM 上的情况更复杂。检测 SVE 等高级指令集的方法是"大胆执行一次，看会不会崩溃"：

```c
// simsimd.h 中的 ARM Linux 检测思路
// 使用 POSIX 信号处理：
// 1. 注册 SIGILL（非法指令信号）处理器
// 2. 尝试执行 `mrs` 指令（读取系统寄存器）
// 3. 如果指令合法：支持 SVE
// 4. 如果触发 SIGILL：不支持，跳回安全点

#if defined(_SIMSIMD_DEFINED_LINUX) && defined(_POSIX_VERSION)
#include <setjmp.h>  // sigjmp_buf, sigsetjmp, siglongjmp
#include <signal.h>  // sigaction, SIGILL
#define _SIMSIMD_HAS_POSIX_EXTENSIONS 1
#endif
```

**打比方**：就像测试一扇门是否能通过——直接去推，如果推开了说明能进，如果被弹回来（SIGILL）说明通不过，但没有真的受伤（程序捕获了信号，安全恢复）。

### 7.3 macOS 的方法：sysctl API

Apple Silicon 不允许用户程序执行 `mrs` 指令（读取系统寄存器），所以用操作系统 API：

```c
#if defined(_SIMSIMD_DEFINED_APPLE)
#include <sys/sysctl.h>  // sysctlbyname

// 查询能力
int supported = 0;
size_t size = sizeof(supported);
sysctlbyname("hw.optional.arm.FEAT_SVE", &supported, &size, NULL, 0);
if (supported) { /* 支持 SVE */ }
#endif
```

### 7.4 Windows ARM 的方法：API 调用

```c
#if defined(_SIMSIMD_DEFINED_WINDOWS) && _SIMSIMD_TARGET_ARM
#include <processthreadsapi.h>

if (IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)) {
    /* 支持 SVE */
}
#endif
```

---

## 八、函数指针类型：统一的调用接口

为了实现运行时分发，`simsimd.h` 定义了几种函数指针类型：

```c
// 稠密向量度量（最常用）
typedef void (*simsimd_metric_dense_punned_t)(
    void const *a,           // 第一个向量
    void const *b,           // 第二个向量
    simsimd_size_t n,        // 向量长度
    simsimd_distance_t *d    // 输出：距离值
);

// 稀疏向量度量
typedef void (*simsimd_metric_sparse_punned_t)(
    void const *a, void const *b,
    simsimd_size_t a_length, simsimd_size_t b_length,
    simsimd_distance_t *d
);

// 曲率空间度量（需要度量张量）
typedef void (*simsimd_metric_curved_punned_t)(
    void const *a, void const *b, void const *c,  // c 是度量张量
    simsimd_size_t n,
    simsimd_distance_t *d
);
```

注意 `void const *a` 的用法——这是 C 语言的**类型擦除（Type Erasure）**技术。所有数据类型（f32、f16、i8 等）都通过同一个函数指针接口传递，运行时再强制转换为正确类型。

**打比方**：就像一个通用快递接口，包裹可以装任何东西，只有打开的人知道里面是什么。

---

## 九、数据类型枚举

```c
// simsimd.h 中的 simsimd_datatype_t 枚举
// 每种类型占一个位，可以组合使用

simsimd_datatype_b8_k    = 1 << 1   // 位向量
simsimd_datatype_i4x2_k  = 1 << 19  // 4位整数（两个打包成1字节）
simsimd_datatype_i8_k    = 1 << 2   // 有符号8位整数
simsimd_datatype_u8_k    = 1 << 6   // 无符号8位整数
simsimd_datatype_f16_k   = 1 << 12  // 半精度浮点
simsimd_datatype_bf16_k  = 1 << 13  // 脑浮点
simsimd_datatype_f32_k   = 1 << 11  // 单精度浮点
simsimd_datatype_f64_k   = 1 << 10  // 双精度浮点

// 复数版本（实部+虚部各占一个元素）
simsimd_datatype_f32c_k  = 1 << 21  // 复数单精度
simsimd_datatype_f64c_k  = 1 << 20  // 复数双精度
```

---

## 十、度量枚举的设计技巧

```c
// simsimd.h 第 145-190 行
typedef enum {
    simsimd_metric_dot_k   = 'i',  // 注意：用 ASCII 字符作为枚举值！
    simsimd_metric_inner_k = 'i',  // 两个名字，同一个值 → 别名
    simsimd_metric_cos_k   = 'c',
    simsimd_metric_cosine_k = 'c', // 别名
    simsimd_metric_l2_k    = '2',
    simsimd_metric_hamming_k = 'h',
    // ...
} simsimd_metric_kind_t;
```

用 ASCII 字符做枚举值有两个好处：
1. **可读性**：在调试器里看到 `'c'` 就知道是余弦
2. **序列化友好**：存入文件/网络时直接就是可读字母

---

## 十一、小结

| 概念 | 文件 | 核心机制 |
|------|------|----------|
| 类型别名 | `types.h` | `typedef` 确保位宽精确 |
| 平台检测 | `types.h` | 编译器预定义宏 |
| SIMD能力宏 | `types.h` | 条件编译开关 |
| 符号可见性 | `types.h` | GCC属性 / DLL导出 |
| 能力枚举 | `simsimd.h` | 位标志组合 |
| CPU运行时检测 | `simsimd.h` | CPUID/sysctl/信号陷阱 |
| 函数指针 | `simsimd.h` | void* 类型擦除 |
| 度量枚举 | `simsimd.h` | ASCII字符值 |

下一篇将深入讲解**空间距离（L2、余弦）和点积的具体算法实现**。
