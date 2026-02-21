# 补充课程4：运行时CPU检测与动态分发机制深度剖析

## 课程目标
- 理解编译时分发与运行时分发的本质区别
- 逐行分析 `CPUID` 汇编指令及其寄存器结构
- 掌握 SimSIMD 能力位掩码（`simsimd_capability_t`）的设计哲学
- 理解 `simsimd_find_kernel_punned` 函数指针分发表
- 掌握非规格化数（Denormal）对 SIMD 性能的影响和 FTZ/DAZ 处理

**核心源码**：`include/simsimd/simsimd.h`（2485行）

---

## 第一部分：编译时分发 vs 运行时分发

### 1.1 问题背景

```
编译器选项 -march=native：
  - 生成专用于当前 CPU 的代码
  - 分发的二进制 → 在旧 CPU 上崩溃（Illegal Instruction）

编译器选项 -march=x86-64（通用）：
  - 只使用 SSE2（2001年的指令集）
  - 性能损失：5~50x

困境：
  ┌──────────────────────────────────┐
  │ 如何在分发时保持通用性，          │
  │ 在运行时又能利用最新硬件特性？    │
  └──────────────────────────────────┘
```

### 1.2 SimSIMD 的解决方案

```
simsimd.h 头文件
│
├── 编译多个后端（每个目标CPU生成一份）
│   ├── simsimd_dot_f32_serial   ← 总是可用（兜底）
│   ├── simsimd_dot_f32_haswell  ← 需要 AVX2+FMA
│   ├── simsimd_dot_f32_skylake  ← 需要 AVX-512
│   └── simsimd_dot_f32_sapphire ← 需要 AVX-512 FP16
│
├── 运行时检测（CPUID / ARM mrs）
│   └── 确定当前 CPU 支持哪些指令集
│
└── 分发表（simsimd_find_kernel_punned）
    └── 选择能力最强的后端 → 返回函数指针
```

### 1.3 两种分发模式

```c
// simsimd.h:101-103
#if !defined(SIMSIMD_DYNAMIC_DISPATCH)
#define SIMSIMD_DYNAMIC_DISPATCH (0)  // 默认：静态分发
#endif
```

```
模式1：静态分发（SIMSIMD_DYNAMIC_DISPATCH=0，默认）
  - 编译时：用户指定 -DSIMSIMD_TARGET_HASWELL=1 等宏
  - 头文件 include-guard 按宏选择后端
  - 生成的代码中只有一个实现
  - 优点：代码小，无运行时开销
  - 适用：明确知道目标 CPU 的场景

模式2：动态分发（SIMSIMD_DYNAMIC_DISPATCH=1）
  - 运行时检测 CPU 能力
  - 所有后端都编译进二进制
  - 首次调用时检测并缓存最佳后端
  - 优点：单一二进制支持所有 CPU
  - 适用：分发给未知 CPU 的 Python 库等
```

---

## 第二部分：CPUID 指令深度解析

### 2.1 什么是 CPUID

```
CPUID（CPU Identification）是 x86 专用指令：
  - 输入：EAX（功能号），ECX（子功能号）
  - 输出：EAX, EBX, ECX, EDX（四个32位寄存器）
  - 可以查询：CPU型号、支持的特性、缓存信息等

历史：1993年，Intel 486DX2 引入
特点：用户空间可用（无需内核权限）
```

### 2.2 SimSIMD 的 CPUID 调用

```c
// simsimd.h:365-444
SIMSIMD_PUBLIC simsimd_capability_t _simsimd_capabilities_x86(void) {

    // 结构体用于存储 CPUID 的4个输出寄存器
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7, info7sub1;

    // CPUID 调用 1：基本 CPU 信息
    // EAX=1 → 基本特性标志
#if defined(_MSC_VER)
    __cpuidex(info1.array, 1, 0);   // Windows 专用内置
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(info1.named.eax),    // 输出：EAX
          "=b"(info1.named.ebx),    // 输出：EBX
          "=c"(info1.named.ecx),    // 输出：ECX
          "=d"(info1.named.edx)     // 输出：EDX
        : "a"(1), "c"(0));          // 输入：EAX=1, ECX=0
#endif
```

**CPUID EAX=1 返回值布局**：
```
ECX 寄存器（功能 1）：
  Bit 0:  SSE3
  Bit 1:  PCLMULQDQ
  Bit 9:  SSSE3
  Bit 12: FMA ←── SimSIMD 使用！
  Bit 19: SSE4.1
  Bit 20: SSE4.2
  Bit 28: AVX
  Bit 29: F16C ←── SimSIMD 使用！

EBX 寄存器（功能 7）：
  Bit 5:  AVX2 ←── SimSIMD 使用！
  Bit 16: AVX-512F ←── SimSIMD 使用！
  Bit 21: AVX-512IFMA ←── SimSIMD 使用！

ECX 寄存器（功能 7）：
  Bit 6:  AVX-512VBMI2 ←── SimSIMD 使用！
  Bit 11: AVX-512VNNI ←── SimSIMD 使用！
  Bit 12: AVX-512BITALG ←── SimSIMD 使用！
  Bit 14: AVX-512VPOPCNTDQ ←── SimSIMD 使用！
```

### 2.3 三次 CPUID 调用的原因

```c
// 为什么需要3次 CPUID 调用？

// 第1次：EAX=1, ECX=0（基本信息）
__asm__ __volatile__("cpuid" : ... : "a"(1), "c"(0));
// 获取：FMA, F16C（F16C需要 leaf 1）

// 第2次：EAX=7, ECX=0（扩展特性）
__asm__ __volatile__("cpuid" : ... : "a"(7), "c"(0));
// 获取：AVX2, AVX-512F/VNNI/IFMA/BITALG/VBMI2/VPOPCNTDQ, FP16

// 第3次：EAX=7, ECX=1（扩展特性的子叶）
__asm__ __volatile__("cpuid" : ... : "a"(7), "c"(1));
// 获取：AVX-512BF16（在 sub-leaf 1 的 EAX bit 5）

// 设计原因：Intel 和 AMD 随着 CPU 迭代，用不同的 leaf 报告不同特性
// 必须逐一查询对应的 leaf+bit
```

### 2.4 位检测代码详解

```c
// simsimd.h:395-432
// Check for AVX2
unsigned supports_avx2 = (info7.named.ebx & 0x00000020) != 0;
//                                          ^^^^^^^^^^^
//                                          bit 5 of EBX（leaf 7）

// Check for F16C（Function ID 1, ECX bit 29）
unsigned supports_f16c = (info1.named.ecx & 0x20000000) != 0;
//                                          0x20000000 = 1 << 29

// Check for FMA（Function ID 1, ECX bit 12）
unsigned supports_fma = (info1.named.ecx & 0x00001000) != 0;
//                                         0x00001000 = 1 << 12

// Check for AVX-512VPOPCNTDQ（Function ID 7, ECX bit 14）
unsigned supports_avx512vpopcntdq = (info7.named.ecx & 0x00004000) != 0;
//                                                     0x00004000 = 1 << 14

// Check for AVX-512BF16（Function ID 7, Sub-leaf 1, EAX bit 5）
unsigned supports_avx512bf16 = (info7sub1.named.eax & 0x00000020) != 0;
//                              ^^^^^^^^ sub-leaf 1！不是 leaf 7 的直接结果

// Turin 需要 VP2INTERSECT，但只想在 AMD Zen5 上启用
// （Intel Tiger Lake 也支持 VP2INTERSECT 但 BF16 不支持）
unsigned supports_turin = supports_avx512vp2intersect && supports_avx512bf16;
//                                                   ^^^^^^^^^^^^^^^^^^^^^^^^
//                                                   组合条件：防止误匹配 Tiger Lake
```

### 2.5 CPU 代际映射

```c
// simsimd.h:425-433
// 将具体特性组合 → CPU 世代标签
unsigned supports_haswell  = supports_avx2 && supports_f16c && supports_fma;
// Haswell(2013)：AVX2 + F16C + FMA 三者同时具备

unsigned supports_skylake  = supports_avx512f;
// Skylake(2016)：AVX-512F 是基础

unsigned supports_ice      = supports_avx512vnni && supports_avx512ifma &&
                             supports_avx512bitalg && supports_avx512vbmi2 &&
                             supports_avx512vpopcntdq;
// Ice Lake(2019)：所有这些特性的集合

unsigned supports_genoa    = supports_avx512bf16;
// AMD Genoa(2023)：BF16

unsigned supports_sapphire = supports_avx512fp16;
// Intel Sapphire Rapids(2023)：FP16

// 能力标志组合成 bitmask
return (simsimd_capability_t)(
    (simsimd_cap_haswell_k * supports_haswell) |
    (simsimd_cap_skylake_k * supports_skylake) |
    ...
    (simsimd_cap_serial_k));
// 注意：serial 总是包含！确保有兜底实现
```

---

## 第三部分：ARM 能力检测

### 3.1 x86 vs ARM 的根本差异

```
x86：CPUID 是用户空间可用的指令
ARM：系统寄存器（如 ID_AA64ISAR0_EL1）需要 EL1 权限

解决方案：
  Linux：  /proc/cpuinfo 或 HWCAP（hardware capability）
  macOS：  sysctl API（Apple Silicon 专用）
  Windows：IsProcessorFeaturePresent
```

### 3.2 Linux ARM 检测

```c
// simsimd.h:对应 ARM Linux 部分
#if _SIMSIMD_TARGET_ARM
#if defined(_SIMSIMD_DEFINED_LINUX)

// Linux 提供 HWCAP：hardware capabilities 位掩码
// 通过 getauxval(AT_HWCAP) 获取
#include <sys/auxv.h>    // getauxval
#include <asm/hwcap.h>   // HWCAP_NEON, HWCAP2_SVE 等

SIMSIMD_PUBLIC simsimd_capability_t _simsimd_capabilities_arm(void) {
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // HWCAP_NEON (bit 12)：基础 NEON
    unsigned supports_neon = (hwcap & HWCAP_ASIMD) != 0;
    //                              ^^^^^^^^^^^ AArch64: HWCAP_ASIMD

    // HWCAP_FPHP + HWCAP_ASIMDHP：半精度 FP16 支持
    unsigned supports_neon_f16 = (hwcap & HWCAP_FPHP) && (hwcap & HWCAP_ASIMDHP);

    // HWCAP2_SVE：可扩展向量扩展
    unsigned supports_sve = (hwcap2 & HWCAP2_SVE) != 0;
```

### 3.3 Apple Silicon 的特殊路径

```c
// simsimd.h:对应 macOS/Apple 部分
// Apple 的 sysctl API 提供 CPU 特性查询

#if defined(_SIMSIMD_DEFINED_APPLE)
#include <sys/sysctl.h>  // sysctlbyname

// 查询是否支持 NEON
int supports_neon = 0;
size_t size = sizeof(supports_neon);
sysctlbyname("hw.optional.AdvSIMD", &supports_neon, &size, NULL, 0);
//           ^^^^^^^^^^^^^^^^^^^^ Apple 私有 sysctl 键

// 查询是否支持 SVE（Apple M 系列目前不支持 SVE，只有 NEON）
// M1: Armv8.5-A（没有SVE）
// M2/M3: Armv8.6-A（没有SVE）
// M4: 预计 Armv9.1-A（可能有SVE2）

// 为什么不用 mrs 指令直接读系统寄存器？
// Apple Silicon 在 EL0（用户态）不允许读取系统寄存器
// 执行 mrs x0, ID_AA64ISAR0_EL1 会触发 EXC_BAD_INSTRUCTION
// 必须通过 sysctl 间接获取
```

---

## 第四部分：能力位掩码设计

### 4.1 simsimd_capability_t 枚举

```c
// simsimd.h:177-199
typedef enum {
    simsimd_cap_serial_k = 1,          // bit 0：串行（总是支持）
    simsimd_cap_any_k = 0x7FFFFFFF,    // 所有位为1（INT_MAX）

    // x86 能力（bit 10-16）
    simsimd_cap_haswell_k = 1 << 10,   // AVX2 + FMA + F16C
    simsimd_cap_skylake_k = 1 << 11,   // AVX-512 基础
    simsimd_cap_ice_k     = 1 << 12,   // AVX-512 + VPOPCNTDQ 等
    simsimd_cap_genoa_k   = 1 << 13,   // AVX-512 + BF16
    simsimd_cap_sapphire_k= 1 << 14,   // AVX-512 + FP16
    simsimd_cap_turin_k   = 1 << 15,   // AVX-512 + VP2INTERSECT
    simsimd_cap_sierra_k  = 1 << 16,   // AVX2 + VNNI（i8 加速）

    // ARM 能力（bit 20-29）
    simsimd_cap_neon_k     = 1 << 20,  // ARM NEON 基础
    simsimd_cap_neon_f16_k = 1 << 21,  // NEON + FP16
    simsimd_cap_neon_bf16_k= 1 << 22,  // NEON + BF16
    simsimd_cap_neon_i8_k  = 1 << 23,  // NEON + INT8 DOTPROD
    simsimd_cap_sve_k      = 1 << 24,  // ARM SVE
    simsimd_cap_sve_f16_k  = 1 << 25,  // SVE + FP16
    simsimd_cap_sve_bf16_k = 1 << 26,  // SVE + BF16
    simsimd_cap_sve_i8_k   = 1 << 27,  // SVE + INT8
    simsimd_cap_sve2_k     = 1 << 28,  // ARM SVE2
    simsimd_cap_sve2p1_k   = 1 << 29,  // ARM SVE2.1
} simsimd_capability_t;
```

**位掩码设计的妙处**：

```
检查单个能力：
  capabilities & simsimd_cap_haswell_k    // 是否支持 Haswell？

检查 x86 vs ARM：
  capabilities >= simsimd_cap_haswell_k && capabilities <= simsimd_cap_sierra_k
  // x86 能力在 bit 10-16

允许多个能力组合：
  allowed_caps = simsimd_cap_serial_k | simsimd_cap_haswell_k
  // 只允许这两个实现，屏蔽更高级的

检查任意：
  allowed_caps = simsimd_cap_any_k  // 不限制，选最好的

典型实际值（Ice Lake CPU）：
  0b000_0000_0000_0_0001100_0_0000001
  = simsimd_cap_serial_k | simsimd_cap_haswell_k | simsimd_cap_skylake_k | simsimd_cap_ice_k
  即：同时包含所有较低代际！（超集关系）

注意：Genoa 和 Sapphire 不自动包含对方
  因为 BF16 和 FP16 是两条不同的演进路线
```

---

## 第五部分：函数指针分发表详解

### 5.1 类型定义

```c
// simsimd.h:248-314

// 密集向量函数指针类型
typedef void (*simsimd_metric_dense_punned_t)(
    void const *a,
    void const *b,
    simsimd_size_t n,
    simsimd_distance_t *d);

// 稀疏向量函数指针类型
typedef void (*simsimd_metric_sparse_punned_t)(
    void const *a, void const *b,
    simsimd_size_t a_length, simsimd_size_t b_length,
    simsimd_distance_t *d);

// 曲率空间函数指针类型（多一个矩阵参数）
typedef void (*simsimd_metric_curved_punned_t)(
    void const *a, void const *b, void const *c,
    simsimd_size_t n, simsimd_distance_t *d);

// 统一类型（类型擦除后的万能指针）
typedef void (*simsimd_kernel_punned_t)(void *);
//                                      ^^^^^^
//                                      "punned" = type-punned（类型双关）
//                                      C 允许通过 void* 转换绕过类型系统
```

**类型双关（Type Punning）的意义**：
```c
// 问题：不同签名的函数无法存在同一个数组里
// 解决：全部转为 void (*)(void*) 存储，调用时再转回

void (*kernel)(void *) = (void (*)(void *))(&simsimd_dot_f32_haswell);
// 存储时：丢弃类型信息（类型双关）

// 使用时：转回正确类型
simsimd_metric_dense_punned_t typed = (simsimd_metric_dense_punned_t)kernel;
typed(a, b, n, &result);
// 只要签名真的匹配，这是合法的（ABI兼容性保证）
```

### 5.2 `simsimd_find_kernel_punned` 分发逻辑

```c
// simsimd.h 中的分发函数（以 f32 为例）
SIMSIMD_INTERNAL void _simsimd_find_kernel_punned_f32(
    simsimd_capability_t v,    // 当前CPU支持的能力
    simsimd_metric_kind_t k,   // 要查找的度量类型
    simsimd_kernel_punned_t *m,// 输出：找到的函数指针
    simsimd_capability_t *c)   // 输出：实际使用的能力级别
{
    typedef simsimd_kernel_punned_t m_t;

    // 优先级从高到低（能力越强越优先）：

    // 1. SVE（ARM 最强）
#if SIMSIMD_TARGET_SVE
    if (v & simsimd_cap_sve_k) switch (k) {
        case simsimd_metric_dot_k:   *m = (m_t)&simsimd_dot_f32_sve, *c = simsimd_cap_sve_k; return;
        case simsimd_metric_cos_k:   *m = (m_t)&simsimd_cos_f32_sve, *c = simsimd_cap_sve_k; return;
        case simsimd_metric_l2sq_k:  *m = (m_t)&simsimd_l2sq_f32_sve, *c = simsimd_cap_sve_k; return;
        default: break;
    }
#endif

    // 2. NEON（ARM 通用）
#if SIMSIMD_TARGET_NEON
    if (v & simsimd_cap_neon_k) switch (k) {
        case simsimd_metric_dot_k:   *m = (m_t)&simsimd_dot_f32_neon, *c = simsimd_cap_neon_k; return;
        ...
    }
#endif

    // 3. Skylake（x86 AVX-512 基础）
#if SIMSIMD_TARGET_SKYLAKE
    if (v & simsimd_cap_skylake_k) switch (k) {
        case simsimd_metric_dot_k:   *m = (m_t)&simsimd_dot_f32_skylake, *c = simsimd_cap_skylake_k; return;
        case simsimd_metric_fma_k:   *m = (m_t)&simsimd_fma_f32_skylake, *c = simsimd_cap_skylake_k; return;
        ...
    }
#endif

    // 4. Haswell（x86 AVX2）
#if SIMSIMD_TARGET_HASWELL
    if (v & simsimd_cap_haswell_k) switch (k) {
        case simsimd_metric_dot_k:   *m = (m_t)&simsimd_dot_f32_haswell, *c = simsimd_cap_haswell_k; return;
        ...
    }
#endif

    // 5. 串行（兜底，总是可用）
    if (v & simsimd_cap_serial_k) switch (k) {
        case simsimd_metric_dot_k:   *m = (m_t)&simsimd_dot_f32_serial, *c = simsimd_cap_serial_k; return;
        ...
    }
    // 注意：如果 k 未知，m 和 c 保持默认值（通常是 NULL 和 0）
}
```

**分发优先级**：
```
ARM：  SVE → NEON
x86：  Sapphire > Genoa > Ice > Skylake > Haswell > Serial

并非所有度量对所有级别都有实现：
  例如：Haversine 没有 SIMD 版本，只有 serial
  分发函数找不到时：m 保持 NULL（调用者需要处理）
```

### 5.3 顶层分发函数

```c
// simsimd.h: simsimd_find_kernel_punned（公共接口）
SIMSIMD_PUBLIC void simsimd_find_kernel_punned(
    simsimd_metric_kind_t kind,          // 要什么度量？
    simsimd_datatype_t datatype,         // 什么数据类型？
    simsimd_capability_t supported,      // CPU 支持什么？
    simsimd_capability_t allowed,        // 用户允许什么？
    simsimd_kernel_punned_t *kernel_output,    // 输出函数指针
    simsimd_capability_t *capability_output)   // 输出实际使用的能力
{
    // 取交集：CPU支持 ∩ 用户允许
    simsimd_capability_t effective = (simsimd_capability_t)(supported & allowed);
    //                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                               位与：只保留两者都同意的能力

    // 根据数据类型分发到对应的分发函数
    switch (datatype) {
    case simsimd_datatype_f64_k:
        _simsimd_find_kernel_punned_f64(effective, kind, kernel_output, capability_output); break;
    case simsimd_datatype_f32_k:
        _simsimd_find_kernel_punned_f32(effective, kind, kernel_output, capability_output); break;
    case simsimd_datatype_f16_k:
        _simsimd_find_kernel_punned_f16(effective, kind, kernel_output, capability_output); break;
    case simsimd_datatype_bf16_k:
        _simsimd_find_kernel_punned_bf16(effective, kind, kernel_output, capability_output); break;
    case simsimd_datatype_i8_k:
        _simsimd_find_kernel_punned_i8(effective, kind, kernel_output, capability_output); break;
    case simsimd_datatype_b8_k:
        _simsimd_find_kernel_punned_b8(effective, kind, kernel_output, capability_output); break;
    // ... 其他类型
    }
}
```

**`allowed` 参数的用途**：
```python
# Python 用户示例：
import simsimd

# 场景1：想要测试 Haswell 性能，即使有更好的 AVX-512
kernel = simsimd.find_kernel('cos', 'f32',
    allowed=simsimd.Capabilities.haswell)

# 场景2：调试时强制串行
kernel = simsimd.find_kernel('cos', 'f32',
    allowed=simsimd.Capabilities.serial)

# 场景3：生产环境，尽可能好
kernel = simsimd.find_kernel('cos', 'f32',
    allowed=simsimd.Capabilities.any)
```

---

## 第六部分：非规格化数（Denormals）与 FTZ/DAZ

### 6.1 什么是非规格化数

```
IEEE 754 单精度浮点数格式（32位）：
┌─┬──────────┬───────────────────────┐
│S│ Exponent │       Mantissa        │
│1│    8     │          23           │
└─┴──────────┴───────────────────────┘

正规数（Normal）：指数部分 ≠ 0 且 ≠ 255
  值 = (-1)^S × 2^(E-127) × 1.M（隐含1）

非规格化数（Denormal/Subnormal）：指数 = 0，尾数 ≠ 0
  值 = (-1)^S × 2^(-126) × 0.M（无隐含1）
  范围：±1.4 × 10^-45 ～ ±1.2 × 10^-38（极小值）

问题：CPU 处理 Denormal 极慢！
  正规计算：2周期（通过 FPU）
  Denormal计算：100+ 周期（微码辅助）
```

### 6.2 SimSIMD 的 FTZ/DAZ 处理

```c
// simsimd.h:344-358
SIMSIMD_PUBLIC int _simsimd_flush_denormals_x86(void) {
    // MXCSR：x86 SSE/AVX 控制和状态寄存器
    // 控制 SSE/AVX 浮点行为的关键寄存器

#if defined(_MSC_VER)
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= 1 << 15;  // bit 15：Flush-To-Zero (FTZ)
    mxcsr |= 1 << 6;   // bit 6：Denormals-Are-Zero (DAZ)
    _mm_setcsr(mxcsr);
#else
    unsigned int mxcsr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    //                   ^^^^^^^^^^ store MXCSR
    mxcsr |= 1 << 15;  // FTZ
    mxcsr |= 1 << 6;   // DAZ
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
    //                   ^^^^^^^^^^ load MXCSR
#endif
    return 1;
}
```

**FTZ 和 DAZ 的含义**：

```
FTZ（Flush-To-Zero）：
  - MXCSR bit 15
  - 效果：SIMD 计算结果如果是 Denormal → 直接输出 0
  - 用途：防止 Denormal "输出"

DAZ（Denormals-Are-Zero）：
  - MXCSR bit 6
  - 效果：SIMD 计算 Denormal 输入时 → 视为 0
  - 用途：防止 Denormal "输入"

组合效果：
  任何 Denormal（无论输入还是输出）都当 0 处理
  代价：轻微的精度损失（在极小值附近）
  收益：消除 100+ 周期的微码辅助惩罚

实际影响：
  在 AI/ML 推理中（激活值接近0），Denormal 很常见
  不启用 FTZ/DAZ 可能导致神经网络推理慢 10x+
```

### 6.3 每线程的重要性

```c
// simsimd.h 注释
// @note This should be called on each thread before any SIMD operations
//       to avoid performance penalties.

// 原因：MXCSR 是 per-thread 的寄存器
// 每个线程有自己的 MXCSR 副本
// 主线程设置 FTZ/DAZ 不影响工作线程

// 正确用法（多线程环境）：
void worker_thread_init(void) {
    simsimd_flush_denormals();  // 每个线程启动时调用
    // ... 后续 SIMD 操作都受益
}
```

---

## 第七部分：编译时目标选择机制

### 7.1 GCC pragma 控制

```c
// 以 binary.h 的 NEON 实现为例
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
//                 ^^^^^^^^^^^^^^^^^^^^^^^
//                 为这个区域的代码指定目标架构
//                 即使全局编译选项是 -march=armv8-a（无NEON）
//                 这个区域内也会生成 NEON 指令

#pragma clang attribute push(
    __attribute__((target("arch=armv8.2-a+simd"))),
    apply_to = function)
// Clang 的等价写法（语法不同）

SIMSIMD_PUBLIC void simsimd_hamming_b8_neon(...) {
    // ... 此函数用 armv8.2-a+simd 编译
}

#pragma clang attribute pop
#pragma GCC pop_options
// 恢复全局设置
```

**这解决了什么问题**：
```
问题：单个编译单元（.c 文件）里需要包含多个 CPU 后端
     不同后端需要不同的编译器 flags

传统方案：每个后端单独一个 .c 文件，分别编译
  simsimd_haswell.c → gcc -mavx2 -mfma -mf16c ...
  simsimd_skylake.c → gcc -mavx512f ...
  缺点：链接复杂，难以头文件only发布

SimSIMD 方案：pragma 动态切换
  头文件内每个实现块用 pragma 指定自己的目标
  单个包含头文件的编译单元就能得到所有后端
  优点：header-only，使用简单

代价：函数不能内联（target attribute 不同的函数不可内联到主代码）
```

### 7.2 编译器宏检测链

```c
// types.h 中的检测链（简化）
// 这是 SimSIMD 最核心的宏设计之一

// 步骤1：检测目标架构
#if defined(__x86_64__) || defined(_M_X64)
#define _SIMSIMD_TARGET_X86 1
#else
#define _SIMSIMD_TARGET_X86 0
#endif

// 步骤2：检测编译器是否已启用对应特性
#if !defined(SIMSIMD_TARGET_HASWELL)
    #if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
        #define SIMSIMD_TARGET_HASWELL 1   // 编译器支持
    #else
        #define SIMSIMD_TARGET_HASWELL 0   // 编译器不支持
    #endif
#endif
// 如果用户手动 -DSIMSIMD_TARGET_HASWELL=0，则跳过检测（覆盖）

// 步骤3：pragma 在检测到支持时才生效
#if SIMSIMD_TARGET_HASWELL
#pragma GCC target("avx2,fma,f16c")
// ... Haswell 实现
#pragma GCC pop_options
#endif
```

---

## 第八部分：度量种类与数据类型枚举设计

### 8.1 度量种类的 ASCII 编码技巧

```c
// simsimd.h:128-173
typedef enum {
    simsimd_metric_dot_k      = 'i',  // Inner product
    simsimd_metric_vdot_k     = 'v',  // complex (Vector) dot
    simsimd_metric_cos_k      = 'c',  // Cosine
    simsimd_metric_l2_k       = '2',  // L2 (Euclidean)
    simsimd_metric_l2sq_k     = 'e',  // Euclidean squared
    simsimd_metric_hamming_k  = 'h',  // Hamming
    simsimd_metric_jaccard_k  = 'j',  // Jaccard
    simsimd_metric_intersect_k= 'x',  // Intersection
    simsimd_metric_bilinear_k = 'b',  // Bilinear
    simsimd_metric_mahalanobis_k = 'm', // Mahalanobis
    simsimd_metric_kl_k       = 'k',  // KL divergence
    simsimd_metric_js_k       = 's',  // Jensen-Shannon
    simsimd_metric_fma_k      = 'f',  // Fused-Multiply-Add
    simsimd_metric_wsum_k     = 'w',  // Weighted Sum
} simsimd_metric_kind_t;
```

**使用字符而非整数的优势**：
```c
// 字符值可以直接传递给 Python/Rust/JS
// 在调试器中可以直接看懂（'c' vs 2）
// 可以用字符串直接解析：
simsimd_metric_kind_t parse_metric(const char* name) {
    // 单字符快速匹配（无需strcmp）
    return (simsimd_metric_kind_t)name[0];
}
```

### 8.2 数据类型的 bitmask 设计

```c
// simsimd.h:209-235
typedef enum {
    simsimd_datatype_unknown_k = 0,
    simsimd_datatype_b8_k      = 1 << 1,   // 二进制（每位1比特）
    simsimd_datatype_i4x2_k    = 1 << 19,  // 4位整数（两个打包在一字节）
    simsimd_datatype_i8_k      = 1 << 2,   // 8位整数
    simsimd_datatype_i32_k     = 1 << 4,   // 32位整数
    simsimd_datatype_f64_k     = 1 << 10,  // 双精度浮点
    simsimd_datatype_f32_k     = 1 << 11,  // 单精度浮点
    simsimd_datatype_f16_k     = 1 << 12,  // 半精度浮点
    simsimd_datatype_bf16_k    = 1 << 13,  // Brain float
    simsimd_datatype_f32c_k    = 1 << 21,  // 复数单精度
    simsimd_datatype_f16c_k    = 1 << 22,  // 复数半精度
} simsimd_datatype_t;
```

**复数类型与实数类型的关系**：
```
f32c_k (bit 21) vs f32_k (bit 11)：
  - 相差 10 位
  - 可以通过 >> 10 检查是否是对应的实数类型
  - 或者通过 mask 快速转换：
    is_complex = (dt & 0xF00000) != 0

这种编码允许用一次位操作判断类型族
```

---

## 第九部分：完整分发流程图解

```
用户调用：simsimd.cdist(a, b, metric='cos')
│
├── Python 层（python/lib.c）
│   ├── 解析参数（METH_FASTCALL）
│   ├── 获取 numpy array 的 dtype
│   └── 调用 C 层
│
├── C 层：simsimd_find_kernel_punned()
│   ├── 输入：kind='c', dtype=f32, supported=<CPU能力>, allowed=any
│   ├── effective = supported & allowed
│   └── 调用 _simsimd_find_kernel_punned_f32(effective, 'c', ...)
│
├── 分发表：_simsimd_find_kernel_punned_f32()
│   ├── SVE 支持且类型匹配？ → simsimd_cos_f32_sve
│   ├── NEON 支持且类型匹配？ → simsimd_cos_f32_neon
│   ├── Skylake 支持且类型匹配？ → simsimd_cos_f32_skylake
│   ├── Haswell 支持且类型匹配？ → simsimd_cos_f32_haswell
│   └── Serial 兜底 → simsimd_cos_f32_serial
│
└── 执行：kernel_punned(a, b, n, &result)
    └── 实际运行最优实现，返回距离值
```

---

## 第十部分：实践练习

### 练习1：读取当前 CPU 能力

```c
#include "simsimd/simsimd.h"
#include <stdio.h>

void print_cpu_capabilities(void) {
    simsimd_capability_t caps = simsimd_capabilities();

    printf("CPU 能力：\n");
    if (caps & simsimd_cap_haswell_k)   printf("  ✓ Haswell (AVX2+FMA+F16C)\n");
    if (caps & simsimd_cap_skylake_k)   printf("  ✓ Skylake (AVX-512)\n");
    if (caps & simsimd_cap_ice_k)       printf("  ✓ Ice Lake (VPOPCNTDQ+...)\n");
    if (caps & simsimd_cap_genoa_k)     printf("  ✓ Genoa (BF16)\n");
    if (caps & simsimd_cap_sapphire_k)  printf("  ✓ Sapphire Rapids (FP16)\n");
    if (caps & simsimd_cap_neon_k)      printf("  ✓ ARM NEON\n");
    if (caps & simsimd_cap_sve_k)       printf("  ✓ ARM SVE\n");
}

// Python 等价：
// import simsimd
// print(simsimd.get_capabilities())
```

### 练习2：手写简化版 CPUID 解析

```c
#include <stdio.h>

typedef struct {
    unsigned eax, ebx, ecx, edx;
} cpuid_result_t;

cpuid_result_t cpuid(unsigned leaf, unsigned subleaf) {
    cpuid_result_t r;
    __asm__ volatile("cpuid"
        : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx), "=d"(r.edx)
        : "a"(leaf), "c"(subleaf));
    return r;
}

void check_avx2(void) {
    cpuid_result_t r = cpuid(7, 0);
    int has_avx2 = (r.ebx >> 5) & 1;
    printf("AVX2：%s\n", has_avx2 ? "支持" : "不支持");
}

void check_avx512_bf16(void) {
    cpuid_result_t r = cpuid(7, 1);  // 子叶 1！
    int has_bf16 = (r.eax >> 5) & 1;
    printf("AVX-512 BF16：%s\n", has_bf16 ? "支持" : "不支持");
}
```

### 练习3：理解 `allowed` 参数的用途

```python
import simsimd
import numpy as np

a = np.random.randn(1536).astype(np.float32)
b = np.random.randn(1536).astype(np.float32)

# 获取当前CPU支持的能力
caps = simsimd.get_capabilities()
print(f"CPU 能力: {caps}")

# 强制使用串行实现（用于调试或基准对比）
result_serial = simsimd.dot(a, b, allowed=simsimd.Capabilities.serial)

# 使用最佳实现
result_best = simsimd.dot(a, b)

print(f"串行结果: {result_serial}")
print(f"最佳结果: {result_best}")
print(f"差异: {abs(result_serial - result_best)}")  # 应该非常小
```

---

## 总结

### 核心设计模式

1. **CPUID 三次调用**：leaf 1 查基础特性，leaf 7 查扩展特性，leaf 7+subleaf 1 查 BF16
2. **位掩码能力系统**：x86 占 bit 10-16，ARM 占 bit 20-29，serial 永远是 bit 0
3. **分层分发**：`find_kernel_punned` → 按数据类型路由 → 按能力选最强实现
4. **类型双关**：所有内核函数存为 `void (*)(void*)`，调用时转回真实类型
5. **FTZ/DAZ**：每线程调用，把 Denormal 当 0 处理，避免 100+ 周期惩罚
6. **pragma 动态切换**：单个头文件包含多个 CPU 后端，编译时按区域切换目标

### 分发机制的权衡

```
方案         优点                  缺点
─────────────────────────────────────────────────────
静态分发      代码最小，无运行时开销  一个二进制只针对一种CPU
动态分发      一个二进制支持所有CPU   稍大，首次检测约~10μs
ifunc         OS级别，更快          仅限Linux，可移植性差
SimSIMD选择：动态分发（默认）+ 静态分发选项
```

---

**下一课程**: [补充5：量化技术与混合精度计算](./补充5_量化技术与混合精度计算.md)
