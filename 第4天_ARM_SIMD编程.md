# 第4天：ARM SIMD编程 - NEON与SVE

## 课程目标
- 掌握ARM NEON指令集
- 理解SVE（可变长度向量扩展）
- 对比x86与ARM的SIMD差异
- 学习跨平台SIMD设计模式

---

## 第一部分：ARM SIMD生态概览

### 1.1 ARM vs x86 SIMD对比

| 特性 | ARM NEON | ARM SVE | x86 AVX2 | x86 AVX-512 |
|------|----------|---------|----------|-------------|
| 向量宽度 | 128 bits | 128-2048 bits | 256 bits | 512 bits |
| 长度 | 固定 | 可变 | 固定 | 固定 |
| 首次引入 | 2005 | 2016 | 2013 | 2015 |
| 掩码支持 | 有限 | 完整（predicate） | 无 | 完整 |
| 生态成熟度 | 高 | 中 | 高 | 中 |

### 1.2 NEON数据类型

```c
// NEON向量类型
int8x16_t    vec_i8;    // 16× int8
int16x8_t    vec_i16;   // 8× int16
int32x4_t    vec_i32;   // 4× int32
float32x4_t  vec_f32;   // 4× float32
float64x2_t  vec_f64;   // 2× float64

// "x2", "x3", "x4" 结构（多向量）
float32x4x2_t  vec_pair;   // 2个向量
float32x4x4_t  vec_quad;   // 4个向量（常用于交错加载）
```

### 1.3 NEON Intrinsics命名规则

```
v<operation><type>_<modifier>

示例：
vaddq_f32      - Vector Add Quad (128-bit) Float32
vmulq_f32      - Vector Multiply Quad Float32
vld1q_f32      - Vector Load 1 Quad Float32
vst1q_f32      - Vector Store 1 Quad Float32

后缀含义：
q  = Quad word (128 bits)
_f32 = float32类型
_s32 = signed int32
_u8  = unsigned int8
```

---

## 第二部分：NEON基础操作

### 2.1 加载与存储

```c
// 加载（无对齐要求）
float32x4_t vld1q_f32(float const* ptr);  // 加载4个float

// 存储
void vst1q_f32(float* ptr, float32x4_t vec);

// 广播（所有元素设为同一值）
float32x4_t vdupq_n_f32(float value);  // {value, value, value, value}

// 初始化为零
float32x4_t vec = vdupq_n_f32(0.0f);
```

### 2.2 算术运算

```c
// 加法
float32x4_t vaddq_f32(float32x4_t a, float32x4_t b);

// 减法
float32x4_t vsubq_f32(float32x4_t a, float32x4_t b);

// 乘法
float32x4_t vmulq_f32(float32x4_t a, float32x4_t b);

// FMA (Fused Multiply-Add)
float32x4_t vfmaq_f32(float32x4_t c, float32x4_t a, float32x4_t b);
// 计算：c + a*b

// FMS (Fused Multiply-Subtract)
float32x4_t vfmsq_f32(float32x4_t c, float32x4_t a, float32x4_t b);
// 计算：c - a*b
```

### 2.3 水平归约（NEON风格）

```c
// NEON的水平归约需要更多步骤
float horizontal_sum_neon(float32x4_t vec) {
    // vec = {v0, v1, v2, v3}

    // 方法1：使用vpadd（pairwise add）
    float32x2_t sum_pair = vadd_f32(
        vget_low_f32(vec),      // {v0, v1}
        vget_high_f32(vec)      // {v2, v3}
    );  // -> {v0+v2, v1+v3}
    sum_pair = vpadd_f32(sum_pair, sum_pair);  // {v0+v1+v2+v3, ...}
    return vget_lane_f32(sum_pair, 0);

    // 方法2：使用vaddvq（ARMv8.1+）
    return vaddvq_f32(vec);  // 直接返回所有元素的和
}
```

---

## 第三部分：NEON点积实现

### 3.1 查看SimSIMD的NEON实现

```bash
grep -A 60 "simsimd_dot_f32_neon" /home/dev/SimSIMD/include/simsimd/dot.h | head -80
```

### 3.2 完整NEON实现

```c
SIMSIMD_PUBLIC void simsimd_dot_f32_neon(
    simsimd_f32_t const* a,
    simsimd_f32_t const* b,
    simsimd_size_t n,
    simsimd_distance_t* result) {

    float32x4_t ab_vec = vdupq_n_f32(0.0f);  // 初始化累加器

    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);

        // FMA累加
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    }

    // 归约
    simsimd_f32_t ab = vaddvq_f32(ab_vec);  // ARMv8.1+
    // 或使用兼容方式：
    // ab = vgetq_lane_f32(ab_vec, 0) + vgetq_lane_f32(ab_vec, 1) +
    //      vgetq_lane_f32(ab_vec, 2) + vgetq_lane_f32(ab_vec, 3);

    // 尾部处理
    for (; i < n; ++i) {
        ab += a[i] * b[i];
    }

    *result = ab;
}
```

### 3.3 NEON的int8优化

NEON对整数运算有特殊支持：

```c
SIMSIMD_PUBLIC void simsimd_dot_i8_neon(
    simsimd_i8_t const* a,
    simsimd_i8_t const* b,
    simsimd_size_t n,
    simsimd_distance_t* result) {

    int32x4_t ab_vec = vdupq_n_s32(0);

    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        // 加载16个int8
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);

        // 方法1：使用vmlal（multiply-accumulate long）
        int16x8_t a_low = vmovl_s8(vget_low_s8(a_vec));   // 扩展为int16
        int16x8_t b_low = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_high = vmovl_s8(vget_high_s8(b_vec));

        // 乘法并累加到int32
        ab_vec = vmlal_s16(ab_vec, vget_low_s16(a_low), vget_low_s16(b_low));
        ab_vec = vmlal_s16(ab_vec, vget_high_s16(a_low), vget_high_s16(b_low));
        // ... (处理high部分)

        // 方法2：使用dot product指令（ARMv8.2+）
        #if defined(__ARM_FEATURE_DOTPROD)
        ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);  // 一条指令！
        #endif
    }

    // 归约并输出
    simsimd_i32_t ab = vaddvq_s32(ab_vec);
    *result = ab;
}
```

---

## 第四部分：SVE - 可变长度向量

### 4.1 SVE的革命性设计

SVE与NEON/AVX-512的本质区别：

```c
// NEON/AVX：固定长度，编译时确定
__m256 vec;  // 总是256位

// SVE：运行时确定长度
svfloat32_t vec;  // 可以是128, 256, 512, 1024, 2048位
```

**VL（Vector Length）**：SVE向量的实际长度在运行时通过硬件决定。

### 4.2 SVE编程模型

```c
// SVE类型
svfloat32_t  vec_f32;  // 可变长度float32向量
svint32_t    vec_i32;  // 可变长度int32向量

// SVE谓词（Predicate，类似AVX-512的mask）
svbool_t  pred;  // 谓词，控制哪些元素参与操作
```

### 4.3 SVE的谓词操作

```c
// 创建谓词：全部为真
svbool_t ptrue = svptrue_b32();  // 所有元素都激活

// 创建谓词：前N个为真
svbool_t p = svwhilelt_b32(0, n);  // 激活前n个元素

// 使用谓词加载
svfloat32_t vec = svld1_f32(pred, ptr);  // 只加载pred=true的元素
```

### 4.4 SVE点积实现

```c
SIMSIMD_PUBLIC void simsimd_dot_f32_sve(
    simsimd_f32_t const* a,
    simsimd_f32_t const* b,
    simsimd_size_t n,
    simsimd_distance_t* result) {

    svfloat32_t ab_vec = svdup_f32(0.0f);  // 初始化累加器

    simsimd_size_t i = 0;
    // 主循环：向量长度自适应
    for (; i + svcntw() <= n; i += svcntw()) {
        // svcntw() 返回当前向量可容纳的float32数量

        svbool_t pg = svptrue_b32();  // 所有元素都激活
        svfloat32_t a_vec = svld1_f32(pg, a + i);
        svfloat32_t b_vec = svld1_f32(pg, b + i);

        // FMA累加
        ab_vec = svmla_f32_x(pg, ab_vec, a_vec, b_vec);
        // 等价于：ab_vec = ab_vec + a_vec * b_vec
    }

    // 尾部处理：使用谓词
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);  // 只激活剩余元素
        svfloat32_t a_vec = svld1_f32(pg, a + i);
        svfloat32_t b_vec = svld1_f32(pg, b + i);
        ab_vec = svmla_f32_m(pg, ab_vec, a_vec, b_vec);
    }

    // 归约（水平求和）
    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);

    *result = ab;
}
```

**SVE的优势**：
1. **无需尾部循环**：谓词自动处理
2. **硬件适应性**：同一代码在不同VL的CPU上都能运行
3. **未来兼容**：代码可以利用未来更宽的向量

---

## 第五部分：跨平台设计模式

### 5.1 SimSIMD的跨平台策略

```c
// 统一的API接口
SIMSIMD_PUBLIC void simsimd_dot_f32(
    simsimd_f32_t const* a,
    simsimd_f32_t const* b,
    simsimd_size_t n,
    simsimd_distance_t* result) {

    // 编译时分发
    #if SIMSIMD_TARGET_SVE
        simsimd_dot_f32_sve(a, b, n, result);
    #elif SIMSIMD_TARGET_NEON
        simsimd_dot_f32_neon(a, b, n, result);
    #elif SIMSIMD_TARGET_SKYLAKE
        simsimd_dot_f32_skylake(a, b, n, result);
    #elif SIMSIMD_TARGET_HASWELL
        simsimd_dot_f32_haswell(a, b, n, result);
    #else
        simsimd_dot_f32_serial(a, b, n, result);
    #endif
}
```

### 5.2 条件编译模式

```c
// 使用pragma控制编译选项
#if defined(_SIMSIMD_TARGET_ARM) && SIMSIMD_TARGET_NEON

#pragma GCC push_options
#pragma GCC target("+simd")

SIMSIMD_PUBLIC void simsimd_dot_f32_neon(...) {
    // NEON实现
}

#pragma GCC pop_options

#endif // SIMSIMD_TARGET_NEON
```

### 5.3 运行时分发示例

```c
// 功能检测
simsimd_capability_t caps = simsimd_capabilities();

// 选择最优实现
if (caps & simsimd_cap_sve_k) {
    kernel = simsimd_dot_f32_sve;
} else if (caps & simsimd_cap_neon_k) {
    kernel = simsimd_dot_f32_neon;
} else {
    kernel = simsimd_dot_f32_serial;
}

// 调用
kernel(a, b, n, &result);
```

---

## 第六部分：实战与作业

### 6.1 在ARM平台编译测试

```bash
# 检查CPU特性
lscpu | grep -i neon
lscpu | grep -i sve

# 编译NEON版本
gcc -O3 -march=armv8-a+simd -o test_neon test_arm.c -lm

# 编译SVE版本（需要ARMv9或Graviton3+）
gcc -O3 -march=armv8-a+sve -o test_sve test_arm.c -lm

# 运行
./test_neon
```

### 6.2 作业

#### 作业1：NEON实现
实现NEON版本的cosine distance和L2 distance。

#### 作业2：性能对比
对比NEON (4×) vs SVE (可变) vs Serial的性能。

#### 作业3：跨平台测试
在x86和ARM平台上运行相同代码，对比结果。

#### 作业4：阅读源码
```bash
# 查看所有NEON实现
grep "simsimd.*_neon" /home/dev/SimSIMD/include/simsimd/*.h

# 查看SVE实现
grep "simsimd.*_sve" /home/dev/SimSIMD/include/simsimd/*.h
```

### 预习第5天内容

明天我们将学习：
- **稀疏向量算法**
- **集合交集优化**
- **Galloping search**
- **AVX-512的VP2INTERSECT指令**

---

**下一课**：[第5天：稀疏向量与高级算法](./第5天_稀疏向量与高级算法.md)
