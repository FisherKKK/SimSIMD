/**
 *  @file       spatial.h
 *  @brief      SIMD-accelerated Spatial Similarity Measures.
 *  @author     Ash Vardanian
 *  @date       March 14, 2023
 *
 *  Contains:
 *  - L2 (Euclidean) regular and squared distance
 *  - Cosine (Angular) distance - @b not similarity!
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integral numbers
 *  - 8-bit signed integral numbers
 *  - 4-bit signed integral numbers
 *
 *  For hardware architectures:
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_SPATIAL_H
#define SIMSIMD_SPATIAL_H

#include "types.h"

#include "dot.h" // `_simsimd_reduce_f32x8_haswell`

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_l2_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const*, simsimd_size_t n, simsimd_distance_t* d);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
SIMSIMD_PUBLIC void simsimd_l2_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const*, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 *  Arm v8之前的所有设备
 */
SIMSIMD_PUBLIC void simsimd_l2_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
SIMSIMD_PUBLIC void simsimd_l2_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 *  AVX2, 支持最广泛的CPU架构, 从Haswell开始, 并且编译器已经为f32和f64实现了良好的auto vectorized
 */
SIMSIMD_PUBLIC void simsimd_l2_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer, using 32-bit arithmetic over 512-bit words.
 *  Skylake was launched in 2015, and discontinued in 2019. Skylake had support for F, CD, VL, DQ, and BW extensions,
 *  as well as masked operations. This is enough to supersede auto-vectorization on `f32` and `f64` types.
 * 
 *  Sadly, we can't effectively interleave different kinds of arithmetic instructions to utilize more ports:
 * 
 *  > Like Intel server architectures since Skylake-X, SPR cores feature two 512-bit FMA units, and organize them in a similar fashion. 
 *  > One 512-bit FMA unit is created by fusing two 256-bit ones on port 0 and port 1. The other is added to port 5, as a server-specific 
 *  > core extension. The FMA units on port 0 and 1 are configured into 2×256-bit or 1×512-bit mode depending on whether 512-bit FMA 
 *  > instructions are present in the scheduler. That means a mix of 256-bit and 512-bit FMA instructions will not achieve higher IPC 
 *  > than executing 512-bit instructions alone.
 * 
 *  Source: https://chipsandcheese.com/p/a-peek-at-sapphire-rapids
 */
SIMSIMD_PUBLIC void simsimd_l2_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for AVX512 CPUs of Ice Lake generation and newer, using mixed arithmetic over 512-bit words.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
SIMSIMD_PUBLIC void simsimd_l2_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);
SIMSIMD_PUBLIC void simsimd_cos_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* d);

/*  SIMD-powered backends for AVX-INT8-VNNI extensions on Xeon 6 CPUs, including Sierra Forest and Granite Rapids.
 *  The packs many "efficiency" cores into a single socket, avoiding heavy 512-bit operations, and focusing on 256-bit ones.
 */
SIMSIMD_PUBLIC void simsimd_cos_i8_sierra(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* d);
// clang-format on

// 这边之后应该就是宏的动态分发
// 主要是根据: 架构 + 输入类型 + 中间结果累加类型 + 类型转换器 --> 函数生成

// 最基本的L2 Square
#define SIMSIMD_MAKE_L2SQ(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_l2sq_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                           simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                           simsimd_distance_t *result) {                        \
        simsimd_##accumulator_type##_t d2 = 0;                                                                  \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                               \
            simsimd_##accumulator_type##_t ai = load_and_convert(a + i);                                        \
            simsimd_##accumulator_type##_t bi = load_and_convert(b + i);                                        \
            d2 += (ai - bi) * (ai - bi);                                                                        \
        }                                                                                                       \
        *result = d2;                                                                                           \
    }

// 最基本的L2
#define SIMSIMD_MAKE_L2(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_l2_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                         simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                         simsimd_distance_t *result) {                        \
        simsimd_l2sq_##input_type##_##name(a, b, n, result);                                                  \
        *result = SIMSIMD_SQRT(*result);                                                                      \
    }

// 角度计算
#define SIMSIMD_MAKE_COS(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_cos_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_distance_t *result) {                        \
        simsimd_##accumulator_type##_t ab = 0, a2 = 0, b2 = 0;                                                 \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            simsimd_##accumulator_type##_t ai = load_and_convert(a + i);                                       \
            simsimd_##accumulator_type##_t bi = load_and_convert(b + i);                                       \
            ab += ai * bi;                                                                                     \
            a2 += ai * ai;                                                                                     \
            b2 += bi * bi;                                                                                     \
        }                                                                                                      \
        if (a2 == 0 && b2 == 0) { *result = 0; }                                                               \
        else if (ab == 0) { *result = 1; }                                                                     \
        else {                                                                                                 \
            simsimd_distance_t unclipped_result = 1 - ab * SIMSIMD_RSQRT(a2) * SIMSIMD_RSQRT(b2);              \
            *result = unclipped_result > 0 ? unclipped_result : 0;                                             \
        }                                                                                                      \
    }

// 对于顺序的直接解引用load and convert直接解引用即可
SIMSIMD_MAKE_COS(serial, f64, f64, SIMSIMD_DEREFERENCE)  // simsimd_cos_f64_serial
SIMSIMD_MAKE_L2SQ(serial, f64, f64, SIMSIMD_DEREFERENCE) // simsimd_l2sq_f64_serial
SIMSIMD_MAKE_L2(serial, f64, f64, SIMSIMD_DEREFERENCE)   // simsimd_l2_f64_serial

SIMSIMD_MAKE_COS(serial, f32, f32, SIMSIMD_DEREFERENCE)  // simsimd_cos_f32_serial
SIMSIMD_MAKE_L2SQ(serial, f32, f32, SIMSIMD_DEREFERENCE) // simsimd_l2sq_f32_serial
SIMSIMD_MAKE_L2(serial, f32, f32, SIMSIMD_DEREFERENCE)   // simsimd_l2_f32_serial

// 根据累加器类型要进行变换
SIMSIMD_MAKE_COS(serial, f16, f32, SIMSIMD_F16_TO_F32)  // simsimd_cos_f16_serial
SIMSIMD_MAKE_L2SQ(serial, f16, f32, SIMSIMD_F16_TO_F32) // simsimd_l2sq_f16_serial
SIMSIMD_MAKE_L2(serial, f16, f32, SIMSIMD_F16_TO_F32)   // simsimd_l2_f16_serial

SIMSIMD_MAKE_COS(serial, bf16, f32, SIMSIMD_BF16_TO_F32)  // simsimd_cos_bf16_serial
SIMSIMD_MAKE_L2SQ(serial, bf16, f32, SIMSIMD_BF16_TO_F32) // simsimd_l2sq_bf16_serial
SIMSIMD_MAKE_L2(serial, bf16, f32, SIMSIMD_BF16_TO_F32)   // simsimd_l2_bf16_serial

SIMSIMD_MAKE_COS(serial, i8, i32, SIMSIMD_DEREFERENCE)  // simsimd_cos_i8_serial
SIMSIMD_MAKE_L2SQ(serial, i8, i32, SIMSIMD_DEREFERENCE) // simsimd_l2sq_i8_serial
SIMSIMD_MAKE_L2(serial, i8, i32, SIMSIMD_DEREFERENCE)   // simsimd_l2_i8_serial

SIMSIMD_MAKE_COS(serial, u8, i32, SIMSIMD_DEREFERENCE)  // simsimd_cos_u8_serial
SIMSIMD_MAKE_L2SQ(serial, u8, i32, SIMSIMD_DEREFERENCE) // simsimd_l2sq_u8_serial
SIMSIMD_MAKE_L2(serial, u8, i32, SIMSIMD_DEREFERENCE)   // simsimd_l2_u8_serial

// 累加器是f64, 更加高精度版本
SIMSIMD_MAKE_COS(accurate, f32, f64, SIMSIMD_DEREFERENCE)  // simsimd_cos_f32_accurate
SIMSIMD_MAKE_L2SQ(accurate, f32, f64, SIMSIMD_DEREFERENCE) // simsimd_l2sq_f32_accurate
SIMSIMD_MAKE_L2(accurate, f32, f64, SIMSIMD_DEREFERENCE)   // simsimd_l2_f32_accurate

SIMSIMD_MAKE_COS(accurate, f16, f64, SIMSIMD_F16_TO_F32)  // simsimd_cos_f16_accurate
SIMSIMD_MAKE_L2SQ(accurate, f16, f64, SIMSIMD_F16_TO_F32) // simsimd_l2sq_f16_accurate
SIMSIMD_MAKE_L2(accurate, f16, f64, SIMSIMD_F16_TO_F32)   // simsimd_l2_f16_accurate

SIMSIMD_MAKE_COS(accurate, bf16, f64, SIMSIMD_BF16_TO_F32)  // simsimd_cos_bf16_accurate
SIMSIMD_MAKE_L2SQ(accurate, bf16, f64, SIMSIMD_BF16_TO_F32) // simsimd_l2sq_bf16_accurate
SIMSIMD_MAKE_L2(accurate, bf16, f64, SIMSIMD_BF16_TO_F32)   // simsimd_l2_bf16_accurate

/** #pragma GCC push_options 用于保存目前已有的编译器选项
 *  #pragma GCC xxx 用于设置新的GCC
 *  #pragma GCC pop_options 用于还原之前保存的编译器选项
 * 
 * #pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
 * #pragma clang attribute pop
 */

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_f32_t _simsimd_sqrt_f32_neon(simsimd_f32_t x) {
    return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
}
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_sqrt_f64_neon(simsimd_f64_t x) {
    return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0);
}
SIMSIMD_INTERNAL simsimd_distance_t _simsimd_cos_normalize_f32_neon(simsimd_f32_t ab, simsimd_f32_t a2,
                                                                    simsimd_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    simsimd_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third party research suggests, that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts = vrsqrte_f32(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    vst1_f32(squares_arr, rsqrts);
    simsimd_distance_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

SIMSIMD_INTERNAL simsimd_distance_t _simsimd_cos_normalize_f64_neon(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                    simsimd_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    simsimd_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third party research suggests, that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts = vrsqrteq_f64(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    vst1q_f64(squares_arr, rsqrts);
    simsimd_distance_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

SIMSIMD_PUBLIC void simsimd_l2_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_l2sq_f32_neon(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        simsimd_f32_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_cos_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    }
    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f64_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_l2sq_f64_neon(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    float64x2_t sum_vec = vdupq_n_f64(0);
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        float64x2_t diff_vec = vsubq_f64(a_vec, b_vec);
        sum_vec = vfmaq_f64(sum_vec, diff_vec, diff_vec);
    }
    simsimd_f64_t sum = vaddvq_f64(sum_vec);
    for (; i < n; ++i) {
        simsimd_f64_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_cos_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    float64x2_t ab_vec = vdupq_n_f64(0), a2_vec = vdupq_n_f64(0), b2_vec = vdupq_n_f64(0);
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        ab_vec = vfmaq_f64(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f64(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f64(b2_vec, b_vec, b_vec);
    }
    simsimd_f64_t ab = vaddvq_f64(ab_vec), a2 = vaddvq_f64(a2_vec), b2 = vaddvq_f64(b2_vec);
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f64_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_l2sq_f16_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    float32x4_t a_vec, b_vec;
    float32x4_t sum_vec = vdupq_n_f32(0);

simsimd_l2sq_f16_neon_cycle:
    if (n < 4) {
        a_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
    sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    if (n) goto simsimd_l2sq_f16_neon_cycle;

    *result = vaddvq_f32(sum_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    float32x4_t a_vec, b_vec;

simsimd_cos_f16_neon_cycle:
    if (n < 4) {
        a_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
    b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    if (n) goto simsimd_cos_f16_neon_cycle;

    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_cos_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {

    // Similar to `simsimd_cos_i8_neon`, we can use the `BFMMLA` instruction through
    // the `vbfmmlaq_f32` intrinsic to compute matrix products and later drop 1/4 of values.
    // The only difference is that `zip` isn't provided for `bf16` and we need to reinterpret back
    // and forth before zipping. Same as with integers, on modern Arm CPUs, this "smart"
    // approach is actually slower by around 25%.
    //
    //   float32x4_t products_low_vec = vdupq_n_f32(0.0f);
    //   float32x4_t products_high_vec = vdupq_n_f32(0.0f);
    //   for (; i + 8 <= n; i += 8) {
    //       bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)a + i);
    //       bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)b + i);
    //       int16x8_t a_vec_s16 = vreinterpretq_s16_bf16(a_vec);
    //       int16x8_t b_vec_s16 = vreinterpretq_s16_bf16(b_vec);
    //       int16x8x2_t y_w_vecs_s16 = vzipq_s16(a_vec_s16, b_vec_s16);
    //       bfloat16x8_t y_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[0]);
    //       bfloat16x8_t w_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[1]);
    //       bfloat16x4_t a_low = vget_low_bf16(a_vec);
    //       bfloat16x4_t b_low = vget_low_bf16(b_vec);
    //       bfloat16x4_t a_high = vget_high_bf16(a_vec);
    //       bfloat16x4_t b_high = vget_high_bf16(b_vec);
    //       bfloat16x8_t x_vec = vcombine_bf16(a_low, b_low);
    //       bfloat16x8_t v_vec = vcombine_bf16(a_high, b_high);
    //       products_low_vec = vbfmmlaq_f32(products_low_vec, x_vec, y_vec);
    //       products_high_vec = vbfmmlaq_f32(products_high_vec, v_vec, w_vec);
    //   }
    //   float32x4_t products_vec = vaddq_f32(products_high_vec, products_low_vec);
    //   simsimd_f32_t a2 = products_vec[0], ab = products_vec[1], b2 = products_vec[3];
    //
    // Another way of accomplishing the same thing is to process the odd and even elements separately,
    // using special `vbfmlaltq_f32` and `vbfmlalbq_f32` intrinsics:
    //
    //      ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_vec, b_vec);
    //      ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_vec, b_vec);
    //      a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_vec, a_vec);
    //      a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_vec, a_vec);
    //      b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_vec, b_vec);
    //      b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_vec, b_vec);
    //

    float32x4_t ab_vec = vdupq_n_f32(0);
    float32x4_t a2_vec = vdupq_n_f32(0);
    float32x4_t b2_vec = vdupq_n_f32(0);
    bfloat16x8_t a_vec, b_vec;

simsimd_cos_bf16_neon_cycle:
    if (n < 8) {
        a_vec = _simsimd_partial_load_bf16x8_neon(a, n);
        b_vec = _simsimd_partial_load_bf16x8_neon(b, n);
        n = 0;
    }
    else {
        a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a);
        b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b);
        n -= 8, a += 8, b += 8;
    }
    ab_vec = vbfdotq_f32(ab_vec, a_vec, b_vec);
    a2_vec = vbfdotq_f32(a2_vec, a_vec, a_vec);
    b2_vec = vbfdotq_f32(b2_vec, b_vec, b_vec);
    if (n) goto simsimd_cos_bf16_neon_cycle;

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_l2sq_bf16_neon(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    float32x4_t diff_high_vec, diff_low_vec;
    float32x4_t sum_high_vec = vdupq_n_f32(0), sum_low_vec = vdupq_n_f32(0);

simsimd_l2sq_bf16_neon_cycle:
    if (n < 8) {
        bfloat16x8_t a_vec = _simsimd_partial_load_bf16x8_neon(a, n);
        bfloat16x8_t b_vec = _simsimd_partial_load_bf16x8_neon(b, n);
        diff_high_vec = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_vec)), vcvt_f32_bf16(vget_high_bf16(b_vec)));
        diff_low_vec = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_vec)), vcvt_f32_bf16(vget_low_bf16(b_vec)));
        n = 0;
    }
    else {
        bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a);
        bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b);
        diff_high_vec = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_vec)), vcvt_f32_bf16(vget_high_bf16(b_vec)));
        diff_low_vec = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_vec)), vcvt_f32_bf16(vget_low_bf16(b_vec)));
        n -= 8, a += 8, b += 8;
    }
    sum_high_vec = vfmaq_f32(sum_high_vec, diff_high_vec, diff_high_vec);
    sum_low_vec = vfmaq_f32(sum_low_vec, diff_low_vec, diff_low_vec);
    if (n) goto simsimd_l2sq_bf16_neon_cycle;

    *result = vaddvq_f32(vaddq_f32(sum_high_vec, sum_low_vec));
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {
    simsimd_l2sq_i8_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {

    // The naive approach is to upcast 8-bit signed integers into 16-bit signed integers
    // for subtraction, then multiply within 16-bit integers and accumulate the results
    // into 32-bit integers. This approach is slow on modern Arm CPUs. On Graviton 4,
    // that approach results in 17 GB/s of throughput, compared to 39 GB/s for `i8`
    // dot-products.
    //
    // Luckily we can use the `vabdq_s8` which technically returns `i8` values, but it's a
    // matter of reinterpret-casting! That approach boosts us to 33 GB/s of throughput.
    uint32x4_t d2_vec = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        uint8x16_t d_vec = vreinterpretq_u8_s8(vabdq_s8(a_vec, b_vec));
        d2_vec = vdotq_u32(d2_vec, d_vec, d_vec);
    }
    simsimd_u32_t d2 = vaddvq_u32(d2_vec);
    for (; i < n; ++i) {
        simsimd_i32_t n = (simsimd_i32_t)a[i] - b[i];
        d2 += (simsimd_u32_t)(n * n);
    }
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {

    simsimd_size_t i = 0;

    // Variant 1.
    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    //
    //  int32x4_t ab_vec = vdupq_n_s32(0);
    //  int32x4_t a2_vec = vdupq_n_s32(0);
    //  int32x4_t b2_vec = vdupq_n_s32(0);
    //  for (simsimd_size_t i = 0; i != n; i += 8) {
    //      int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //      int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //      int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //      int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //      int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //      ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(ab_part_vec))));
    //      a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(a2_part_vec))));
    //      b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(b2_part_vec))));
    //  }
    //
    // Variant 2.
    // With the 128-bit `vdotq_s32` intrinsic, we can use the following code:
    //
    //  for (; i + 16 <= n; i += 16) {
    //      int8x16_t a_vec = vld1q_s8(a + i);
    //      int8x16_t b_vec = vld1q_s8(b + i);
    //      ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    //      a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
    //      b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    //  }
    //
    // Variant 3.
    // To use MMLA instructions, we need to reorganize the contents of the vectors.
    // On input we have `a_vec` and `b_vec`:
    //
    //   a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]
    //   b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //
    // We will be multiplying matrices of size 2x8 and 8x2. So we need to perform a few shuffles:
    //
    //   X =
    //      a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    //      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    //   Y =
    //      a[0], b[0],
    //      a[1], b[1],
    //      a[2], b[2],
    //      a[3], b[3],
    //      a[4], b[4],
    //      a[5], b[5],
    //      a[6], b[6],
    //      a[7], b[7]
    //
    //   V =
    //      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
    //      b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //   W =
    //      a[8],   b[8],
    //      a[9],   b[9],
    //      a[10],  b[10],
    //      a[11],  b[11],
    //      a[12],  b[12],
    //      a[13],  b[13],
    //      a[14],  b[14],
    //      a[15],  b[15]
    //
    // Performing matrix multiplications we can aggregate into a matrix `products_low_vec` and `products_high_vec`:
    //
    //      X * X, X * Y                V * W, V * V
    //      Y * X, Y * Y                W * W, W * V
    //
    // Of those values we need only 3/4, as the (X * Y) and (Y * X) are the same.
    //
    //      int32x4_t products_low_vec = vdupq_n_s32(0), products_high_vec = vdupq_n_s32(0);
    //      int8x16_t a_low_b_low_vec, a_high_b_high_vec;
    //      for (; i + 16 <= n; i += 16) {
    //          int8x16_t a_vec = vld1q_s8(a + i);
    //          int8x16_t b_vec = vld1q_s8(b + i);
    //          int8x16x2_t y_w_vecs = vzipq_s8(a_vec, b_vec);
    //          int8x16_t x_vec = vcombine_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
    //          int8x16_t v_vec = vcombine_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
    //          products_low_vec = vmmlaq_s32(products_low_vec, x_vec, y_w_vecs.val[0]);
    //          products_high_vec = vmmlaq_s32(products_high_vec, v_vec, y_w_vecs.val[1]);
    //      }
    //      int32x4_t products_vec = vaddq_s32(products_high_vec, products_low_vec);
    //      simsimd_i32_t a2 = products_vec[0];
    //      simsimd_i32_t ab = products_vec[1];
    //      simsimd_i32_t b2 = products_vec[3];
    //
    // That solution is elegant, but it requires the additional `+i8mm` extension and is currently slower,
    // at least on AWS Graviton 3.
    int32x4_t ab_vec = vdupq_n_s32(0);
    int32x4_t a2_vec = vdupq_n_s32(0);
    int32x4_t b2_vec = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    }
    simsimd_i32_t ab = vaddvq_s32(ab_vec);
    simsimd_i32_t a2 = vaddvq_s32(a2_vec);
    simsimd_i32_t b2 = vaddvq_s32(b2_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {
    simsimd_l2sq_u8_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    uint32x4_t d2_vec = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
        d2_vec = vdotq_u32(d2_vec, d_vec, d_vec);
    }
    simsimd_u32_t d2 = vaddvq_u32(d2_vec);
    for (; i < n; ++i) {
        simsimd_i32_t n = (simsimd_i32_t)a[i] - b[i];
        d2 += (simsimd_u32_t)(n * n);
    }
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {

    simsimd_size_t i = 0;
    uint32x4_t ab_vec = vdupq_n_u32(0);
    uint32x4_t a2_vec = vdupq_n_u32(0);
    uint32x4_t b2_vec = vdupq_n_u32(0);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        ab_vec = vdotq_u32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_u32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_u32(b2_vec, b_vec, b_vec);
    }
    simsimd_u32_t ab = vaddvq_u32(ab_vec);
    simsimd_u32_t a2 = vaddvq_u32(a2_vec);
    simsimd_u32_t b2 = vaddvq_u32(b2_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_u32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_I8

#if SIMSIMD_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {
    simsimd_l2sq_f32_sve(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        svfloat32_t a_minus_b_vec = svsub_f32_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntw();
    } while (i < n);
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntw();
    } while (i < n);

    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = _simsimd_cos_normalize_f64_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {
    simsimd_l2sq_f64_sve(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat64_t d2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        svfloat64_t a_minus_b_vec = svsub_f64_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f64_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntd();
    } while (i < n);
    simsimd_f64_t d2 = svaddv_f64(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f64_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f64_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntd();
    } while (i < n);

    simsimd_f64_t ab = svaddv_f64(svptrue_b32(), ab_vec);
    simsimd_f64_t a2 = svaddv_f64(svptrue_b32(), a2_vec);
    simsimd_f64_t b2 = svaddv_f64(svptrue_b32(), b2_vec);
    *result = _simsimd_cos_normalize_f64_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_SVE_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_sve(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {
    simsimd_l2sq_f16_sve(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const *a_enum, simsimd_f16_t const *b_enum, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat16_t d2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const *a = (simsimd_f16_for_arm_simd_t const *)(a_enum);
    simsimd_f16_for_arm_simd_t const *b = (simsimd_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        svfloat16_t a_minus_b_vec = svsub_f16_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f16_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f16_for_arm_simd_t d2_f16 = svaddv_f16(svptrue_b16(), d2_vec);
    *result = d2_f16;
}

SIMSIMD_PUBLIC void simsimd_cos_f16_sve(simsimd_f16_t const *a_enum, simsimd_f16_t const *b_enum, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t a2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t b2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const *a = (simsimd_f16_for_arm_simd_t const *)(a_enum);
    simsimd_f16_for_arm_simd_t const *b = (simsimd_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f16_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f16_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    simsimd_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    simsimd_f16_for_arm_simd_t a2 = svaddv_f16(svptrue_b16(), a2_vec);
    simsimd_f16_for_arm_simd_t b2 = svaddv_f16(svptrue_b16(), b2_vec);
    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE_F16

#if SIMSIMD_TARGET_SVE_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_bf16_sve(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    simsimd_l2sq_bf16_sve(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_sve(simsimd_bf16_t const *a_enum, simsimd_bf16_t const *b_enum, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t d2_low_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t d2_high_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    simsimd_u16_t const *a = (simsimd_u16_t const *)(a_enum);
    simsimd_u16_t const *b = (simsimd_u16_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svuint16_t a_vec = svld1_u16(pg_vec, a + i);
        svuint16_t b_vec = svld1_u16(pg_vec, b + i);

        // There is no `bf16` subtraction in SVE, so we need to convert to `u32` and shift.
        svbool_t pg_low_vec = svwhilelt_b32((unsigned int)(i), (unsigned int)n);
        svbool_t pg_high_vec = svwhilelt_b32((unsigned int)(i + svcnth() / 2), (unsigned int)n);
        svfloat32_t a_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(a_vec), 16));
        svfloat32_t a_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(a_vec), 16));
        svfloat32_t b_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(b_vec), 16));
        svfloat32_t b_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(b_vec), 16));

        svfloat32_t a_minus_b_low_vec = svsub_f32_x(pg_low_vec, a_low_vec, b_low_vec);
        svfloat32_t a_minus_b_high_vec = svsub_f32_x(pg_high_vec, a_high_vec, b_high_vec);
        d2_low_vec = svmla_f32_x(pg_vec, d2_low_vec, a_minus_b_low_vec, a_minus_b_low_vec);
        d2_high_vec = svmla_f32_x(pg_vec, d2_high_vec, a_minus_b_high_vec, a_minus_b_high_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_low_vec) + svaddv_f32(svptrue_b32(), d2_high_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_bf16_sve(simsimd_bf16_t const *a_enum, simsimd_bf16_t const *b_enum, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    simsimd_bf16_for_arm_simd_t const *a = (simsimd_bf16_for_arm_simd_t const *)(a_enum);
    simsimd_bf16_for_arm_simd_t const *b = (simsimd_bf16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svbfloat16_t a_vec = svld1_bf16(pg_vec, a + i);
        svbfloat16_t b_vec = svld1_bf16(pg_vec, b + i);
        ab_vec = svbfdot_f32(ab_vec, a_vec, b_vec);
        a2_vec = svbfdot_f32(a2_vec, a_vec, a_vec);
        b2_vec = svbfdot_f32(b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = _simsimd_cos_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE_BF16
#endif // _SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_f32_t _simsimd_sqrt_f32_haswell(simsimd_f32_t x) {
    // _mm_set_ss: copy to float --> dst and return
    // _mm_sqrt_ps: compute sqrt
    // _mm_cvtss_f32: copy lower-32bit --> dst
    // convert single --> f32
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x)));
}

// f64的sqrt
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_sqrt_f64_haswell(simsimd_f64_t x) {
    // set double
    // sqrt double
    // cvts double --> f64
    return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x)));
}

// 所以这个函数本质上是计算f64的余弦, 但是采用 f32向量化 + 牛顿迭代
SIMSIMD_INTERNAL simsimd_distance_t _simsimd_cos_normalize_f64_haswell(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                       simsimd_f64_t b2) {
    /** 1. <a, b> = (a \dot b) / (|a| * |b|)
     *      如果两个模中存在任意一个0, 那么结果就是0
     *      如果分子为0, 那么也是1 - 0 = 1
     */

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0)
        return 1;
    // We want to avoid the `simsimd_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The latency of the native instruction is 4 cycles and it's broadly supported.
    // For single-precision floats it has a maximum relative error of 1.5*2^-12.
    // Higher precision isn't implemented on older CPUs. See `_simsimd_cos_normalize_f64_skylake` for that.

    // set_pack_double --> 128 bit == 2 * double
    __m128d squares = _mm_set_pd(a2, b2);
    
    // 这里使用的是reverse sqrt, 但是事先都转换成了f32
    __m128d rsqrts = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(squares)));
    // Newton-Raphson iteration for reciprocal square root:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    
    // 这里相当于做了牛顿迭代
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    // unpackhi_pd --> unpack high bit and interleave
    simsimd_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    simsimd_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    simsimd_distance_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

// f32的sqrt, 这里就可以直接算
SIMSIMD_INTERNAL simsimd_distance_t _simsimd_cos_normalize_f32_haswell(simsimd_f32_t ab, simsimd_f32_t a2,
                                                                       simsimd_f32_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0.0f)
        return 1.0f;

    // Load the squares into an __m128 register for single-precision floating-point operations
    __m128 squares = _mm_set_ps(a2, b2, a2, b2); // We replicate to make use of full register

    // Compute the reciprocal square root of the squares using `_mm_rsqrt_ps` (single-precision)
    __m128 rsqrts = _mm_rsqrt_ps(squares);

    // Perform one iteration of Newton-Raphson refinement to improve the precision of rsqrt, 进行了一轮精度的提高:
    // Formula: y' = y * (1.5 - 0.5 * x * y * y)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_halves = _mm_set1_ps(1.5f);
    rsqrts =
        _mm_mul_ps(rsqrts, _mm_sub_ps(three_halves, _mm_mul_ps(half, _mm_mul_ps(squares, _mm_mul_ps(rsqrts, rsqrts)))));

    // Extract the reciprocal square roots of a2 and b2 from the __m128 register
    // 这里相当于要将值从_mm128中提取出来, 采用shuffle (根据mask进行选择, 这里就是选择最低位)
    // ctvss_f32变成f32
    simsimd_f32_t a2_reciprocal = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    simsimd_f32_t b2_reciprocal = _mm_cvtss_f32(rsqrts);

    // Calculate the cosine distance: 1 - ab * a2_reciprocal * b2_reciprocal
    simsimd_distance_t result = 1.0f - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

// 计算f16的l2
SIMSIMD_PUBLIC void simsimd_l2_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    simsimd_l2sq_f16_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}

// 计算f16的l2sq (square), 这里相当于所有的结果先被转换为f32做正常操作, reduce部分全部转换为f64, 因为dist是f64!
SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {

    /** AVX2, 采用__m256: 16 * 16 --> 16个f16
     * 
     */
    __m256 a_vec, b_vec;
    
    // 获得一个全0的256
    __m256 d2_vec = _mm256_setzero_ps();

    //! 这里相当于是一个goto分支, 实现了类似循环的作用: 循环展开!
simsimd_l2sq_f16_haswell_cycle:
    if (n < 8) {
        // 剩余的元素数目小于8
        // 从剩余元素中载入所以是partial_load
        a_vec = _simsimd_partial_load_f16x8_haswell(a, n);
        b_vec = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        // unaligned load 128
        // 这里相当于正常处理, 从a和b中读8个f16 --> __m128i
        // 然后将8个f16 --> 8个f32 --> __m256
        // 对n, a, b进行计数
        a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }

    // 对8个f32进行操作, sub ==> fma
    // fma: x[i] * x[i] + x[i + 1] * x[i + 1] ...
    __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
    d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_f16_haswell_cycle;

    // 当计算完成之后进行reduce
    *result = _simsimd_reduce_f32x8_haswell(d2_vec);
}

// 针对f16的向量余弦计算
SIMSIMD_PUBLIC void simsimd_cos_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    // 初始化
    __m256 a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps(), a2_vec = _mm256_setzero_ps(), b2_vec = _mm256_setzero_ps();

    // 循环载入f16 * 8
simsimd_cos_f16_haswell_cycle:
    if (n < 8) {
        // partial load
        a_vec = _simsimd_partial_load_f16x8_haswell(a, n);
        b_vec = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        // load a --> int128 --> f32 * 8
        // update a, b, n
        a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    // 计算内积后求和, fma
    // a * b, a * a, b * b
    ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
    b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    if (n) goto simsimd_cos_f16_haswell_cycle;

    // reduce求和
    simsimd_f32_t ab = _simsimd_reduce_f32x8_haswell(ab_vec);
    simsimd_f32_t a2 = _simsimd_reduce_f32x8_haswell(a2_vec);
    simsimd_f32_t b2 = _simsimd_reduce_f32x8_haswell(b2_vec);
    // 计算cosine
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

// 计算L2
SIMSIMD_PUBLIC void simsimd_l2_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_l2sq_bf16_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}

// 计算bf16的square, 基本上和f16没什么区别, 核心思路都是转换成f32进行后续处理
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                              simsimd_distance_t *result) {
    __m256 a_vec, b_vec;
    __m256 d2_vec = _mm256_setzero_ps();

simsimd_l2sq_bf16_haswell_cycle:
    if (n < 8) {
        a_vec = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(a, n));
        b_vec = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(b, n));
        n = 0;
    }
    else {
        // 转译成f32进行后续计算
        a_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
    d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_bf16_haswell_cycle;

    *result = _simsimd_reduce_f32x8_haswell(d2_vec);
}

// 同样的没区别
SIMSIMD_PUBLIC void simsimd_cos_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    __m256 a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps(), a2_vec = _mm256_setzero_ps(), b2_vec = _mm256_setzero_ps();

simsimd_cos_bf16_haswell_cycle:
    if (n < 8) {
        a_vec = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(a, n));
        b_vec = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(b, n));
        n = 0;
    }
    else {
        a_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
    b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    if (n) goto simsimd_cos_bf16_haswell_cycle;

    simsimd_f32_t ab = _simsimd_reduce_f32x8_haswell(ab_vec);
    simsimd_f32_t a2 = _simsimd_reduce_f32x8_haswell(a2_vec);
    simsimd_f32_t b2 = _simsimd_reduce_f32x8_haswell(b2_vec);
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

// 计算i8的L2
SIMSIMD_PUBLIC void simsimd_l2_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    simsimd_l2sq_i8_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}

// 计算i8的L2_Sqr
SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {

    // initialize, 这里相当于是求和的结果
    __m256i d2_i32_low_vec = _mm256_setzero_si256();
    __m256i d2_i32_high_vec = _mm256_setzero_si256();

    // _m256i = i8 * 32
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Sign extend `i8` to `i16`
        // 将i8 --> 有符号扩展到i16:
        // i8 * 32 --> i8 * 16 | i8 * 16 --> i16 * 16 | i16 * 16
        __m256i a_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8_vec));
        __m256i a_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 1));
        __m256i b_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8_vec));
        __m256i b_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 1));

        // Subtract
        // After this we will be squaring the values. The sign will be dropped
        // and each difference will be in the range [0, 255].
        // 减法, 这里相当于要保证范围
        __m256i d_i16_low_vec = _mm256_sub_epi16(a_i16_low_vec, b_i16_low_vec);
        __m256i d_i16_high_vec = _mm256_sub_epi16(a_i16_high_vec, b_i16_high_vec);

        // Accumulate into `i32` vectors
        // madd: i16 * i16 --> i32, 16个i16 --> 8个i32
        d2_i32_low_vec = _mm256_add_epi32(d2_i32_low_vec, _mm256_madd_epi16(d_i16_low_vec, d_i16_low_vec));
        d2_i32_high_vec = _mm256_add_epi32(d2_i32_high_vec, _mm256_madd_epi16(d_i16_high_vec, d_i16_high_vec));
    }

    // Accumulate the 32-bit integers from `d2_i32_high_vec` and `d2_i32_low_vec`
    // 将结果进行reduce
    int d2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(d2_i32_low_vec, d2_i32_high_vec));

    // Take care of the tail:
    // 处理尾部
    for (; i < n; ++i) {
        int n = (int)(a[i]) - b[i];
        d2 += n * n;
    }

    *result = (simsimd_f64_t)d2;
}

SIMSIMD_PUBLIC void simsimd_cos_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {

    __m256i ab_i32_low_vec = _mm256_setzero_si256();
    __m256i ab_i32_high_vec = _mm256_setzero_si256();
    __m256i a2_i32_low_vec = _mm256_setzero_si256();
    __m256i a2_i32_high_vec = _mm256_setzero_si256();
    __m256i b2_i32_low_vec = _mm256_setzero_si256();
    __m256i b2_i32_high_vec = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping it's sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Unpack `int8` to `int16`
        __m256i a_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 0));
        __m256i a_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 1));
        __m256i b_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 0));
        __m256i b_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 1));

        // Multiply and accumulate as `int16`, accumulate products as `int32`:
        ab_i32_low_vec = _mm256_add_epi32(ab_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, b_i16_low_vec));
        ab_i32_high_vec = _mm256_add_epi32(ab_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, b_i16_high_vec));
        a2_i32_low_vec = _mm256_add_epi32(a2_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, a_i16_low_vec));
        a2_i32_high_vec = _mm256_add_epi32(a2_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, a_i16_high_vec));
        b2_i32_low_vec = _mm256_add_epi32(b2_i32_low_vec, _mm256_madd_epi16(b_i16_low_vec, b_i16_low_vec));
        b2_i32_high_vec = _mm256_add_epi32(b2_i32_high_vec, _mm256_madd_epi16(b_i16_high_vec, b_i16_high_vec));
    }

    // Further reduce to a single sum for each vector
    int ab = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    int a2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(a2_i32_low_vec, a2_i32_high_vec));
    int b2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(b2_i32_low_vec, b2_i32_high_vec));

    // Take care of the tail:
    for (; i < n; ++i) {
        int ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

// 如下是针对u8的一套
SIMSIMD_PUBLIC void simsimd_l2_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    simsimd_l2sq_u8_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {

    __m256i d2_i32_low_vec = _mm256_setzero_si256();
    __m256i d2_i32_high_vec = _mm256_setzero_si256();
    __m256i const zeros_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8_vec = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8_vec = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Substracting unsigned vectors in AVX2 is done by saturating subtraction:
        __m256i d_u8_vec = _mm256_or_si256(_mm256_subs_epu8(a_u8_vec, b_u8_vec), _mm256_subs_epu8(b_u8_vec, a_u8_vec));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        // 这里本质上是补0操作  
        __m256i d_i16_low_vec = _mm256_unpacklo_epi8(d_u8_vec, zeros_vec);
        __m256i d_i16_high_vec = _mm256_unpackhi_epi8(d_u8_vec, zeros_vec);

        // Multiply and accumulate at `int16` level, accumulate at `int32` level:
        d2_i32_low_vec = _mm256_add_epi32(d2_i32_low_vec, _mm256_madd_epi16(d_i16_low_vec, d_i16_low_vec));
        d2_i32_high_vec = _mm256_add_epi32(d2_i32_high_vec, _mm256_madd_epi16(d_i16_high_vec, d_i16_high_vec));
    }

    // Accumulate the 32-bit integers from `d2_i32_high_vec` and `d2_i32_low_vec`
    int d2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(d2_i32_low_vec, d2_i32_high_vec));

    // Take care of the tail:
    for (; i < n; ++i) {
        int n = (int)(a[i]) - b[i];
        d2 += n * n;
    }

    *result = (simsimd_f64_t)d2;
}

SIMSIMD_PUBLIC void simsimd_cos_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {

    __m256i ab_i32_low_vec = _mm256_setzero_si256();
    __m256i ab_i32_high_vec = _mm256_setzero_si256();
    __m256i a2_i32_low_vec = _mm256_setzero_si256();
    __m256i a2_i32_high_vec = _mm256_setzero_si256();
    __m256i b2_i32_low_vec = _mm256_setzero_si256();
    __m256i b2_i32_high_vec = _mm256_setzero_si256();
    __m256i const zeros_vec = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping it's sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8_vec = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8_vec = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i a_i16_low_vec = _mm256_unpacklo_epi8(a_u8_vec, zeros_vec);
        __m256i a_i16_high_vec = _mm256_unpackhi_epi8(a_u8_vec, zeros_vec);
        __m256i b_i16_low_vec = _mm256_unpacklo_epi8(b_u8_vec, zeros_vec);
        __m256i b_i16_high_vec = _mm256_unpackhi_epi8(b_u8_vec, zeros_vec);

        // Multiply and accumulate as `int16`, accumulate products as `int32`
        ab_i32_low_vec = _mm256_add_epi32(ab_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, b_i16_low_vec));
        ab_i32_high_vec = _mm256_add_epi32(ab_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, b_i16_high_vec));
        a2_i32_low_vec = _mm256_add_epi32(a2_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, a_i16_low_vec));
        a2_i32_high_vec = _mm256_add_epi32(a2_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, a_i16_high_vec));
        b2_i32_low_vec = _mm256_add_epi32(b2_i32_low_vec, _mm256_madd_epi16(b_i16_low_vec, b_i16_low_vec));
        b2_i32_high_vec = _mm256_add_epi32(b2_i32_high_vec, _mm256_madd_epi16(b_i16_high_vec, b_i16_high_vec));
    }

    // Further reduce to a single sum for each vector
    int ab = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    int a2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(a2_i32_low_vec, a2_i32_high_vec));
    int b2 = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(b2_i32_low_vec, b2_i32_high_vec));

    // Take care of the tail:
    for (; i < n; ++i) {
        int ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

// 针对f32
SIMSIMD_PUBLIC void simsimd_l2_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    simsimd_l2sq_f32_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {

    __m256 d2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }

    simsimd_f64_t d2 = _simsimd_reduce_f32x8_haswell(d2_vec);
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        d2 += d * d;
    }

    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {

    __m256 ab_vec = _mm256_setzero_ps();
    __m256 a2_vec = _mm256_setzero_ps();
    __m256 b2_vec = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    simsimd_f64_t ab = _simsimd_reduce_f32x8_haswell(ab_vec);
    simsimd_f64_t a2 = _simsimd_reduce_f32x8_haswell(a2_vec);
    simsimd_f64_t b2 = _simsimd_reduce_f32x8_haswell(b2_vec);
    for (; i < n; ++i) {
        float ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }
    *result = _simsimd_cos_normalize_f64_haswell(ab, a2, b2);
}

// 如下是针对f64
SIMSIMD_PUBLIC void simsimd_l2_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    simsimd_l2sq_f64_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f64_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {

    __m256d d2_vec = _mm256_setzero_pd();
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d d_vec = _mm256_sub_pd(a_vec, b_vec);
        d2_vec = _mm256_fmadd_pd(d_vec, d_vec, d2_vec);
    }

    simsimd_f64_t d2 = _simsimd_reduce_f64x4_haswell(d2_vec);
    for (; i < n; ++i) {
        simsimd_f64_t d = a[i] - b[i];
        d2 += d * d;
    }

    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_cos_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {

    __m256d ab_vec = _mm256_setzero_pd();
    __m256d a2_vec = _mm256_setzero_pd();
    __m256d b2_vec = _mm256_setzero_pd();
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        ab_vec = _mm256_fmadd_pd(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_pd(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_pd(b_vec, b_vec, b2_vec);
    }

    simsimd_f64_t ab = _simsimd_reduce_f64x4_haswell(ab_vec);
    simsimd_f64_t a2 = _simsimd_reduce_f64x4_haswell(a2_vec);
    simsimd_f64_t b2 = _simsimd_reduce_f64x4_haswell(b2_vec);
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }
    *result = _simsimd_cos_normalize_f64_haswell(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512bw", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512bw,avx512vl,bmi2"))), apply_to = function)

// 如下是针对skylake的指令
SIMSIMD_PUBLIC void simsimd_l2_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    simsimd_l2sq_f32_skylake(a, b, n, result);
    *result = _simsimd_sqrt_f64_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    // 初始化
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

    // 512 = f32 * 16
simsimd_l2sq_f32_skylake_cycle:
    // 如果不够16个
    if (n < 16) {
        // _bzhi_u32相当于copy x, 然后n以上的位reset为0, 这里制作掩码
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        // 使用掩码load, 非掩码位归0
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        // 满足16个, 直接load
        // update a, b, n
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    // sub --> fma
    __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
    d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_f32_skylake_cycle;

    // reduce --> result
    *result = _simsimd_reduce_f32x16_skylake(d2_vec);
}

SIMSIMD_INTERNAL simsimd_distance_t _simsimd_cos_normalize_f64_skylake(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                       simsimd_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0)
        return 1;

    // We want to avoid the `simsimd_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The maximum relative error for this approximation is less than 2^-14, which is 6x lower than
    // for single-precision floats in the `_simsimd_cos_normalize_f64_haswell` implementation.
    // Mysteriously, MSVC has no `_mm_rsqrt14_pd` intrinsic, but has it's masked variants,
    // so let's use `_mm_maskz_rsqrt14_pd(0xFF, ...)` instead.
    __m128d squares = _mm_set_pd(a2, b2);
    __m128d rsqrts = _mm_maskz_rsqrt14_pd(0xFF, squares);

    // Let's implement a single Newton-Raphson iteration to refine the result.
    // This is how it affects downstream applications:
    //
    // +--------+------+----------+---------------------+---------------------+---------------------+
    // | Metric | NDim |  DType   |   Baseline Error    |  Old SimSIMD Error  |  New SimSIMD Error  |
    // +--------+------+----------+---------------------+---------------------+---------------------+
    // | cosine | 1536 | bfloat16 | 1.89e-08 ± 1.59e-08 | 3.07e-07 ± 3.09e-07 | 3.53e-09 ± 2.70e-09 |
    // | cosine | 1536 | float16  | 1.67e-02 ± 1.44e-02 | 2.68e-05 ± 1.95e-05 | 2.02e-05 ± 1.39e-05 |
    // | cosine | 1536 | float32  | 2.21e-08 ± 1.65e-08 | 3.47e-07 ± 3.49e-07 | 3.77e-09 ± 2.84e-09 |
    // | cosine | 1536 | float64  | 0.00e+00 ± 0.00e+00 | 3.80e-07 ± 4.50e-07 | 1.35e-11 ± 1.85e-11 |
    // | cosine | 1536 |   int8   | 0.00e+00 ± 0.00e+00 | 4.60e-04 ± 3.36e-04 | 4.20e-04 ± 4.88e-04 |
    // +--------+------+----------+---------------------+---------------------+---------------------+
    //
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    simsimd_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    simsimd_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    simsimd_distance_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

// 计算f32的cos
SIMSIMD_PUBLIC void simsimd_cos_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m512 ab_vec = _mm512_setzero();
    __m512 a2_vec = _mm512_setzero();
    __m512 b2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

simsimd_cos_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
    a2_vec = _mm512_fmadd_ps(a_vec, a_vec, a2_vec);
    b2_vec = _mm512_fmadd_ps(b_vec, b_vec, b2_vec);
    if (n) goto simsimd_cos_f32_skylake_cycle;

    simsimd_f64_t ab = _simsimd_reduce_f32x16_skylake(ab_vec);
    simsimd_f64_t a2 = _simsimd_reduce_f32x16_skylake(a2_vec);
    simsimd_f64_t b2 = _simsimd_reduce_f32x16_skylake(b2_vec);
    *result = _simsimd_cos_normalize_f64_skylake(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    simsimd_l2sq_f64_skylake(a, b, n, result);
    *result = _simsimd_sqrt_f64_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    __m512d d2_vec = _mm512_setzero_pd();
    __m512d a_vec, b_vec;

simsimd_l2sq_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d d_vec = _mm512_sub_pd(a_vec, b_vec);
    d2_vec = _mm512_fmadd_pd(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_f64_skylake_cycle;

    // f64可以直接reduce!
    *result = _mm512_reduce_add_pd(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m512d ab_vec = _mm512_setzero_pd();
    __m512d a2_vec = _mm512_setzero_pd();
    __m512d b2_vec = _mm512_setzero_pd();
    __m512d a_vec, b_vec;

simsimd_cos_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    ab_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_vec);
    a2_vec = _mm512_fmadd_pd(a_vec, a_vec, a2_vec);
    b2_vec = _mm512_fmadd_pd(b_vec, b_vec, b2_vec);
    if (n) goto simsimd_cos_f64_skylake_cycle;

    simsimd_f64_t ab = _mm512_reduce_add_pd(ab_vec);
    simsimd_f64_t a2 = _mm512_reduce_add_pd(a2_vec);
    simsimd_f64_t b2 = _mm512_reduce_add_pd(b2_vec);
    *result = _simsimd_cos_normalize_f64_skylake(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

SIMSIMD_INTERNAL __m512i _simsimd_substract_bf16x32_genoa(__m512i a_i16, __m512i b_i16) {

    union {
        __m512 fvec;
        __m512i ivec;
        simsimd_f32_t f32[16];
        simsimd_u16_t u16[32];
        simsimd_bf16_t bf16[32];
    } d_odd, d_even, d, a_f32_even, b_f32_even, d_f32_even, a_f32_odd, b_f32_odd, d_f32_odd, a, b;
    a.ivec = a_i16;
    b.ivec = b_i16;

    // There are several approaches to perform subtraction in `bf16`. The first one is:
    //
    //      Perform a couple of casts - each is a bitshift. To convert `bf16` to `f32`,
    //      expand it to 32-bit integers, then shift the bits by 16 to the left.
    //      Then subtract as floats, and shift back. During expansion, we will double the space,
    //      and should use separate registers for top and bottom halves.
    //      Some compilers don't have `_mm512_extracti32x8_epi32`, so we use `_mm512_extracti64x4_epi64`:
    //
    //          a_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_i16)), 16));
    //          b_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_i16)), 16));
    //          a_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_i16, 1)), 16));
    //          b_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_i16, 1)), 16));
    //          d_f32_top.fvec = _mm512_sub_ps(a_f32_top.fvec, b_f32_top.fvec);
    //          d_f32_bot.fvec = _mm512_sub_ps(a_f32_bot.fvec, b_f32_bot.fvec);
    //          d.ivec = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_bot.fvec), 16)));
    //          d.ivec = _mm512_inserti64x4(d.ivec, _mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_top.fvec), 16)), 1);
    //
    // Instead of using multple shifts and an insertion, we can achieve similar result with fewer expensive
    // calls to `_mm512_permutex2var_epi16`, or a cheap `_mm512_mask_shuffle_epi8` and blend:
    //
    a_f32_odd.ivec = _mm512_and_si512(a_i16, _mm512_set1_epi32(0xFFFF0000));
    a_f32_even.ivec = _mm512_slli_epi32(a_i16, 16);
    b_f32_odd.ivec = _mm512_and_si512(b_i16, _mm512_set1_epi32(0xFFFF0000));
    b_f32_even.ivec = _mm512_slli_epi32(b_i16, 16);

    d_f32_odd.fvec = _mm512_sub_ps(a_f32_odd.fvec, b_f32_odd.fvec);
    d_f32_even.fvec = _mm512_sub_ps(a_f32_even.fvec, b_f32_even.fvec);

    d_f32_even.ivec = _mm512_srli_epi32(d_f32_even.ivec, 16);
    d.ivec = _mm512_mask_blend_epi16(0x55555555, d_f32_odd.ivec, d_f32_even.ivec);

    return d.ivec;
}

SIMSIMD_PUBLIC void simsimd_l2_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {
    simsimd_l2sq_bf16_genoa(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m512 d2_vec = _mm512_setzero_ps();
    __m512i a_i16_vec, b_i16_vec, d_i16_vec;

simsimd_l2sq_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    d_i16_vec = _simsimd_substract_bf16x32_genoa(a_i16_vec, b_i16_vec);
    d2_vec = _mm512_dpbf16_ps(d2_vec, (__m512bh)(d_i16_vec), (__m512bh)(d_i16_vec));
    if (n) goto simsimd_l2sq_bf16_genoa_cycle;

    *result = _simsimd_reduce_f32x16_skylake(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                           simsimd_distance_t *result) {
    __m512 ab_vec = _mm512_setzero_ps();
    __m512 a2_vec = _mm512_setzero_ps();
    __m512 b2_vec = _mm512_setzero_ps();
    __m512i a_i16_vec, b_i16_vec;

simsimd_cos_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    ab_vec = _mm512_dpbf16_ps(ab_vec, (__m512bh)(a_i16_vec), (__m512bh)(b_i16_vec));
    a2_vec = _mm512_dpbf16_ps(a2_vec, (__m512bh)(a_i16_vec), (__m512bh)(a_i16_vec));
    b2_vec = _mm512_dpbf16_ps(b2_vec, (__m512bh)(b_i16_vec), (__m512bh)(b_i16_vec));
    if (n) goto simsimd_cos_bf16_genoa_cycle;

    simsimd_f32_t ab = _simsimd_reduce_f32x16_skylake(ab_vec);
    simsimd_f32_t a2 = _simsimd_reduce_f32x16_skylake(a2_vec);
    simsimd_f32_t b2 = _simsimd_reduce_f32x16_skylake(b2_vec);
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_l2sq_f16_sapphire(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}

// 针对f16的乘法, 也就是说sapphire架构原生支持f16的运算, 相当于 -, fma都有
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                              simsimd_distance_t *result) {
    __m512h d2_vec = _mm512_setzero_ph();
    __m512i a_i16_vec, b_i16_vec;

simsimd_l2sq_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h d_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec));
    d2_vec = _mm512_fmadd_ph(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(d2_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    __m512h ab_vec = _mm512_setzero_ph();
    __m512h a2_vec = _mm512_setzero_ph();
    __m512h b2_vec = _mm512_setzero_ph();
    __m512i a_i16_vec, b_i16_vec;

simsimd_cos_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec), ab_vec);
    a2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(a_i16_vec), a2_vec);
    b2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(b_i16_vec), _mm512_castsi512_ph(b_i16_vec), b2_vec);
    if (n) goto simsimd_cos_f16_sapphire_cycle;

    simsimd_f32_t ab = _mm512_reduce_add_ph(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ph(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ph(b2_vec);
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result) {
    simsimd_l2sq_i8_ice(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    __m512i d2_i32_vec = _mm512_setzero_si512();
    __m512i a_i16_vec, b_i16_vec, d_i16s_vec;

simsimd_l2sq_i8_ice_cycle:
    if (n < 32) { // TODO: Avoid eaarly i16 upcast to step through 64 values at a time
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }
    d_i16s_vec = _mm512_sub_epi16(a_i16_vec, b_i16_vec);
    // multile + pair add --> i32
    d2_i32_vec = _mm512_dpwssd_epi32(d2_i32_vec, d_i16s_vec, d_i16s_vec);
    if (n) goto simsimd_l2sq_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(d2_i32_vec);
}

SIMSIMD_PUBLIC void simsimd_cos_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {

    __m512i ab_i32_vec = _mm512_setzero_si512();
    __m512i a2_i32_vec = _mm512_setzero_si512();
    __m512i b2_i32_vec = _mm512_setzero_si512();
    __m512i a_i16_vec, b_i16_vec;
simsimd_cos_i8_ice_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }

    // We can't directly use the `_mm512_dpbusd_epi32` intrinsic everywhere,
    // as it's asymmetric with respect to the sign of the input arguments:
    //
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    //
    // To compute the squares, we could just drop the sign bit of the second argument.
    // But this would lead to big-big problems on values like `-128`!
    // For dot-products we don't have the luxury of optimizing the sign bit away.
    // Assuming this is an approximate kernel (with reciprocal square root approximations)
    // in the end, we can allow clamping the value to [-127, 127] range.
    //
    // On Ice Lake:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can only execute on port 0, with 5 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also only execute on port 0, with 5 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 5, with 5 cycle latency.
    //
    // On Zen4 Genoa:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 4 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also execute on ports 0 and 1, with 4 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 3 cycle latency.
    //
    // The old solution was complex replied on 1. and 2.:
    //
    //    a_i8_abs_vec = _mm512_abs_epi8(a_i8_vec);
    //    b_i8_abs_vec = _mm512_abs_epi8(b_i8_vec);
    //    a2_i32_vec = _mm512_dpbusds_epi32(a2_i32_vec, a_i8_abs_vec, a_i8_abs_vec);
    //    b2_i32_vec = _mm512_dpbusds_epi32(b2_i32_vec, b_i8_abs_vec, b_i8_abs_vec);
    //    ab_i32_low_vec = _mm512_dpwssds_epi32(                      //
    //        ab_i32_low_vec,                                         //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)), //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_vec)));
    //    ab_i32_high_vec = _mm512_dpwssds_epi32(                           //
    //        ab_i32_high_vec,                                              //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)), //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_vec, 1)));
    //
    // The new solution is simpler and relies on 3.:
    // 处理内积相关操作
    ab_i32_vec = _mm512_add_epi32(ab_i32_vec, _mm512_madd_epi16(a_i16_vec, b_i16_vec));
    a2_i32_vec = _mm512_add_epi32(a2_i32_vec, _mm512_madd_epi16(a_i16_vec, a_i16_vec));
    b2_i32_vec = _mm512_add_epi32(b2_i32_vec, _mm512_madd_epi16(b_i16_vec, b_i16_vec));
    if (n) goto simsimd_cos_i8_ice_cycle;

    int ab = _mm512_reduce_add_epi32(ab_i32_vec);
    int a2 = _mm512_reduce_add_epi32(a2_i32_vec);
    int b2 = _mm512_reduce_add_epi32(b2_i32_vec);
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}
SIMSIMD_PUBLIC void simsimd_l2_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result) {
    simsimd_l2sq_u8_ice(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_distance_t *result) {
    __m512i d2_i32_low_vec = _mm512_setzero_si512();
    __m512i d2_i32_high_vec = _mm512_setzero_si512();
    __m512i const zeros_vec = _mm512_setzero_si512();
    __m512i d_i16_low_vec, d_i16_high_vec;
    __m512i a_u8_vec, b_u8_vec, d_u8_vec;

simsimd_l2sq_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_si512(a);
        b_u8_vec = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
    d_u8_vec = _mm512_or_si512(_mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec));
    // 通过填充0变成i16
    d_i16_low_vec = _mm512_unpacklo_epi8(d_u8_vec, zeros_vec);
    d_i16_high_vec = _mm512_unpackhi_epi8(d_u8_vec, zeros_vec);

    // Multiply and accumulate at `int16` level, accumulate at `int32` level:
    d2_i32_low_vec = _mm512_dpwssd_epi32(d2_i32_low_vec, d_i16_low_vec, d_i16_low_vec);
    d2_i32_high_vec = _mm512_dpwssd_epi32(d2_i32_high_vec, d_i16_high_vec, d_i16_high_vec);
    if (n) goto simsimd_l2sq_u8_ice_cycle;

    *result = _mm512_reduce_add_epi32(_mm512_add_epi32(d2_i32_low_vec, d2_i32_high_vec));
}

SIMSIMD_PUBLIC void simsimd_cos_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *result) {

    __m512i ab_i32_low_vec = _mm512_setzero_si512();
    __m512i ab_i32_high_vec = _mm512_setzero_si512();
    __m512i a2_i32_low_vec = _mm512_setzero_si512();
    __m512i a2_i32_high_vec = _mm512_setzero_si512();
    __m512i b2_i32_low_vec = _mm512_setzero_si512();
    __m512i b2_i32_high_vec = _mm512_setzero_si512();
    __m512i const zeros_vec = _mm512_setzero_si512();
    __m512i a_i16_low_vec, a_i16_high_vec, b_i16_low_vec, b_i16_high_vec;
    __m512i a_u8_vec, b_u8_vec;

simsimd_cos_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_si512(a);
        b_u8_vec = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_i16_low_vec = _mm512_unpacklo_epi8(a_u8_vec, zeros_vec);
    a_i16_high_vec = _mm512_unpackhi_epi8(a_u8_vec, zeros_vec);
    b_i16_low_vec = _mm512_unpacklo_epi8(b_u8_vec, zeros_vec);
    b_i16_high_vec = _mm512_unpackhi_epi8(b_u8_vec, zeros_vec);

    // Multiply and accumulate as `int16`, accumulate products as `int32`:
    ab_i32_low_vec = _mm512_dpwssds_epi32(ab_i32_low_vec, a_i16_low_vec, b_i16_low_vec);
    ab_i32_high_vec = _mm512_dpwssds_epi32(ab_i32_high_vec, a_i16_high_vec, b_i16_high_vec);
    a2_i32_low_vec = _mm512_dpwssds_epi32(a2_i32_low_vec, a_i16_low_vec, a_i16_low_vec);
    a2_i32_high_vec = _mm512_dpwssds_epi32(a2_i32_high_vec, a_i16_high_vec, a_i16_high_vec);
    b2_i32_low_vec = _mm512_dpwssds_epi32(b2_i32_low_vec, b_i16_low_vec, b_i16_low_vec);
    b2_i32_high_vec = _mm512_dpwssds_epi32(b2_i32_high_vec, b_i16_high_vec, b_i16_high_vec);
    if (n) goto simsimd_cos_u8_ice_cycle;

    int ab = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    int a2 = _mm512_reduce_add_epi32(_mm512_add_epi32(a2_i32_low_vec, a2_i32_high_vec));
    int b2 = _mm512_reduce_add_epi32(_mm512_add_epi32(b2_i32_low_vec, b2_i32_high_vec));
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

// 针对i4的处理
SIMSIMD_PUBLIC void simsimd_l2_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                        simsimd_distance_t *result) {
    simsimd_l2sq_i4x2_ice(a, b, n_words, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                          simsimd_distance_t *result) {

    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7] --> [0, 1, 2, 3, 4, --> 15].
    // The absolute difference between two 4-bit integers is at most 15 and it is always a `uint4_t` value!
    // Moreover, it's square is at most 225, which fits into `uint8_t` and can be computed with a single
    // lookup table. Accumulating those values is similar to checksumming, a piece of cake for SIMD!
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const u4_squares_lookup_vec = _mm512_set_epi8(                                        //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    /// 这里相当于是低位的掩码, 每一个i8的低位掩码
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i d_u8_low_vec, d_u8_high_vec; //! Only the low 4 bits are actually used
    __m512i d2_u8_low_vec, d2_u8_high_vec;
    __m512i d2_u16_low_vec, d2_u16_high_vec;

    // Accumulators:
    __m512i d2_u32_vec = _mm512_setzero_si512();

    // 这里读取等操作都是按照i8, 但是运算是按照i4
simsimd_l2sq_i4x2_ice_cycle:
    if (n_words < 64) {
        // 载入剩余的点
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // 这里相当于是a的低4位
    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec); // 每个i8的低位
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec); // 每个i8的高位, 这里相当于错开4位就可以了

    // 这里又相当于是b的低4位
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);

    // 这里相当于将: i4 --> i8, 这里a_i8_low_vec相当于是一个index
    // 这里来细看一下shffle_epi8: a, b
    // a -> [128] [128] [128] [128]
    // b -> [4|4] * 16  (4bit) --> 16 * 8bit
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec); // i4转换为i8的格式
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // We can implement subtraction with a lookup table, or using `_mm512_sub_epi8`.
    // 减法之后计算绝对值
    d_u8_low_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_low_vec, b_i8_low_vec));
    d_u8_high_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_high_vec, b_i8_high_vec));

    // Now we can use the lookup table to compute the squares of the 4-bit unsigned integers
    // in the low nibbles of the `d_u8_low_vec` and `d_u8_high_vec` vectors.
    // 使用平方的lookup table
    // 总之这个shuffle很有意思, 因为是i8中的low 4位 --> 所以只能只能选0-16个i8 --> 对应最大就是128bit
    // 那么就是128bit --> 128bit, 这里相当于实在同一个lane中进行选择
    d2_u8_low_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_low_vec);
    d2_u8_high_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_high_vec);

    // 计算平方之后, 后面需要加法, 加法如果是8bit将会溢出, 因此这里upcast到16bit
    // Aggregating into 16-bit integers, we need to first upcast our 8-bit values to 16 bits.
    // After that, we will perform one more operation, upcasting further into 32-bit integers.
    d2_u16_low_vec =      //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_low_vec, _mm512_setzero_si512()), // 取128的低64
            _mm512_unpackhi_epi8(d2_u8_low_vec, _mm512_setzero_si512()));
    d2_u16_high_vec =     //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_high_vec, _mm512_setzero_si512()), // 取128的高64
            _mm512_unpackhi_epi8(d2_u8_high_vec, _mm512_setzero_si512()));
    
    // 这里是upcast到32位进行计算
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_low_vec, _mm512_setzero_si512()));
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_high_vec, _mm512_setzero_si512()));
    if (n_words) goto simsimd_l2sq_i4x2_ice_cycle;

    // Finally, we can reduce the 16-bit integers to 32-bit integers and sum them up.
    // 进行32的reduce
    int d2 = _mm512_reduce_add_epi32(d2_u32_vec);
    *result = d2;
}
SIMSIMD_PUBLIC void simsimd_cos_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                         simsimd_distance_t *result) {

    // We need to compose a lookup table for all the scalar products of 4-bit integers.
    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // Practically speaking, the product of two 4-bit signed integers is a 7-bit integer,
    // as the maximum absolute value of the product is `abs(-8 * -8) == 64`.
    //
    // To store 128 possible values of 2^7 bits we only need 128 single-byte scalars,
    // or just 2x ZMM registers. In that case our lookup will only take `vpermi2b` instruction,
    // easily inokable with `_mm512_permutex2var_epi8` intrinsic with latency of 6 on Sapphire Rapids.
    // The problem is converting 2d indices of our symmetric matrix into 1d offsets in the dense array.
    //
    // Alternatively, we can take the entire symmetric (16 x 16) matrix of products,
    // put into 4x ZMM registers, and use it with `_mm512_shuffle_epi8`, remembering
    // that it can only lookup with 128-bit lanes (16x 8-bit values).
    // That intrinsic has latency 1, but will need to be repeated and combined with
    // multiple iterations of `_mm512_shuffle_i64x2` that has latency 3.
    //
    // Altenatively, we can get down to 3 cycles per lookup with `vpermb` and `_mm512_permutexvar_epi8` intrinsics.
    // For that we can split our (16 x 16) matrix into 4x (8 x 8) submatrices, and use 4x ZMM registers.
    //
    // Still, all of those solutions are quite heavy compared to two parallel calls to `_mm512_dpbusds_epi32`
    // for the dot product. But we can still use the `_mm512_permutexvar_epi8` to compute the squares of the
    // 16 possible `int4_t` values faster.
    //
    // Here is how our `int4_t` range looks:
    //
    //      dec:     0   1   2   3   4   5   6   7  -8  -7  -6  -5  -4  -3  -2  -1
    //      hex:     0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    //
    // Squared:
    //
    //      dec2:    0   1   4   9  16  25  36  49  64  49  36  25  16   9   4   1
    //      hex2:    0   1   4   9  10  19  24  31  40  31  24  19  10   9   4   1
    //
    // Broadcast it to every lane, so that: `square(x) == _mm512_shuffle_epi8(i4_squares_lookup_vec, x)`.
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const i4_squares_lookup_vec = _mm512_set_epi8(       //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i a2_u8_vec, b2_u8_vec;

    // Accumulators:
    __m512i a2_u16_low_vec = _mm512_setzero_si512();
    __m512i a2_u16_high_vec = _mm512_setzero_si512();
    __m512i b2_u16_low_vec = _mm512_setzero_si512();
    __m512i b2_u16_high_vec = _mm512_setzero_si512();
    __m512i ab_i32_low_vec = _mm512_setzero_si512();
    __m512i ab_i32_high_vec = _mm512_setzero_si512();

simsimd_cos_i4x2_ice_cycle:
    if (n_words < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    // For now, they are not really 8-bit integers, as they are not sign-extended.
    // That part will come later, using the `i4_to_i8_lookup_vec` lookup.

    // 低4位
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec);
    // 高4位
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec);
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);

    // Compute the squares of the 4-bit integers.
    // For symmetry we could have used 4 registers, aka "a2_i8_low_vec", "a2_i8_high_vec", "b2_i8_low_vec",
    // "b2_i8_high_vec". But the largest square value is just 64, so we can safely aggregate into 8-bit unsigned values.

    // 累加平法
    a2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_high_vec));
    b2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_high_vec));

    // We can safely aggregate into just 16-bit sums without overflow, if the vectors have less than:
    //      (2 scalars / byte) * (64 bytes / register) * (256 non-overflowing 8-bit additions in 16-bit intesgers)
    //      = 32'768 dimensions.
    //
    // We use saturated addition here to clearly inform in case of overflow.
    a2_u16_low_vec = _mm512_adds_epu16(a2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    a2_u16_high_vec = _mm512_adds_epu16(a2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));
    b2_u16_low_vec = _mm512_adds_epu16(b2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    b2_u16_high_vec = _mm512_adds_epu16(b2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));

    // Time to perform the proper sign extension of the 4-bit integers to 8-bit integers.
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec);
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // The same trick won't work for the primary dot-product, as the signs vector
    // components may differ significantly. So we have to use two `_mm512_dpwssds_epi32`
    // intrinsics instead, upcasting four chunks to 16-bit integers beforehand!
    // Alternatively, we can flip the signs of the second argument and use `_mm512_dpbusds_epi32`,
    // but it ends up taking more instructions.
    ab_i32_low_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_low_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_low_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_low_vec)));
    ab_i32_low_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_low_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_low_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_low_vec, 1)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_high_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_high_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_high_vec)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_high_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_high_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_high_vec, 1)));
    if (n_words) goto simsimd_cos_i4x2_ice_cycle;

    int ab = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    unsigned short a2_u16[32], b2_u16[32];
    _mm512_storeu_si512(a2_u16, _mm512_add_epi16(a2_u16_low_vec, a2_u16_high_vec));
    _mm512_storeu_si512(b2_u16, _mm512_add_epi16(b2_u16_low_vec, b2_u16_high_vec));
    unsigned int a2 = 0, b2 = 0;
    for (int i = 0; i < 32; ++i) a2 += a2_u16[i], b2 += b2_u16[i];
    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_cos_i8_sierra(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                          simsimd_distance_t *result) {

    __m256i ab_i32_vec = _mm256_setzero_si256();
    __m256i a2_i32_vec = _mm256_setzero_si256();
    __m256i b2_i32_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b + i));
        ab_i32_vec = _mm256_dpbssds_epi32(ab_i32_vec, a_i8_vec, b_i8_vec);
        a2_i32_vec = _mm256_dpbssds_epi32(a2_i32_vec, a_i8_vec, a_i8_vec);
        b2_i32_vec = _mm256_dpbssds_epi32(b2_i32_vec, b_i8_vec, b_i8_vec);
    }

    // Further reduce to a single sum for each vector
    int ab = _simsimd_reduce_i32x8_haswell(ab_i32_vec);
    int a2 = _simsimd_reduce_i32x8_haswell(a2_i32_vec);
    int b2 = _simsimd_reduce_i32x8_haswell(b2_i32_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        int ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_cos_normalize_f32_haswell(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SIERRA
#endif // _SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
