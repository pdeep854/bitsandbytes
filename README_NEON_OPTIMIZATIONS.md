# ARM NEON Intrinsics Optimizations for Windows ARM64

This document describes the ARM NEON SIMD optimizations added to `bitsandbytes` for Windows ARM64 (AArch64) platforms such as Snapdragon X Elite, Microsoft SQ series, etc.

---

## Overview

The bitsandbytes CPU backend previously relied on x86 AVX512 intrinsics for SIMD acceleration, with a scalar C++ fallback for all other platforms. On ARM64, all operations ran at scalar speed.

These changes add **ARM NEON intrinsics** to accelerate the most performance-critical CPU operations, delivering **2.3× to 6.7× speedup** across all major quantization and inference operations.

---

## Files Modified

| File | Changes |
|------|---------|
| `csrc/cpu_ops.cpp` | Added ~150 lines of ARM NEON optimized code |
| `CMakeLists.txt` | Fixed `/arch:AVX2` to not apply on ARM64 MSVC |

---

## Performance Results

| Operation | Original (scalar) | NEON Optimized | Speedup |
|-----------|-------------------|----------------|---------|
| 8-bit quantize [4096×4096] | 17.17 ms | 2.57 ms | **6.7×** |
| 8-bit dequant [1024×1024] | 0.19 ms | 0.07 ms | **2.7×** |
| NF4 dequant fp32 [1024×1024] | 4.09 ms | 1.42 ms | **2.9×** |
| NF4 dequant fp32 [4096×4096] | 46.50 ms | 20.40 ms | **2.3×** |
| NF4 dequant bf16 [1024×1024] | 4.47 ms | 0.93 ms | **4.8×** |
| NF4 dequant bf16 [4096×4096] | 52.75 ms | 16.16 ms | **3.3×** |
| FP4 dequant fp32 [4096×4096] | 51.05 ms | 21.59 ms | **2.4×** |
| Linear4bit fwd [bs=1, 1024→1024] | 5.77 ms | 1.09 ms | **5.3×** |
| Linear4bit fwd [bs=1, 4096→4096] | 68.43 ms | 19.39 ms | **3.5×** |

All 2663 tests pass with 0 failures.

---

## Detailed Changes in `csrc/cpu_ops.cpp`

All NEON code is guarded by:
```cpp
#if defined(_M_ARM64) || defined(__aarch64__)
#include <arm_neon.h>
// ... NEON code ...
#endif
```

This ensures the code only compiles on ARM64 targets (both MSVC `_M_ARM64` and GCC/Clang `__aarch64__`).

---

### 1. NEON 4-bit NF4/FP4 Dequantization (`neon_dequant_4bit_16values`)

**What it does:** Dequantizes 8 packed bytes (16 × 4-bit values) into 16 float32 outputs in one call.

**How it works:**

```cpp
static inline void neon_dequant_4bit_16values(
    const uint8_t* packed, float scale, const float32x4_t lut[4], float* out
)
```

1. **Load 8 bytes** using `vld1_u8(packed)` — each byte contains two 4-bit quantized values
2. **Extract nibbles:**
   - Low nibble: `vand_u8(raw, vdup_n_u8(0x0F))` — masks bits [3:0]
   - High nibble: `vshr_n_u8(raw, 4)` — shifts bits [7:4] to [3:0]
3. **Interleave:** `vzip_u8(hi_nibbles, lo_nibbles)` — creates the correct output order (high nibble = first value, low nibble = second value for each byte)
4. **LUT lookup:** Uses a flat 16-entry float array on stack. Each 4-bit index (0-15) maps to the corresponding NF4/FP4 dequantized float value
5. **Scale multiply:** `vmulq_f32(vld1q_f32(values), vdupq_n_f32(scale))` — applies the per-block scale factor using NEON vectorized multiply
6. **Store:** `vst1q_f32(out, ...)` — writes 4 floats at a time

**Why it's fast:** Processes 16 values per call (vs 2 per iteration in scalar), with NEON-vectorized scale multiplication eliminating 16 scalar multiplies.

**Guard condition:** Only activates when `dim_1 % 16 == 0 && blocksize >= 16 && dim_1 % blocksize == 0` to ensure correct scale indexing.

---

### 2. NEON NF4/FP4 Lookup Tables (`neon_nf4_lut`, `neon_fp4_lut`)

**What they do:** Initialize the 16-entry float lookup tables as four `float32x4_t` vectors.

```cpp
static inline void neon_nf4_lut(float32x4_t lut[4]) {
    static const float nf4_values[16] = {
        -1.0f, -0.6962f, ..., 0.7230f, 1.0f
    };
    lut[0] = vld1q_f32(nf4_values);      // indices 0-3
    lut[1] = vld1q_f32(nf4_values + 4);  // indices 4-7
    lut[2] = vld1q_f32(nf4_values + 8);  // indices 8-11
    lut[3] = vld1q_f32(nf4_values + 12); // indices 12-15
}
```

**Explanation:** NF4 (Normal Float 4-bit) has exactly 16 possible values derived from a normal distribution quantile function. These are stored in sorted order matching the 4-bit index encoding used by bitsandbytes.

---

### 3. NEON BF16→Float Conversion (`neon_bf16x4_to_f32`)

**What it does:** Converts 4 bfloat16 values to float32 using a single NEON shift instruction.

```cpp
static inline float32x4_t neon_bf16x4_to_f32(const bf16_t* src) {
    uint16x4_t raw = vld1_u16(reinterpret_cast<const uint16_t*>(src));
    uint32x4_t wide = vshll_n_u16(raw, 16);  // shift left by 16 bits
    return vreinterpretq_f32_u32(wide);
}
```

**Explanation:** BF16 is simply the upper 16 bits of a float32. Converting BF16→FP32 is just a left shift by 16 to place the bits in the correct position. `vshll_n_u16` widens uint16 to uint32 while shifting, making this a single-instruction conversion.

---

### 4. NEON Float→BF16 Conversion (`neon_f32_to_bf16x4`)

**What it does:** Converts 4 float32 values to bfloat16 with round-to-nearest-even.

```cpp
static inline void neon_f32_to_bf16x4(const float32x4_t src, bf16_t* dst) {
    uint32x4_t bits = vreinterpretq_u32_f32(src);
    // Round to nearest even: add 0x7FFF + ((bits >> 16) & 1)
    uint32x4_t lsb = vshrq_n_u32(bits, 16);
    lsb = vandq_u32(lsb, vdupq_n_u32(1));
    uint32x4_t rounding = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    bits = vaddq_u32(bits, rounding);
    // Extract upper 16 bits
    uint16x4_t result = vshrn_n_u32(bits, 16);
    vst1_u16(reinterpret_cast<uint16_t*>(dst), result);
}
```

**Explanation:** 
- Extracts the LSB of the BF16 result (bit 16 of the float32) for round-to-nearest-even
- Adds rounding bias (0x7FFF + LSB) to handle the truncated mantissa bits
- Uses `vshrn_n_u32` (shift-right-narrow) to extract the upper 16 bits into a uint16x4

This replaces the scalar bit-manipulation loop that processed one value at a time.

---

### 5. NEON Float→FP16 Conversion (`neon_f32_to_fp16x4`)

**What it does:** Converts 4 float32 values to IEEE float16 using ARM's native hardware instruction.

```cpp
static inline void neon_f32_to_fp16x4(const float32x4_t src, fp16_t* dst) {
    float16x4_t half = vcvt_f16_f32(src);  // Hardware FP16 conversion
    vst1_u16(reinterpret_cast<uint16_t*>(dst), vreinterpret_u16_f16(half));
}
```

**Explanation:** ARM64 has native FP16 support via the `vcvt_f16_f32` instruction, which handles all edge cases (denormals, infinity, NaN, rounding) in hardware. This replaces 30+ lines of scalar bit manipulation per value.

---

### 6. NEON Absmax Computation (`neon_absmax_f32`)

**What it does:** Finds the maximum absolute value in a float32 array using 16-element loop unrolling.

```cpp
static inline float neon_absmax_f32(const float* data, long long n) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    long long i = 0;
    // Process 16 elements per iteration
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(data + i));
        float32x4_t v1 = vabsq_f32(vld1q_f32(data + i + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(data + i + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(data + i + 12));
        vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
    }
    // ... 4-element and scalar remainder handling ...
    float result = vmaxvq_f32(vmax);  // Horizontal max reduction
    return result;
}
```

**Explanation:**
- `vabsq_f32`: Computes absolute value of 4 floats simultaneously
- `vmaxq_f32`: Element-wise max of two 4-float vectors
- 16-element unrolling hides instruction latency and maximizes throughput
- `vmaxvq_f32`: ARM64-specific horizontal max (reduces 4-element vector to single scalar)

**Used in:** `quantize_cpu_impl` for computing per-block absmax during 8-bit blockwise quantization.

---

### 7. NEON Norm-to-LUT-Index (`neon_norm_to_lut_index_x4`)

**What it does:** Converts 4 normalized float values in [-1, 1] to 16-bit LUT indices [0, 65535].

```cpp
static inline uint16x4_t neon_norm_to_lut_index_x4(float32x4_t vals) {
    vals = vmaxq_f32(vals, vdupq_n_f32(-1.0f));  // clamp min
    vals = vminq_f32(vals, vdupq_n_f32(1.0f));   // clamp max
    // (val + 1.0) * 0.5 * 65535 + 0.5
    float32x4_t result = vmlaq_f32(
        vdupq_n_f32(0.5f),                       // accumulator (rounding bias)
        vaddq_f32(vals, vdupq_n_f32(1.0f)),      // (val + 1.0)
        vdupq_n_f32(0.5f * 65535.0f)             // scale factor
    );
    uint32x4_t u32 = vcvtq_u32_f32(result);      // float → uint32
    return vmovn_u32(u32);                         // narrow uint32 → uint16
}
```

**Explanation:**
- `vmlaq_f32`: Fused multiply-add: `acc + a * b` in one instruction
- `vcvtq_u32_f32`: Float-to-integer conversion (truncates toward zero)
- `vmovn_u32`: Narrows 4 × uint32 → 4 × uint16 (takes lower 16 bits)

**Used in:** `quantize_cpu_impl` to vectorize the normalization + LUT index calculation for 8-bit blockwise quantization.

---

## Integration Points

### In `dequantizeBlockwise4bitCpu` (4-bit dequantization):
```cpp
#if defined(_M_ARM64) || defined(__aarch64__)
    if (dim_1 % VEC_LEN == 0 && blocksize >= VEC_LEN && (dim_1 % blocksize == 0)) {
        // NEON path: uses neon_dequant_4bit_16values + neon_f32_to_bf16x4/fp16x4
        ...
        return;
    }
#endif
    // Falls through to scalar fallback if conditions not met
```

### In `quantize_cpu_impl` (8-bit quantization):
```cpp
#if defined(_M_ARM64) || defined(__aarch64__)
    if constexpr (std::is_same<T, float>::value) {
        // NEON absmax
        absmax_block = neon_absmax_f32(...);
        // NEON normalize + LUT index
        for (; i + 4 <= block_len; i += 4) {
            float32x4_t v = vmulq_f32(vld1q_f32(src + i), vinv);
            uint16x4_t indices = neon_norm_to_lut_index_x4(v);
            ...
        }
    }
#endif
```

---

## CMakeLists.txt Change

```cmake
if(MSVC)
    # /arch:AVX2 is only valid for x86/x64 targets, not ARM64
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _msvc_arch)
    if(_msvc_arch MATCHES "x86|x64|amd64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
    endif()
endif()
```

**Explanation:** The original code unconditionally added `/arch:AVX2` for all MSVC builds. This x86-only flag causes a compilation error on ARM64 MSVC. The fix makes it conditional on the target architecture.

---

## Architecture Notes

- **MSVC ARM64** supports NEON intrinsics via `<arm_neon.h>` (automatically available, no special flags needed)
- **All Windows ARM64 chips** (Snapdragon X, SQ series) have full NEON support
- The code uses only **baseline NEON** (Armv8.0) — no optional extensions like BF16 dot-product (`bfdot`) which would require Armv8.6+
- `vcvt_f16_f32` requires the **FP16 extension** which is mandatory in Armv8.0+ (always available)
- OpenMP parallelization (`BNB_OMP_PARALLEL_FOR`) works alongside NEON — each thread processes its own blocks using NEON intrinsics

---

## Test Results

```
Platform: Windows 11 ARM64 (Snapdragon X Elite)
Python: 3.12 ARM64
PyTorch: 2.12.0+cpu (win_arm64)
Tests: 2663 passed, 0 failed, 1166 skipped, 30 xfailed