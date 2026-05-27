# Building bitsandbytes for Windows ARM64

This document describes the changes made and the steps required to build a Windows ARM64 (AArch64) wheel for bitsandbytes.

## Overview

bitsandbytes on Windows ARM64 builds as a **CPU-only** backend since CUDA/HIP/XPU are not available for this platform. The native C++ code contains AVX512/AVX2 SIMD optimizations that are compile-time guarded (`#if defined(__AVX512F__)`) and automatically excluded on ARM64, falling back to portable scalar implementations.

---

## Changes Made

### `CMakeLists.txt` — Fix MSVC ARM64 Compilation

**Problem:** The original code unconditionally set `/arch:AVX2` for all MSVC builds:

```cmake
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
endif()
```

`/arch:AVX2` is an x86/x64-only compiler flag and causes a build failure on ARM64 MSVC.

**Fix:** Made the flag conditional on the target architecture:

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

---

## Prerequisites

1. **Windows 11 ARM64** machine (e.g., Snapdragon X Elite, Microsoft SQ series)
2. **ARM64 Python** (e.g., Python 3.12 ARM64 from python.org)
3. **Visual Studio 2022+** with the following workloads/components:
   - "Desktop development with C++"
   - "MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools"
4. **CMake** (3.22.1 or later; can be installed via pip)

---

## Build Steps

### 1. Create a virtual environment

> **Important:** Do NOT name the venv starting with `bitsandbytes` — `setup.py` uses `find_packages()` which would incorrectly include the venv directory in the wheel.

```powershell
python -m venv bnb-winarm
```

If `ensurepip` hangs, create without pip and bootstrap separately:

```powershell
python -m venv --without-pip bnb-winarm
bnb-winarm\Scripts\python.exe -m ensurepip --upgrade
```

### 2. Install build dependencies

```powershell
bnb-winarm\Scripts\python.exe -m pip install setuptools wheel scikit-build-core cmake build "trove-classifiers>=2025.8.6.13"
```

### 3. Configure CMake (CPU backend)

```powershell
bnb-winarm\Scripts\cmake.exe -B build -DCOMPUTE_BACKEND=cpu -S .
```

Expected output should show:
- Compiler: `Hostarm64/arm64/cl.exe`
- Backend: `cpu`
- OpenMP found (optional, for multithreaded performance)

### 4. Build the native library

```powershell
bnb-winarm\Scripts\cmake.exe --build build --config Release
```

This produces `bitsandbytes/libbitsandbytes_cpu.dll` (ARM64 native).

### 5. Build the Python wheel

Set `BNB_SKIP_CMAKE=1` to skip the CMake step during wheel packaging (since we already built the DLL):

```powershell
$env:BNB_SKIP_CMAKE="1"
bnb-winarm\Scripts\python.exe -m build --wheel --no-isolation
```

### 6. Verify the output

The wheel will be in the `dist/` directory:

```powershell
dir dist\*.whl
```

Expected output:
```
bitsandbytes-0.50.0.dev0-cp312-cp312-win_arm64.whl
```

Verify it contains the DLL:

```powershell
bnb-winarm\Scripts\python.exe -c "import zipfile; z=zipfile.ZipFile('dist/bitsandbytes-0.50.0.dev0-cp312-cp312-win_arm64.whl'); [print(f.filename, f.file_size) for f in z.infolist() if f.filename.endswith('.dll')]"
```

Expected:
```
bitsandbytes/libbitsandbytes_cpu.dll 25088
```

---

## Installing the wheel

```powershell
pip install dist\bitsandbytes-0.50.0.dev0-cp312-cp312-win_arm64.whl
```

---

## Running Tests

### Required: Visual Studio Environment Variables

`torch.compile` (Inductor backend) compiles C++ code at runtime using `cl.exe`. For this to work on Windows ARM64, you must set the MSVC include and library paths. Without these, `torch.compile` tests will fail with errors like `Cannot open include file: 'omp.h'`.

Set these environment variables before running tests (adjust paths to match your VS installation):

```powershell
# MSVC and Windows SDK versions (adjust to your installation)
$msvcVer = "14.51.36231"
$sdkVer = "10.0.28000.0"

# Include paths
$msvcInc = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\$msvcVer\include"
$ucrtInc = "C:\Program Files (x86)\Windows Kits\10\Include\$sdkVer\ucrt"
$sharedInc = "C:\Program Files (x86)\Windows Kits\10\Include\$sdkVer\shared"
$umInc = "C:\Program Files (x86)\Windows Kits\10\Include\$sdkVer\um"
$env:INCLUDE = "$msvcInc;$ucrtInc;$sharedInc;$umInc"

# Library paths (ARM64)
$msvcLib = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\$msvcVer\lib\arm64"
$ucrtLib = "C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVer\ucrt\arm64"
$umLib = "C:\Program Files (x86)\Windows Kits\10\Lib\$sdkVer\um\arm64"
$env:LIB = "$msvcLib;$ucrtLib;$umLib"
```

**Alternatively**, run tests from a "Developer Command Prompt for VS" or "Developer PowerShell for VS", which sets these variables automatically.

### Running the test suite

```powershell
$env:BNB_TEST_DEVICE = "cpu"
bnb-winarm\Scripts\python.exe -m pytest tests/ -v --tb=short -q
```

### Expected results (with environment properly configured)

```
2663 passed, 1166 skipped, 29 deselected, 30 xfailed
```

All tests pass. Skipped tests are those requiring CUDA/GPU hardware.

---

## Notes

- **Performance:** The ARM64 build uses scalar C++ implementations. There are no ARM NEON SIMD optimizations yet — this is a functional but not performance-optimized build.
- **OpenMP:** If Visual Studio's OpenMP support is detected, multithreading is enabled for quantization/dequantization operations.
- **GPU support:** This build is CPU-only. GPU acceleration (CUDA/ROCm) is not available on Windows ARM64.
- **Python version:** The wheel is tagged for the specific CPython version used to build it (e.g., `cp312`). To support multiple Python versions, repeat the build with each Python version.
- **torch.compile:** Requires MSVC environment variables (`INCLUDE`, `LIB`) to be set for Inductor's runtime C++ compilation. See "Running Tests" section above.
