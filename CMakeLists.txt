cmake_minimum_required(VERSION 3.28)

# 参考实现（reference implementation）为 sm_120a (RTX 5090) 编译。
# zatfung 移植线在此之上扩展了四档架构：
#   120a  Blackwell   RTX 5090            参考实现，warp-specialized NVFP4 W4A4 TMA 内核
#   89    Ada         RTX 4090/4070       本 fork 原始目标（48 KiB static smem cap）
#   86    Ampere      RTX 3090/3060 Ti    本机主力（NINFER_SM86）
#   75    Turing      RTX 2080 Ti         最保守档（无 cp.async / 无 bf16 MMA）
# 非 120 构建会排除 warp-specialized SM120 NVFP4 W4A4 TMA 内核（见 src/CMakeLists.txt，
# 那里用 `MATCHES "^120"` 把 TMA 内核换成 stub，故 75/86/89 自动落到架构通用内核集）。
# 架构必须在 CUDA 编译器探测之前校验，否则 CMake 会注入通用的宿主工具链默认值，
# 或接受一个不受支持的目标。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|89|86|75)$")
  message(FATAL_ERROR
    "zatfung supports CMAKE_CUDA_ARCHITECTURES=120a, 89, 86, or 75; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(zatfung LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 架构宏：内核层一律用这些宏做编译期门控（与 ninfer-2080ti-22g 的 Turing port 保持同名，
# 便于把那边已验证的 SM75/SM86 降级分支直接合并进来）。
#   NINFER_SM75  : Turing，无 cp.async、无 bf16/tf32 MMA、无 PDL、smem/SM 64 KiB
#   NINFER_SM86  : Ampere，有 cp.async/bf16 MMA，无 nvfp4；smem/SM 100 KiB
#   NINFER_SM89  : Ada，同 sm_86 的共享内存上限档位（capped w8 变体），无 nvfp4
#   默认(不定义) : sm_120a 参考实现，保留高共享内存调度
# sm_89/sm_86 均启用 capped-shared-memory 的 w8 内核变体（见 src/ops/linear/w8/w8_config.h）；
# 120a 构建保持其默认的高内存调度。
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^75")
  add_compile_definitions(NINFER_SM75=1)
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^86")
  add_compile_definitions(NINFER_SM86=1)
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^89")
  add_compile_definitions(NINFER_SM89=1)
endif()

if(MSVC)
  add_compile_definitions(NOMINMAX UTF8PROC_STATIC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C>:/FS>
    $<$<COMPILE_LANGUAGE:C>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/FS>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
  )
endif()

# nvcc 会对"比自己发布的 MSVC 更新"的宿主编译器直接报
# "unsupported Microsoft Visual Studio version"（例：CUDA 12.8 遇到 MSVC 19.44）。
# 这类组合多数情况下实际可用，只是 nvcc 的白名单没跟上；逃生舱由构建器在
# 探测到该报错时自动带上（见 builder.go 的 -cuda-allow-unsupported）。
option(NINFER_CUDA_ALLOW_UNSUPPORTED "Pass -allow-unsupported-compiler to nvcc" OFF)
if(NINFER_CUDA_ALLOW_UNSUPPORTED)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

# CUDA 工具链门槛。
# 上游参考线要求 CUDA >= 13.1（它服务 Blackwell/sm_120a）。但 75/86/89 三档只需要
# CUDA 12.8 即可完整编译，而 12.8 正是本机已装版本 —— 为了不必为了一个策略性版本号
# 去装 6 GB 的新工具链，这里提供逃生舱：
#   - NINFER_ALLOW_LEGACY_CUDA=ON 且目标架构不含 120a 时，放宽到 12.8 并给出警告；
#   - 目标含 120a 时永不放宽（sm_120a 的 TMA/setmaxnreg 需要 12.9+/13.x 工具链）。
# 另设 12.8 为硬下限：再老的 CUDA 连 sm_86 的 C++20 支持都不完整。
option(NINFER_ALLOW_LEGACY_CUDA "Allow CUDA < 13.1 for non-Blackwell targets" OFF)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "zatfung requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  if(NINFER_ALLOW_LEGACY_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^120")
    message(WARNING
      "zatfung: CUDA ${CMAKE_CUDA_COMPILER_VERSION} < 13.1 — 已按 NINFER_ALLOW_LEGACY_CUDA "
      "放宽（目标架构 ${CMAKE_CUDA_ARCHITECTURES}）。该组合不在上游验证范围内；"
      "若遇到 nvcc/MSVC 兼容性报错，请改装 CUDA 13.x 后重试。")
  else()
    message(FATAL_ERROR
      "zatfung requires CUDA 13.1 or newer for this architecture; found CUDA compiler "
      "${CMAKE_CUDA_COMPILER_VERSION}. 若目标是 75/86/89，可加 "
      "-DNINFER_ALLOW_LEGACY_CUDA=ON 放宽到 12.8。")
  endif()
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(MSVC)
  # ffmpeg 随发布包提供（ffmpeg/include + ffmpeg/lib + ffmpeg/bin/*.dll），源码树里默认没有。
  # 若强行定义这个 IMPORTED target，ninfer_media_decode 会去找不存在的 avcodec.lib 而在
  # 链接期炸掉（LNK1104）。所以这里做能力探测：目录在才开媒体，不在就自动降级为纯文本
  # （decode.cpp 全身由 #ifdef NINFER_HAVE_FFMPEG 包住，关掉即可编译）。
  if(EXISTS "${PROJECT_SOURCE_DIR}/ffmpeg/include" AND EXISTS "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
    add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
    target_include_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/include")
    target_link_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
    target_link_libraries(PkgConfig::FFMPEG INTERFACE avcodec avformat avutil swscale swresample)
  else()
    set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
    message(STATUS
      "zatfung: 未找到 ffmpeg/（include+lib），媒体解码已禁用，本次为纯文本构建。"
      "需要图像/视频输入时把 ffmpeg 目录放到源码树根后重新 configure。")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()
find_package(Threads REQUIRED)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
