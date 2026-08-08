# csrc/CMakeLists.txt — builds the wisp._wisp_core extension.
#
# GPU build (default when a CUDA toolkit is found):
#   core C + CUDA kernels + bindings, OpenMP enabled.
# CPU-only build:
#   -DWISP_NO_CUDA=ON  -> same core C sources compile against the
#   host-memory shim in wisp_engine.c; no .cu files, no CUDA dependency.

cmake_minimum_required(VERSION 3.20)

option(WISP_NO_CUDA "Build without CUDA (CPU-only engine)" OFF)
option(WISP_HAS_NCCL "Link NCCL for 3+ GPU pipeline mode" OFF)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11: prefer the pip-installed package's cmake dir
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP_RESULT
)
if(PYBIND11_LOOKUP_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)

if(NOT WISP_NO_CUDA)
    # Target every consumer generation from Turing through Blackwell.
    # sm_120 = RTX 5070/5080/5090 (Blackwell) — supported natively by
    # CUDA 12.8+ / 13.x toolkits. Must be set BEFORE enable_language(CUDA).
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Minimum 12.0; newer toolkits (12.8, 13.3, ...) satisfy this —
        # find_package version args are minimums, never exact pins.
        find_package(CUDAToolkit 12.0 REQUIRED)
    else()
        message(STATUS "No CUDA compiler found — falling back to CPU-only "
                       "build (WISP_NO_CUDA=ON)")
        set(WISP_NO_CUDA ON)
    endif()
endif()

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
set(WISP_CORE_SOURCES
    core/wisp_engine.c
    core/lru_cache.c
    core/double_buffer.c
    core/mem_pool.c
    core/wisp_avx512.c
    core/wisp_async_io.c
)

set(WISP_CUDA_SOURCES
    cuda/expert_loader.cu
    cuda/attention.cu
    cuda/router.cu
    cuda/matmul.cu
    cuda/kv_cache.cu
    cuda/speculative.cu
    cuda/multi_gpu.cu
)

# Утилита конвертации моделей в оптимизированный формат WBIN
add_executable(wisp_convert wisp_convert.c)
target_include_directories(wisp_convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_C_FOUND)
    target_link_libraries(wisp_convert PRIVATE OpenMP::OpenMP_C)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(LIBURING NAMES io_uring)
    if(LIBURING)
        target_link_libraries(wisp_convert PRIVATE ${LIBURING})
    endif()
endif()

# Применяем те же оптимизации компиляции к утилите конвертации
if(NOT MSVC)
    set_target_properties(wisp_convert PROPERTIES
        COMPILE_FLAGS "${GCC_OPT_FLAGS_STR}"
    )
    
    # AVX-512 для конвертации
    if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512VL)
        set_source_files_properties(wisp_convert.c PROPERTIES 
            COMPILE_OPTIONS "${AVX512_FLAGS}")
    endif()
endif()

if(WISP_NO_CUDA)
    pybind11_add_module(_wisp_core
        bindings/wisp_bindings.cpp
        ${WISP_CORE_SOURCES}
    )
    target_compile_definitions(_wisp_core PRIVATE WISP_NO_CUDA)
else()
    pybind11_add_module(_wisp_core
        bindings/wisp_bindings.cpp
        ${WISP_CORE_SOURCES}
        ${WISP_CUDA_SOURCES}
    )
    target_link_libraries(_wisp_core PRIVATE CUDA::cudart)
    set_target_properties(_wisp_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    # NCCL (3+ GPU pipeline collectives). Auto-detected; CMake ships no
    # FindNCCL module, so probe the library + header directly. The 2-GPU
    # strategies use cudaMemcpyPeerAsync and never need NCCL.
    find_library(NCCL_LIBRARY nccl
                 HINTS $ENV{NCCL_ROOT}/lib $ENV{NCCL_HOME}/lib)
    find_path(NCCL_INCLUDE_DIR nccl.h
              HINTS $ENV{NCCL_ROOT}/include $ENV{NCCL_HOME}/include)
    if(WISP_HAS_NCCL OR (NCCL_LIBRARY AND NCCL_INCLUDE_DIR))
        if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
            message(STATUS "NCCL found: ${NCCL_LIBRARY}")
            target_include_directories(_wisp_core PRIVATE ${NCCL_INCLUDE_DIR})
            target_link_libraries(_wisp_core PRIVATE ${NCCL_LIBRARY})
        else()
            target_link_libraries(_wisp_core PRIVATE nccl)
        endif()
        target_compile_definitions(_wisp_core PRIVATE
            WISP_HAS_NCCL WISP_USE_NCCL)
    else()
        message(STATUS "NCCL not found — 3+ GPU pipeline disabled "
                       "(2-GPU peer copies still available)")
    endif()
endif()

target_include_directories(_wisp_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_C_FOUND)
    target_link_libraries(_wisp_core PRIVATE OpenMP::OpenMP_C)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(_wisp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# --------------------------------------------------------------------------
# Compiler flags — MSVC and gcc/clang both supported
# --------------------------------------------------------------------------

if(MSVC)
    target_compile_options(_wisp_core PRIVATE
        $<$<COMPILE_LANGUAGE:C>:/O2 /W3>
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3 /EHsc>
    )
    target_compile_definitions(_wisp_core PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    # Intel Xeon Gold 6348 (Ice Lake) — максимальные оптимизации производительности:
    # -Ofast: агрессивная оптимизация с нарушением strict IEEE (быстрее ~15-20%)
    # -march=icelake-server: точная микроархитектура Ice Lake серверного класса
    # -mtune=icelake-server: тонкая настройка планировщика под Ice Lake
    # -flto=auto: межмодульная оптимизация (LTO) с авто-выбором потоков
    # -fuse-linker-plugin: интеграция LTO с линковщиком для dead code elimination
    # -funroll-loops: разворот циклов для уменьшения branch overhead
    # -floop-nest-optimize: оптимизация вложенных циклов для лучшей векторизации
    # -floop-interchange: перестановка циклов для лучшей локальности кэша
    # -funsafe-math-optimizations: быстрые математические операции (ассоциативность)
    # -mprefer-vector-width=512: приоритет 512-битным векторам AVX-512
    # -frename-registers: переименование регистров для лучшего ILP
    # -fira-hoist-pressure: давление на вынесение инструкций из IRA
    # -falign-functions=32: выравнивание функций для лучшего branch prediction
    # -falign-loops=16: выравнивание циклов для лучшего предсказания ветвлений
    # -fopenmp: поддержка многопоточности OpenMP
    # -fprefetch-loop-arrays: аппаратный prefetch для массивов в циклах
    # -fno-math-errno: не устанавливать errno для math функций (быстрее)
    # -fno-trapping-math: отключить trapping для NaN/Inf (быстрее)
    # -ftree-vectorize: автоматическая векторизация деревьев выражений
    # -ftree-loop-distribution: распределение циклов для лучшей векторизации
    # -ftree-loop-if-convert: конвертация условий в циклах в ветвления-free код
    # -fgcse-after-reload: глобальная оптимизация общих подвыражений после reload
    
    set(GCC_OPT_FLAGS
        "-Ofast"
        "-march=icelake-server"
        "-mtune=icelake-server"
        "-flto=auto"
        "-fuse-linker-plugin"
        "-funroll-loops"
        "-floop-nest-optimize"
        "-floop-interchange"
        "-funsafe-math-optimizations"
        "-mprefer-vector-width=512"
        "-frename-registers"
        "-fira-hoist-pressure"
        "-falign-functions=32"
        "-falign-loops=16"
        "-fopenmp"
        "-fprefetch-loop-arrays"
        "-fno-math-errno"
        "-fno-trapping-math"
        "-ftree-vectorize"
        "-ftree-loop-distribution"
        "-ftree-loop-if-convert"
        "-fgcse-after-reload"
    )
    
    string(REPLACE ";" " " GCC_OPT_FLAGS_STR "${GCC_OPT_FLAGS}")
    message(STATUS "GCC оптимизации для Ice Lake: ${GCC_OPT_FLAGS_STR}")
    
    # Применить глобальные оптимизации ко всем C/C++ файлам
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_OPT_FLAGS_STR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_OPT_FLAGS_STR}")
    
    # AVX-512 compile flags for wisp_avx512.c (Intel Ice Lake+ optimized)
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_c_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
    check_c_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
    check_c_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
    check_c_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)
    check_c_compiler_flag("-mavx512vbmi2" COMPILER_SUPPORTS_AVX512VBMI2)
    check_c_compiler_flag("-mavx512vp2intersect" COMPILER_SUPPORTS_AVX512VP2INTERSECT)
    
    if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512VL AND 
       COMPILER_SUPPORTS_AVX512BW AND COMPILER_SUPPORTS_AVX512DQ)
        set(AVX512_FLAGS "-mavx512f" "-mavx512vl" "-mavx512bw" "-mavx512dq")
        if(COMPILER_SUPPORTS_AVX512VNNI)
            list(APPEND AVX512_FLAGS "-mavx512vnni")
        endif()
        if(COMPILER_SUPPORTS_AVX512VBMI2)
            list(APPEND AVX512_FLAGS "-mavx512vbmi2")
        endif()
        
        # Дополнительные оптимизации для AVX-512 файла
        set_source_files_properties(core/wisp_avx512.c PROPERTIES 
            COMPILE_OPTIONS "${AVX512_FLAGS}")
        message(STATUS "AVX-512 + Ice Lake оптимизации для wisp_avx512.c включены")
    else()
        message(WARNING "AVX-512 не поддерживается компилятором, используются базовые Ice Lake оптимизации")
    endif()
    
    # io_uring support for wisp_async_io.c on Linux
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_library(LIBURING NAMES io_uring)
        if(LIBURING)
            message(STATUS "io_uring library found: ${LIBURING}")
            target_link_libraries(_wisp_core PRIVATE ${LIBURING})
            target_compile_definitions(_wisp_core PRIVATE WISP_HAS_IO_URING)
        else()
            message(STATUS "io_uring not found, using POSIX fallback for async I/O")
        endif()
    endif()
endif()

# Threads: pthreads on POSIX (Windows uses the Win32 API directly)
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(_wisp_core PRIVATE Threads::Threads)
endif()

# Install straight into the wisp package directory
install(TARGETS _wisp_core LIBRARY DESTINATION wisp)
