cmake_minimum_required(VERSION 3.22)
project(openastronv LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Bare numbers (not "-real") on purpose: CMake emits both SASS and PTX for each,
# so the binary keeps a virtual target and a driver can JIT for an architecture
# newer than anything listed here. 120 is Blackwell / RTX 50-series and needs
# CUDA >= 12.8 to compile; older toolkits will reject it.
set(CMAKE_CUDA_ARCHITECTURES "86;89;90;120" CACHE STRING "CUDA architectures")
set(DIRTYBIRD_VERSION "dev" CACHE STRING "Miner version shown by --version")

if(WIN32)
  add_definitions(-D_WIN32_WINNT=0x0601 -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
  set(PLATFORM_LIBS ws2_32 crypt32)
else()
  set(PLATFORM_LIBS pthread)
endif()

if(MSVC AND "${VCPKG_TARGET_TRIPLET}" MATCHES "-static$")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  set(OPENSSL_MSVC_STATIC_RT TRUE)
endif()

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  add_definitions(-DHAS_OPENSSL)
  set(SSL_LIBS OpenSSL::SSL OpenSSL::Crypto)
else()
  set(SSL_LIBS "")
endif()

add_executable(openastronv_v3
  src/main.cpp
  src/gpu/astrobwt_gpu.cu
  extern/libsais/libsais.c
)
set_source_files_properties(extern/libsais/libsais.c PROPERTIES LANGUAGE C)

target_include_directories(openastronv_v3 PRIVATE
  src
  extern/libsais
)
target_link_libraries(openastronv_v3 PRIVATE ${PLATFORM_LIBS} ${SSL_LIBS})
target_compile_definitions(openastronv_v3 PRIVATE MINER_VER="${DIRTYBIRD_VERSION}")

target_compile_options(openastronv_v3 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
if(MSVC)
  target_compile_options(openastronv_v3 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
  )
endif()

set_target_properties(openastronv_v3 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
