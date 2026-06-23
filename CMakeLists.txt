cmake_minimum_required(VERSION 3.22)
project(openastronv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86;89;90" CACHE STRING "CUDA architectures")

if(WIN32)
  add_definitions(-D_WIN32_WINNT=0x0601 -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
  set(PLATFORM_LIBS ws2_32 crypt32)
else()
  set(PLATFORM_LIBS pthread)
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

set_target_properties(openastronv_v3 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  COMPILE_OPTIONS "-Xptxas=-v"
)
