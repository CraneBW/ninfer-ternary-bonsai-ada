add_library(ninfer_tp2_collectives STATIC nccl_collectives.cpp v3_rank_manifest.cpp)
ninfer_internal_includes(ninfer_tp2_collectives)
target_link_libraries(ninfer_tp2_collectives PUBLIC CUDA::cudart)

set(NINFER_NCCL_ROOT "" CACHE PATH "Optional NCCL prefix for native TP2")
find_path(NINFER_NCCL_INCLUDE_DIR nccl.h
  HINTS ${NINFER_NCCL_ROOT} ${NINFER_NCCL_ROOT}/include)
find_library(NINFER_NCCL_LIBRARY NAMES nccl nccl.so.2
  HINTS ${NINFER_NCCL_ROOT} ${NINFER_NCCL_ROOT}/lib ${NINFER_NCCL_ROOT}/lib64)
if(NINFER_NCCL_INCLUDE_DIR AND NINFER_NCCL_LIBRARY)
  target_compile_definitions(ninfer_tp2_collectives PRIVATE NINFER_HAVE_NCCL=1)
  target_include_directories(ninfer_tp2_collectives PRIVATE ${NINFER_NCCL_INCLUDE_DIR})
  target_link_libraries(ninfer_tp2_collectives PUBLIC ${NINFER_NCCL_LIBRARY})
  message(STATUS "Native TP2 NCCL collectives enabled: ${NINFER_NCCL_LIBRARY}")
else()
  target_compile_definitions(ninfer_tp2_collectives PRIVATE NINFER_HAVE_NCCL=0)
  message(STATUS "Native TP2 NCCL collectives disabled (NCCL development library unavailable)")
endif()
