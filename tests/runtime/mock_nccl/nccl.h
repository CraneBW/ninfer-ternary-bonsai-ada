#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ncclComm* ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;
typedef enum { ncclSuccess = 0, ncclInvalidArgument = 1 } ncclResult_t;
typedef enum { ncclBfloat16 = 9 } ncclDataType_t;
typedef enum { ncclSum = 0 } ncclRedOp_t;

ncclResult_t ncclGetUniqueId(ncclUniqueId* id);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int world, ncclUniqueId id, int rank);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclAllReduce(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclAllGather(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream);
const char* ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif
