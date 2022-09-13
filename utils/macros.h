//
// Created by vukasin on 7/3/21.
//

#ifndef MOOREPENROSE_MACROS_H
#define MOOREPENROSE_MACROS_H

#include <cublas_v2.h>

#define CUDA_CALL(func_res) { \
	if (func_res != cudaSuccess) { \
		printf("Error: %s: %dims, ", __FILE__, __LINE__); \
		printf("code: %dims, reason: %s/n", func_res, cudaGetErrorString(func_res)); \
		exit(1);\
	}\
}

#define CURAND_CALL(function)\
{\
	const curandStatus_t error = function;\
		if (error != CURAND_STATUS_SUCCESS) {\
			printf("Error: %s: %dims, ", __FILE__, __LINE__);\
			printf("reason: %s/n", error);\
			exit(1);\
		}\
}

#define CUBLAS_CALL(function)\
{\
	const cublasStatus_t error = function;\
		if (error != CUBLAS_STATUS_SUCCESS) {\
			printf("Error: %s: %dims, ", __FILE__, __LINE__); \
		    printf("code: %dims, reason: %s/n", error, _cudaGetErrorEnum(error)); \
		    exit(1);\
		}\
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


#endif //MOOREPENROSE_MACROS_H
