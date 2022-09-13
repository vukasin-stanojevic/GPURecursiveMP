//
// Created by vukasin on 7/9/21.
//

#ifndef MOOREPENROSE_MOORE_PENROSE_H
#define MOOREPENROSE_MOORE_PENROSE_H

#include "strassen_matrix_inv.h"
#include "cholesky.h"
#include <fstream>

namespace gpu_implementation {

    template<typename Type>
    Matrix<Type> calculate_moore_penrose_pseudo_inverse(const gpu_implementation::Matrix<Type> A) {
        cudaEvent_t start, end;
        CUDA_CALL(cudaEventCreate(&start));
        CUDA_CALL(cudaEventCreate(&end));
        CUDA_CALL(cudaEventRecord(start));

        int n = A.n_cols;
        Matrix<Type> A_p(n, n);
        Matrix<Type> MP(A.n_cols, A.n_rows);
        MP.allocate_memory();
        A_p.allocate_memory();

        matmul<Type, USE_SLOW_MATMUL>(A, A, A_p, true);

        Matrix<Type> U(n, n), Y(n, n);
        U.allocate_memory();
        Y.allocate_memory();

        generalized_cholesky_factorization(A_p, U, Y);

        A_p.free_memory();
        Matrix<Type> LT = remove_zero_rows(U);

        U.free_memory();
        Y.free_memory();

        Matrix<Type> T(LT.n_rows, LT.n_rows);
        T.allocate_memory();
        matmul<Type, USE_SLOW_MATMUL>(LT, LT, T, false, true);

        Matrix<Type> M (T.n_rows, T.n_cols);
        M.allocate_memory();

        strassen_matrix_inversion(T, M);

        T.free_memory();

        cudaStream_t s1 = StreamManager::get_stream();
        cudaStream_t s2 = StreamManager::get_stream();

        Matrix<Type> tmp(LT.n_rows, A.n_rows);
        Matrix<Type> tmp2(M.n_rows, tmp.n_cols);
        Matrix<Type> tmp3(M.n_rows, M.n_cols);
        tmp.allocate_memory();
        tmp2.allocate_memory();
        tmp3.allocate_memory();

        matmul<Type, USE_SLOW_MATMUL>(M, M, tmp3, false, false, (Type)1, s1, false);
        matmul<Type, USE_SLOW_MATMUL>(LT, A, tmp, false, true, (Type)1, s2, false);
        StreamManager::synchronize();

        matmul<Type, USE_SLOW_MATMUL>(tmp3, tmp, tmp2);
        matmul<Type, USE_SLOW_MATMUL>(LT, tmp2, MP, true);
        CUDA_CALL(cudaDeviceSynchronize());

        float exec_time;
        M.free_memory();
        tmp.free_memory();
        tmp2.free_memory();
        tmp3.free_memory();
        LT.free_memory();
        CUDA_CALL(cudaEventRecord(end));
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventElapsedTime(&exec_time, start, end));
        CUDA_CALL(cudaEventDestroy(start));
        CUDA_CALL(cudaEventDestroy(end));
        printf("Calculation time: %f ms\n", exec_time);
        return MP;
    }

}

#endif //MOOREPENROSE_MOORE_PENROSE_H
