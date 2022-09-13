//
// Created by vukasin on 7/9/21.
//

#ifndef MOOREPENROSE_CHOLESKY_H
#define MOOREPENROSE_CHOLESKY_H
#include <fstream>

namespace gpu_implementation {

    template<typename T>
    __global__
    void generalized_cholesky_factorization_1x1_case(const Matrix<T> A, Matrix<T> U, Matrix<T> Y) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i == 0 && j == 0) {
            if (A.get_element(i, j) == 0) {
                U.set_element(i, j, 0);
                Y.set_element(i, j, 0);
            } else {
                U.set_element(i, j, sqrt(A.get_element(i, j)));
                Y.set_element(i, j, sqrt(1/A.get_element(i, j)));
            }
        }
    }

    template<typename T>
    void generalized_cholesky_factorization(const Matrix<T> A, Matrix<T> U, Matrix<T> Y, cudaStream_t stream = nullptr) {
        assert(A.n_rows == A.n_cols);

        int n = A.n_rows;
        if (n == 1) {
            generalized_cholesky_factorization_1x1_case<T><<<1, 1>>>(A, U, Y);
            CUDA_CALL(cudaStreamSynchronize(stream));
        } else {
            const int k = n / 2;
            // we partition the matrices
            Matrix<T> A11 = A.get_submatrix(0, k, 0, k);
            Matrix<T> A12 = A.get_submatrix(0, k, k, n);
            Matrix<T> A22 = A.get_submatrix(k, n, k, n);

            Matrix<T> U11 = U.get_submatrix(0, k, 0, k);
            Matrix<T> U12 = U.get_submatrix(0, k, k, n);
            Matrix<T> U22 = U.get_submatrix(k, n, k, n);

            Matrix<T> Y11 = Y.get_submatrix(0, k, 0, k);
            Matrix<T> Y12 = Y.get_submatrix(0, k, k, n);
            Matrix<T> Y22 = Y.get_submatrix(k, n, k, n);

            // calculate U11 and Y11
            generalized_cholesky_factorization(A11, U11, Y11);

            matmul<T, USE_SLOW_MATMUL>(Y11, A12, U12, true); // U12 = Y11.T @ A12

            // create temporary matrices and allocate memory
            Matrix<T> T1(U12.n_cols, U12.n_cols);
            Matrix<T> T2(A22.n_rows, A22.n_cols);
            Matrix<T> T3(Y11.n_rows, U12.n_cols);

            T1.allocate_memory();
            T2.allocate_memory();
            T3.allocate_memory();

            matmul<T, USE_SLOW_MATMUL>(U12, U12, T1, true); // T1 = U12.T @ U12
            Matrix<T>::subtract(A22, T1, T2); // T2 = A22 - T2

            // calculate U22 and Y22
            generalized_cholesky_factorization(T2, U22, Y22);

            matmul<T, USE_SLOW_MATMUL>(Y11, U12, T3, false, false, (T)-1); // T3 = -Y11 @ U12
            matmul<T, USE_SLOW_MATMUL>(T3, Y22, Y12); // Y12 = T3 @ Y22

            // free memory
            T1.free_memory();
            T2.free_memory();
            T3.free_memory();
        }
    }
}


#endif //MOOREPENROSE_CHOLESKY_H
