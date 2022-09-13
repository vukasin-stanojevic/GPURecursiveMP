//
// Created by vukasin on 7/9/21.
//

#ifndef MOOREPENROSE_STRASSEN_MATRIX_INV_H
#define MOOREPENROSE_STRASSEN_MATRIX_INV_H

namespace gpu_implementation {
    template<typename T>
    class Matrix;

    template<typename T>
    __global__
    void strassen_matrix_inversion_1x1(const Matrix<T> A, Matrix<T> Inv) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i == 0 && j == 0) {
            Inv.set_element(i, j, 1/A.get_element(i, j));
        }
    }

    template<typename T>
    void strassen_matrix_inversion(const Matrix<T> A, Matrix<T> Inv, cudaStream_t stream = nullptr) {
        assert(A.n_rows == A.n_cols);

        int n = A.n_rows;
        if (n == 1) {
            strassen_matrix_inversion_1x1<T><<<1, 1>>>(A, Inv);
            CUDA_CALL(cudaStreamSynchronize(stream));
        } else {
            const int k = n / 2;
            Matrix<T> A11 = A.get_submatrix(0, k, 0, k);
            Matrix<T> A12 = A.get_submatrix(0, k, k, n);
            Matrix<T> A21 = A.get_submatrix(k, n, 0, k);
            Matrix<T> A22 = A.get_submatrix(k, n, k, n);

            Matrix<T> X11 = Inv.get_submatrix(0, k, 0, k);
            Matrix<T> X12 = Inv.get_submatrix(0, k, k, n);
            Matrix<T> X21 = Inv.get_submatrix(k, n, 0, k);
            Matrix<T> X22 = Inv.get_submatrix(k, n, k, n);

            Matrix<T> R1(A11.n_rows, A11.n_cols);
            R1.allocate_memory();
            strassen_matrix_inversion(A11, R1);

            Matrix<T> R2(A21.n_rows, R1.n_cols);
            Matrix<T> R3(R1.n_rows, A12.n_cols);
            Matrix<T> R4(A21.n_rows, R3.n_cols);
            Matrix<T> R6(A22.n_rows, A22.n_cols);
            Matrix<T> R7(R3.n_rows, X21.n_cols);

            R2.allocate_memory();
            R3.allocate_memory();
            R4.allocate_memory();
            R6.allocate_memory();
            R7.allocate_memory();

            cudaStream_t s1 = StreamManager::get_stream();
            cudaStream_t s2 = StreamManager::get_stream();

            matmul<T, USE_SLOW_MATMUL>(A21, R1, R2, false, false, (T)1, s1, false);
            matmul<T, USE_SLOW_MATMUL>(R1, A12, R3, false, false, (T)1, s2, false);
            matmul<T, USE_SLOW_MATMUL>(A21, R3, R4, false, false, (T)1, s2);
            Matrix<T>::subtract(R4, A22, R4, s2, false);

            StreamManager::synchronize();
            strassen_matrix_inversion(R4, R6);

            matmul<T, USE_SLOW_MATMUL>(R3, R6, X12, false, false, (T)1, s1, false);
            matmul<T, USE_SLOW_MATMUL>(R6, R2, X21, false, false, (T)1, s2, false);
            matmul<T, USE_SLOW_MATMUL>(R3, X21, R7, false, false, (T)1, s2, false);

            Matrix<T>::subtract(R1, R7, X11, s2, false);
            Matrix<T>::scale(R6, -1, X22, s1, false);

            StreamManager::synchronize();
            R1.free_memory();
            R2.free_memory();
            R3.free_memory();
            R4.free_memory();
            R6.free_memory();
            R7.free_memory();
        }
    }
}


#endif //MOOREPENROSE_STRASSEN_MATRIX_INV_H
