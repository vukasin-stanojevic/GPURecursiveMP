//
// Created by vukasin on 11.9.22..
//

#ifndef MOOREPENROSE2_PINV_CPU_H
#define MOOREPENROSE2_PINV_CPU_H
#include<cblas-openblas.h>
#include<cblas.h>
#include <random>
#include<math.h>

namespace cpu_implementation {

    template<typename T>
    struct Matrix {
        T *data;
        int n_rows;
        int n_cols;
        int row_stride;
        int offset;

        Matrix(int n_rows, int n_cols, T *data = nullptr, int offset = 0) : n_rows(n_rows), n_cols(n_cols), data(data), offset(offset) {
            row_stride = n_cols;
        }

        Matrix(int n_rows, int n_cols, int row_stride, T *data = nullptr, int offset = 0) : n_rows(n_rows), n_cols(n_cols), data(data), offset(offset) {
            this->row_stride = row_stride;
        }

        Matrix<T> get_submatrix(const int i_start, const int i_end, const int j_start, const int j_end) const {
            if (i_end > i_start && j_start < j_end && i_start >= 0 && i_end <= n_rows && j_start >= 0 && j_end <= n_cols) {
                return Matrix(i_end - i_start, j_end - j_start, row_stride, data,
                              offset + i_start * row_stride + j_start);
            }

            assert(0);
            return Matrix(0, 0);
        }

        void print() const {
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    std::cout << data[offset + i * row_stride + j] << " ";
                }
                std::cout << std::endl;
            }
        }

        T get_element(const int i, const int j) const {
            if (i >= 0 && i < n_rows && j >= 0 && j < n_cols) {
                return data[offset + i * row_stride + j];
            }
            return 0;
        }

        void set_element(const int i, const int j, T elem) {
            if (i >= 0 && i < n_rows && j >= 0 && j < n_cols) {
                data[offset + i * row_stride + j] = elem;
            }
        }

        int get_offset() const { return offset; }

        size_t get_size() const { return n_rows * n_cols; }

        void allocate_memory(InitMode init_mode = InitMode::NO_INIT) {
            if (this->data) return;
            this->data = new T[this->get_size()];

            std::default_random_engine generator;
            std::uniform_real_distribution<T> distribution(-1, 1);

            if (init_mode == InitMode::RANDOM) {
                for (int i = 0; i < n_cols * n_rows; i++) {
                    data[i] = distribution(generator);
                }
            }
        }

        void free_memory() {
            if (!data) return;
            delete[] this->data;
            this->data = nullptr;
        }

        friend bool operator==(const Matrix &A, const Matrix &B) {
            if (A.n_cols != B.n_cols || A.n_rows != B.n_rows) return false;
            int c = 0;
            for (int i = 0; i < A.n_rows; i++) {
                for (int j = 0; j < A.n_cols; j++) {
                    if (std::abs(A.get_element(i, j) - B.get_element(i, j)) > 5 * 1e-5) c++;
                }
            }
            //std::cout << c << std::endl;
            return c == 0;
        }

        static void subtract(const Matrix &A, const Matrix &B, Matrix &C) {
            assert(A.n_cols == B.n_cols && A.n_rows == B.n_rows);
            C.allocate_memory();
            for (int i = 0; i < A.n_rows; i++) {
                for (int j = 0; j < A.n_cols; j++) {
                    C.set_element(i, j, A.get_element(i, j) - B.get_element(i, j));
                }
            }
        }

        static void scale(const Matrix<T>& A, const T scale, Matrix<T> A_scaled) {
            A_scaled.allocate_memory();
            for (int i = 0; i < A.n_rows; i++) {
                for (int j = 0; j < A.n_cols; j++) {
                    A_scaled.set_element(i, j, A.get_element(i, j) * scale);
                }
            }
        }
    };


    template<typename T>
    __host__ __device__ void matmul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, bool transposeA = false,
                                    bool transposeB = false, const T c = 1) {

        CBLAS_ORDER rowMajorOrder = CBLAS_ORDER::CblasRowMajor;
        CBLAS_TRANSPOSE transA = transposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
        CBLAS_TRANSPOSE transB = transposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;

        int m = transposeA ? A.n_cols : A.n_rows;
        int n = transposeB ? B.n_rows : B.n_cols;
        int k1 = transposeA ? A.n_rows : A.n_cols;
        int k2 = transposeB ? B.n_cols : B.n_rows;

        assert(k1 == k2);
        assert(C.n_rows == m && C.n_cols == n);

        int k = k1;

        const T alpha = c;

        int lda = A.row_stride;
        int ldb = B.row_stride;
        int ldc = C.row_stride;

        if constexpr (std::is_same<T, float>::value) {
            cblas_sgemm(
                    rowMajorOrder,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    alpha,
                    A.data + A.offset,
                    lda,
                    B.data + B.offset,
                    ldb,
                    0,
                    C.data + C.offset,
                    ldc
            );
        } else {
            cblas_dgemm(
                    rowMajorOrder,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    alpha,
                    A.data + A.offset,
                    lda,
                    B.data + B.offset,
                    ldb,
                    0,
                    C.data + C.offset,
                    ldc
            );
        }
    }

    template<typename T>
    void generalized_cholesky_factorization_1x1_case(const Matrix<T>& A, Matrix<T>& U, Matrix<T>& Y) {

        assert(U.data);
        assert(Y.data);
        assert(A.data);

        int i = 0;
        int j = 0;

        if (A.get_element(i, j) == 0) {
            U.set_element(i, j, 0);
            Y.set_element(i, j, 0);
        } else {
            U.set_element(i, j, sqrt(A.get_element(i, j)));
            Y.set_element(i, j, sqrt(1 / A.get_element(i, j)));
        }
    }

    template<typename T>
    void generalized_cholesky_factorization(const Matrix<T> A, Matrix<T> U, Matrix<T> Y) {
        assert(A.n_rows == A.n_cols);

        int n = A.n_rows;
        if (n == 1) {
            generalized_cholesky_factorization_1x1_case<T>(A, U, Y);
        } else {
            const int k = n / 2;
            Matrix<T> A11 = A.get_submatrix(0, k, 0, k);
            Matrix<T> A12 = A.get_submatrix(0, k, k, n);
            Matrix<T> A22 = A.get_submatrix(k, n, k, n);

            Matrix<T> U11 = U.get_submatrix(0, k, 0, k);
            Matrix<T> U12 = U.get_submatrix(0, k, k, n);
            Matrix<T> U22 = U.get_submatrix(k, n, k, n);

            Matrix<T> Y11 = Y.get_submatrix(0, k, 0, k);
            Matrix<T> Y12 = Y.get_submatrix(0, k, k, n);
            Matrix<T> Y22 = Y.get_submatrix(k, n, k, n);

            generalized_cholesky_factorization(A11, U11, Y11);

            matmul(Y11, A12, U12, true); // U12 = Y11.T @ A12

            Matrix<T> T1(U12.n_cols, U12.n_cols);
            Matrix<T> T2(A22.n_rows, A22.n_cols);
            Matrix<T> T3(Y11.n_rows, U12.n_cols);

            T1.allocate_memory();
            T2.allocate_memory();
            T3.allocate_memory();

            matmul(U12, U12, T1, true); // T1 = U12.T @ U12
            Matrix<T>::subtract(A22, T1, T2); // T2 = A22 - T2

            generalized_cholesky_factorization(T2, U22, Y22);

            matmul(Y11, U12, T3, false, false, (T) -1); // T3 = -Y11 @ U12
            matmul(T3, Y22, Y12); // Y12 = T3 @ Y22

             T1.free_memory();
             T2.free_memory();
             T3.free_memory();
        }
    }

    template<typename T>
    void calculate_nonzero_values(const Matrix<T> A, bool* count_array) {
        for (int i=0; i < A.n_rows; i++) {
            for (int j=0; j < A.n_cols; j++) {
                if (std::abs(A.get_element(i, j)) > 1e-6) {
                    count_array[i] = true;
                    break;
                }
            }
        }
    }

    template<typename T>
    void filter_out_zeros_rows(const Matrix<T>& A, Matrix<T>& Out, int* index_mapping) {
        for (int i = 0; i < A.n_rows; i++) {
            if (index_mapping[i] == -1) continue;
            for (int j=0; j < A.n_cols; j++) {
                Out.set_element(index_mapping[i], j, A.get_element(i, j));
            }
        }
    }

    template<typename T>
    Matrix<T> remove_zero_rows(const Matrix<T> A) {
        bool* count_array = new bool[A.n_rows];
        calculate_nonzero_values(A, count_array);

        int nonzero_rows_count = 0;
        int * row_mapping = (int*)malloc(A.n_rows * sizeof(int));

        for (int i = 0; i < A.n_rows; ++i) {
            if (count_array[i])
                row_mapping[i] = nonzero_rows_count++;
            else
                row_mapping[i] = -1;
        }

        delete [] count_array;

        Matrix<T> Out(nonzero_rows_count, A.n_cols);
        Out.allocate_memory();

        filter_out_zeros_rows(A, Out, row_mapping);

        return Out;
    }

    template<typename T>
    void strassen_matrix_inversion(const Matrix<T>& A, Matrix<T>& Inv) {
        assert(A.n_rows == A.n_cols);

        int n = A.n_rows;
        if (n == 1) {
            Inv.set_element(0, 0, 1/A.get_element(0, 0));
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


            matmul(A21, R1, R2, false, false, (T)1);
            matmul(R1, A12, R3, false, false, (T)1);
            matmul(A21, R3, R4, false, false, (T)1);
            Matrix<T>::subtract(R4, A22, R4);

            strassen_matrix_inversion(R4, R6);

            matmul(R3, R6, X12, false, false);
            matmul(R6, R2, X21, false, false);
            matmul(R3, X21, R7, false, false);

            Matrix<T>::subtract(R1, R7, X11);
            Matrix<T>::scale(R6, -1, X22);

            R1.free_memory();
            R2.free_memory();
            R3.free_memory();
            R4.free_memory();
            R6.free_memory();
            R7.free_memory();
        }
    }

    template<typename Type>
    Matrix<Type> calculate_moore_penrose_pseudo_inverse(const Matrix<Type>& A) {
        cudaEvent_t start, end;
        CUDA_CALL(cudaEventCreate(&start));
        CUDA_CALL(cudaEventCreate(&end));
        CUDA_CALL(cudaEventRecord(start));

        int n = A.n_cols;
        Matrix<Type> A_p(n, n);
        Matrix<Type> MP(A.n_cols, A.n_rows);
        MP.allocate_memory();
        A_p.allocate_memory();

        matmul(A, A, A_p, true);
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
        matmul(LT, LT, T, false, true);
        Matrix<Type> M (T.n_rows, T.n_cols);
        M.allocate_memory();

        strassen_matrix_inversion(T, M);

        T.free_memory();

        Matrix<Type> tmp(LT.n_rows, A.n_rows);
        Matrix<Type> tmp2(M.n_rows, tmp.n_cols);
        Matrix<Type> tmp3(M.n_rows, M.n_cols);
        tmp.allocate_memory();
        tmp2.allocate_memory();
        tmp3.allocate_memory();

        matmul(M, M, tmp3, false, false);
        matmul(LT, A, tmp, false, true);

        matmul(tmp3, tmp, tmp2);
        matmul(LT, tmp2, MP, true);

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

#endif //MOOREPENROSE2_PINV_CPU_H
