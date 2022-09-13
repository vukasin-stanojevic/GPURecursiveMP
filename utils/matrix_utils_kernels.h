//
// Created by vukasin on 7/4/21.
//

#ifndef MOOREPENROSE_MATRIX_UTILS_KERNELS_H
#define MOOREPENROSE_MATRIX_UTILS_KERNELS_H

namespace gpu_implementation {

    template<typename T>
    class Matrix;

    template<typename T>
    __global__
    void print_matrix(Matrix<T> A) {
        for (int i = 0; i < A.n_rows; ++i) {
            for (int j = 0; j < A.n_cols; ++j)
                if constexpr(std::is_same<T, float>::value)
                    printf("%f ", A.get_element(i, j));
                else if constexpr(std::is_same<T, double>::value)
                    printf("%lf ", A.get_element(i, j));
            printf("\n");
        }
    }

    template<typename T>
    __global__ void set_to(Matrix<T> matrix, const T val) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < matrix.n_rows && j < matrix.n_cols) {
            matrix.set_element(i, j, val);
        }
    }

    template<typename T>
    __global__ void set_variable_to(T* var, T val) {
        *var = val;
    }

    template<typename T>
    __global__
    void flip_signs(T* array, float* aux, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n) {
            if (aux[i] > 0.5) array[i] *= -1;
        }
    }

    template<typename T, template<typename> typename OP>
    __global__ void element_wise_op(const Matrix<T>A, const Matrix<T> B, Matrix<T> C) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        C.set_element(i, j, OP<T>::f(A.get_element(i, j), B.get_element(i, j)));
    }

    template<typename T>
    __global__ void check_if_matrix_is_identity(Matrix<T>A, int * error_num, float tolerance = 1e-7) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;
        int target_val;

        if (i < A.n_rows && j < A.n_cols) {
            target_val = i == j? 1: 0;
            if (abs(A.get_element(i, j) - target_val) > tolerance) {
                atomicAdd(error_num, 1);
            }
        }
    }

    template<typename T>
    __global__ void check_equality(const Matrix<T> A, const Matrix<T> B, int * error_num, float tolerance = 1e-7) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < A.n_rows && j < A.n_cols) {
            if (abs(A.get_element(i, j) - B.get_element(i, j)) > tolerance) {
                atomicAdd(error_num, 1);
            }
        }
    }

    template<typename T>
    __global__ void scale_kernel(const Matrix<T> A, const T scale, Matrix<T> A_scaled) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        A_scaled.set_element(i, j, A.get_element(i, j) * scale);
    }

    template<typename T>
    __global__ void add_diag(Matrix<T> matrix, T val) {
        int n = matrix.n_rows;
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < n) {
            matrix.set_element(i, i, matrix.get_element(i, i) + val);
        }
    }
}


#endif //MOOREPENROSE_MATRIX_UTILS_KERNELS_H
