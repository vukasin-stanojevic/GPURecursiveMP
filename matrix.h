//
// Created by vukasin on 7/3/21.
//

#ifndef MOOREPENROSE_MATRIX_H
#define MOOREPENROSE_MATRIX_H
#include "utils/matrix_utils_kernels.h"
#include "utils/functors.h"
#include <type_traits>
#include <cublasLt.h>
#include <cassert>

enum InitMode {NO_INIT, RANDOM, ZEROS};

#define USE_SLOW_MATMUL false
#define TILE_DIM 32

namespace gpu_implementation {
    template<typename T>
    class Matrix;


    template<typename T>
    class Matrix {
    private:
        int offset = 0;
    public:
        int n_rows;
        int n_cols;
        int row_stride;
        T* d_data;

        __host__ __device__
        Matrix(int n_rows, int n_cols, T* data = nullptr, int row_stride=-1, int offset = 0): n_rows(n_rows), n_cols(n_cols), d_data(data), offset(offset) {
            if (row_stride == -1)
                this->row_stride = n_cols;
            else
                this->row_stride = row_stride;
        }

        __host__ __device__
        T get_element(const int i, const int j) const {
            if (i >= 0 && i < n_rows && j >= 0 && j < n_cols) {
                return d_data[offset + i * row_stride + j];
            }
            return 0;
        }

        __host__ __device__
        void set_element(const int i, const int j, T elem) {
            if (i >= 0 && i < n_rows && j >= 0 && j < n_cols) {
                d_data[offset + i * row_stride + j] = elem;
            }
        }

        __host__ __device__
        Matrix<T> get_submatrix(const int i_start, const int i_end, const int j_start, const int j_end) const {
            if (i_end > i_start && j_start < j_end && i_start >= 0 && i_end <= n_rows && j_start >= 0 && j_end <= n_cols) {
                return Matrix(i_end - i_start, j_end - j_start, d_data, row_stride,
                              offset + i_start * row_stride + j_start);
            }

            assert(0);
            return Matrix(0, 0);
        }

        __host__ __device__
        int get_offset() const { return offset; }

        __host__ __device__
        size_t get_size() const {return n_rows * n_cols;}

        __host__ __device__
        void allocate_memory(InitMode init_mode = InitMode::NO_INIT) {
            if (this->d_data) return;
#ifndef __CUDA_ARCH__ //host code
            CUDA_CALL(cudaMalloc((void**)(&(this->d_data)), this->get_size() * sizeof(T)));
            if (init_mode == InitMode::RANDOM) random_initialize();
#else
            this->d_data = new T[this->get_size()];
#endif
            if (init_mode == InitMode::ZEROS) zeros_initialize();
        }

        __host__ __device__
        void free_memory() {
            if (!d_data) return;

#ifndef __CUDA_ARCH__ //host code
            CUDA_CALL(cudaFree(this->d_data));
#else
            delete[] this->d_data;
#endif
            this->d_data = nullptr;
        }

        __host__
        void random_initialize() {
            // initializes the matrix with values from U(-1, 1)
            if (!this->d_data) {
                this->allocate_memory();
            }
            curandGenerator_t generator;
            CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rand()));

            float* tmp;
            CUDA_CALL(cudaMalloc(&tmp, sizeof(float) * (this->get_size() + this->get_size() % 2)));
            CURAND_CALL(curandGenerateUniform(generator, tmp, this->get_size() + this->get_size() % 2));

            if constexpr(std::is_same<T, float>::value) {
                CURAND_CALL(curandGenerateUniform(generator, this->d_data, this->get_size() + this->get_size() % 2));
            }
            if constexpr(std::is_same<T, double>::value) {
                CURAND_CALL(curandGenerateUniformDouble(generator, this->d_data, this->get_size() + this->get_size() % 2));
            }

            int block_size = 256;
            int grid_size = (this->get_size() + block_size - 1) / block_size;

            flip_signs<<<grid_size, block_size>>>(this->d_data, tmp, this->get_size());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaFree(tmp));
            CURAND_CALL(curandDestroyGenerator(generator));
        }

        __host__ __device__
        void zeros_initialize() {
            dim3 block(32, 32);
            dim3 grid((n_cols + block.x -1) / block.x, (n_rows + block.y -1) / block.y);
            set_to<T><<<grid, block>>>(*this, 0);
            CUDA_CALL(cudaDeviceSynchronize());
        }

        void set_values(T* host_data, cudaStream_t stream = nullptr, bool sync = true) {
            if (!host_data) {
                printf("Host pointer is NULL\n");
                return;
            }
            if (!this->d_data) this->allocate_memory();
            CUDA_CALL(cudaMemcpyAsync(this->d_data, host_data, this->get_size()*sizeof(T), cudaMemcpyHostToDevice, stream));
            if (sync) CUDA_CALL(cudaStreamSynchronize(stream));
        }

        T* get_values(T* host_data = nullptr, cudaStream_t stream = nullptr, bool sync = true) {
            if (!host_data) {
                host_data = (T*)malloc(this->get_size() * sizeof(T));
            }

            CUDA_CALL(cudaMemcpyAsync(host_data, this->d_data, this->get_size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
            if (sync) CUDA_CALL(cudaStreamSynchronize(stream));
            return host_data;

        }

        void print() {
            printf("\n");
            print_matrix<T><<<1, 1>>>(*this);
            CUDA_CALL(cudaDeviceSynchronize());
        }

        template<typename U>
        __host__ __device__ bool operator == (const Matrix<U> other) const {
            if (other.n_cols != this->n_cols) return false;
            if (other.n_rows != this->n_rows) return false;

            dim3 block_size(32, 32);
            dim3 grid_size((other.n_cols + 31)/32, (other.n_rows + 31)/32);

            int* error_num_d;
            int error_num_h;
            CUDA_CALL(cudaMalloc(&error_num_d, sizeof(int)));
            set_variable_to<<<1, 1>>>(error_num_d, 0);
            check_equality<<<grid_size, block_size>>>(*this, other, error_num_d, 5*1e-5);
            CUDA_CALL(cudaMemcpy(&error_num_h, error_num_d, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaFree(error_num_d));
            //printf("\n%d\n", error_num_h);
            return error_num_h == 0;
        }

        template<typename U>
        __host__ __device__ bool operator != (const Matrix<U> other) const {
            return !(*this == other);
        }

        static __host__ __device__
        void scale(const Matrix<T>& A, const T scale, Matrix<T> A_scaled, cudaStream_t stream = nullptr, bool sync = true) {
            dim3 block_size(32, 32);
            dim3 grid_size((A.n_cols + 31)/32, (A.n_rows + 31)/32);
            scale_kernel<<<grid_size, block_size, 0, stream>>>(A, scale, A_scaled);
            if (sync) CUDA_CALL(cudaStreamSynchronize(stream));
        }

        static __host__ __device__
        bool is_identity(const Matrix<T> A, float tolerance = 1e-7) {
            if (A.n_cols != A.n_rows) return false;
            int n = A.n_rows;

            dim3 block_size(32, 32);
            dim3 grid_size((n + 31)/32, (n + 31)/32);

            int* error_num_d;
            int error_num_h;
            CUDA_CALL(cudaMalloc(&error_num_d, sizeof(int)));
            set_variable_to<<<1, 1>>>(error_num_d, 0);
            check_if_matrix_is_identity<<<grid_size, block_size>>>(A, error_num_d, tolerance);
            CUDA_CALL(cudaMemcpy(&error_num_h, error_num_d, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaFree(error_num_d));
            return error_num_h == 0;
        }

        static __host__ __device__
        Matrix<T> get_identity_matrix(int n, cudaStream_t stream = nullptr, bool sync = true) {
            Matrix<T> I(n, n);
            I.allocate_memory(InitMode::ZEROS);
            add_diag<T><<<(n + 255)/256, 256, 0, stream>>>(I, 1);
            if (sync)
                CUDA_CALL(cudaStreamSynchronize(stream));
            return I;
        }

        static __host__ __device__
        Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T> C, cudaStream_t stream = nullptr, bool sync = true) {
            dim3 block_size(32, 32);
            dim3 grid_size((A.n_cols + 31)/32, (A.n_rows + 31)/32);

            element_wise_op<T, Add><<<grid_size, block_size>>>(A, B, C);
            if (sync)
                CUDA_CALL(cudaStreamSynchronize(stream));
        }

        static __host__ __device__
        Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, cudaStream_t stream = nullptr, bool sync = true) {
            dim3 block_size(32, 32);
            dim3 grid_size((A.n_cols + 31)/32, (A.n_rows + 31)/32);

            element_wise_op<T, Subtract><<<grid_size, block_size>>>(A, B, C);
            if (sync)
                CUDA_CALL(cudaStreamSynchronize(stream));

            return C;
        }
    };


    template<typename T, int block_size, bool transposeA = false, bool transposeB = false>
    __global__ void matmul_kernel(Matrix<T> A, Matrix<T> B, Matrix<T> C, const T c = 1) {
        __shared__ T As[block_size][block_size];
        __shared__ T Bs[block_size][block_size + (transposeB? 1 : 0)];

        int block_row_idx = blockIdx.y;
        int block_col_idx = blockIdx.x;

        T result = 0;
        int block_num;

        if (transposeA)
            block_num = (A.n_rows + block_size - 1) / block_size;
        else
            block_num = (A.n_cols + block_size - 1) / block_size;

        for (int m = 0; m < block_num; m++) {
            if (transposeA) {
                As[threadIdx.y][threadIdx.x] = A.get_element(m * block_size + threadIdx.y,
                                                             block_row_idx * block_size + threadIdx.x);
            } else {
                As[threadIdx.y][threadIdx.x] = A.get_element(block_row_idx * block_size + threadIdx.y,
                                                             m * block_size + threadIdx.x);
            }
            if (transposeB) {
                Bs[threadIdx.y][threadIdx.x] = B.get_element(block_col_idx * block_size + threadIdx.y,
                                                             m * block_size + threadIdx.x);
            } else {
                Bs[threadIdx.y][threadIdx.x] = B.get_element(m * block_size + threadIdx.y,
                                                             block_col_idx * block_size + threadIdx.x);
            }

            __syncthreads();

            if (transposeA && transposeB) {
                for (int e = 0; e < block_size; ++e)
                    result += As[e][threadIdx.y] * Bs[threadIdx.x][e];
            } else if (transposeA && !transposeB) {
                for (int e = 0; e < block_size; ++e)
                    result += As[e][threadIdx.y] * Bs[e][threadIdx.x];
            } else if (!transposeA && transposeB) {
                for (int e = 0; e < block_size; ++e)
                    result += As[threadIdx.y][e] * Bs[threadIdx.x][e];
            } else {
                for (int e = 0; e < block_size; ++e)
                    result += As[threadIdx.y][e] * Bs[e][threadIdx.x];
            }

            __syncthreads();
        }

        C.set_element(block_row_idx*block_size + threadIdx.y, block_col_idx * block_size + threadIdx.x, result * c);
    }

    template<typename T>
    __host__ __device__ void matmul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, bool transposeA = false,
                                    bool transposeB = false, const T c = 1, cudaStream_t stream = 0, bool sync = true) {

        cublasLtHandle_t handle;
        CUBLAS_CALL(cublasLtCreate(&handle));
        cublasLtMatrixLayout_t ALayout, BLayout, CLayout;
        cublasComputeType_t computeType;
        cublasDataType_t dataType;
        cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;

        if constexpr (std::is_same<T, float>::value) {
            computeType = CUBLAS_COMPUTE_32F;
            dataType = CUDA_R_32F;
        } else {
            computeType = CUBLAS_COMPUTE_64F;
            dataType = CUDA_R_64F;
        }

        CUBLAS_CALL(cublasLtMatrixLayoutCreate(&ALayout, dataType, A.n_rows, A.n_cols, A.row_stride));
        CUBLAS_CALL(cublasLtMatrixLayoutCreate(&BLayout, dataType, B.n_rows, B.n_cols, B.row_stride));
        CUBLAS_CALL(cublasLtMatrixLayoutCreate(&CLayout, dataType, C.n_rows, C.n_cols, C.row_stride));

        CUBLAS_CALL(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
        CUBLAS_CALL(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));
        CUBLAS_CALL(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)));

        cublasLtMatmulDesc_t matmulDesc;
        CUBLAS_CALL(cublasLtMatmulDescCreate(&matmulDesc, computeType, dataType));

        bool transA = transposeA;
        bool transB = transposeB;

        cublasOperation_t operationA = transA? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t operationB = transB? CUBLAS_OP_T : CUBLAS_OP_N;

        CUBLAS_CALL(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &operationA, sizeof(operationA)));
        CUBLAS_CALL(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &operationB, sizeof(operationB)));

        int returnAlgoCount;
        cublasLtMatmulPreference_t preference;
        CUBLAS_CALL(cublasLtMatmulPreferenceCreate(&preference));
        cublasLtMatmulHeuristicResult_t heuristicResultsArray[1];

        CUBLAS_CALL(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, ALayout, BLayout, CLayout, CLayout, preference, 1, heuristicResultsArray, &returnAlgoCount));
        assert(returnAlgoCount > 0);

        float alpha = c;
        float beta = 0;
        CUBLAS_CALL(cublasLtMatmul(handle, matmulDesc, &alpha, A.d_data+A.get_offset(), ALayout, B.d_data+B.get_offset(), BLayout, &beta, 0, CLayout, C.d_data+C.get_offset(), CLayout, &(heuristicResultsArray[0].algo), 0, 0, stream));

        if (sync)
            CUDA_CALL(cudaStreamSynchronize(stream));

        CUBLAS_CALL(cublasLtMatmulDescDestroy(matmulDesc));
        CUBLAS_CALL(cublasLtMatrixLayoutDestroy(ALayout));
        CUBLAS_CALL(cublasLtMatrixLayoutDestroy(BLayout));
        CUBLAS_CALL(cublasLtMatrixLayoutDestroy(CLayout));
        CUBLAS_CALL(cublasLtMatmulPreferenceDestroy(preference));
    }

    template<typename T, bool slow_version>
    __host__ __device__ void matmul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, bool transposeA = false,
                                     bool transposeB = false, const T c = 1, cudaStream_t stream = 0, bool sync = true) {
        if constexpr (!slow_version) {
            matmul(A, B, C, transposeA, transposeB, c, stream, sync);
            return;
        }

        int block_dim_x = TILE_DIM;
        int block_dim_y = TILE_DIM;
        int n = transposeA? A.n_cols : A.n_rows;
        int m = transposeB? B.n_rows : B.n_cols;
        int k1 = transposeA? A.n_rows : A.n_cols;
        int k2 = transposeB? B.n_cols : B.n_rows;

        assert(k1 == k2);
        assert(C.n_rows == n && C.n_cols == m);

        if (!C.d_data) {
            C.allocate_memory();
        }
        int grid_dim_x = (m + block_dim_x - 1) / block_dim_x;
        int grid_dim_y = (n + block_dim_y - 1) / block_dim_y;
        dim3 block_dim(block_dim_x, block_dim_y);
        dim3 grid_dim(grid_dim_x, grid_dim_y);

        if (transposeA && transposeB)
            matmul_kernel<T, TILE_DIM, true, true><<<grid_dim, block_dim, 0, stream>>>(A, B, C, c);
        else if (!transposeA && transposeB)
            matmul_kernel<T, TILE_DIM, false, true><<<grid_dim, block_dim, 0, stream>>>(A, B, C, c);
        else if (transposeA && !transposeB)
            matmul_kernel<T, TILE_DIM, true, false><<<grid_dim, block_dim, 0, stream>>>(A, B, C, c);
        else
            matmul_kernel<T, TILE_DIM, false, false><<<grid_dim, block_dim, 0, stream>>>(A, B, C, c);

        if (sync)
            CUDA_CALL(cudaStreamSynchronize(stream));
    }
}



#endif //MOOREPENROSE_MATRIX_H
