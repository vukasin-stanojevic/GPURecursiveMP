//
// Created by vukasin on 7/9/21.
//

#ifndef MOOREPENROSE_REMOVE_ZERO_ROWS_H
#define MOOREPENROSE_REMOVE_ZERO_ROWS_H


namespace gpu_implementation {

    template<typename T>
    class Matrix;

    template<typename T>
    __global__ void calculate_nonzero_values(const Matrix<T> A, int * count_array) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;
        if (abs(A.get_element(i, j)) > 1e-6) {
            count_array[i] += 1; // no need to use atomicAdd because we only need to know if there is any nonzero value.
        }
    }

    template<typename T>
    __global__ void filter_out_zeros_rows(const Matrix<T> A, Matrix<T> Out, int* index_mapping) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < A.n_rows) {
            if (index_mapping[i] != -1) {
                Out.set_element(index_mapping[i], j, A.get_element(i, j));
            }
        }
    }

    template<typename T>
    Matrix<T> remove_zero_rows(const Matrix<T> A, cudaStream_t stream = nullptr) {
        int * count_array_d;
        int * count_array_h = (int*)malloc(A.n_rows * sizeof(int));
        int * total_num_of_nonzero;
        CUDA_CALL(cudaMalloc(&count_array_d, A.n_rows * sizeof(int)));
        CUDA_CALL(cudaMalloc(&total_num_of_nonzero, sizeof(int)));
        CUDA_CALL(cudaMemset(count_array_d, 0, A.n_rows * sizeof(int)));
        CUDA_CALL(cudaMemset(total_num_of_nonzero, 0, sizeof(int)));

        dim3 block_size(32, 32);
        dim3 grid_size((A.n_cols + 31)/32, (A.n_rows + 31)/32);
        calculate_nonzero_values<<<grid_size, block_size>>>(A, count_array_d);
        CUDA_CALL(cudaStreamSynchronize(stream));
        CUDA_CALL(cudaMemcpy(count_array_h, count_array_d, A.n_rows * sizeof(int), cudaMemcpyDeviceToHost));

        int nonzero_rows_count = 0;
        int * row_mapping_h = (int*)malloc(A.n_rows * sizeof(int));

        for (int i = 0; i < A.n_rows; ++i) {
            if (count_array_h[i] > 0)
                row_mapping_h[i] = nonzero_rows_count++;
            else
                row_mapping_h[i] = -1;
        }
        int* row_mapping_d = count_array_d;
        CUDA_CALL(cudaMemcpy(row_mapping_d, row_mapping_h, A.n_rows * sizeof(int), cudaMemcpyHostToDevice));
        free(count_array_h);
        free(row_mapping_h);

        Matrix<T> Out(nonzero_rows_count, A.n_cols);
        Out.allocate_memory();

        filter_out_zeros_rows<<<grid_size, block_size>>>(A, Out, row_mapping_d);
        CUDA_CALL(cudaStreamSynchronize(stream));
        CUDA_CALL(cudaFree(count_array_d));
        return Out;
    }
}


#endif //MOOREPENROSE_REMOVE_ZERO_ROWS_H
