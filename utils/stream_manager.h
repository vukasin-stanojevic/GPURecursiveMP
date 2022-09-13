//
// Created by vukasin on 7/5/21.
//

#ifndef MOOREPENROSE_STREAM_MANAGER_H
#define MOOREPENROSE_STREAM_MANAGER_H


#include <unordered_map>

class StreamManager {
    static const int max_num_of_streams;
    static int curr_stream_index;
    static std::unordered_map<cudaStream_t, cublasHandle_t> handles;

public:
    static void init() {
        if (handles.empty()) {
            for (int i = 0; i < max_num_of_streams; ++i) {
                cudaStream_t stream;
                cublasHandle_t handle;
                CUDA_CALL(cudaStreamCreate(&stream));
                CUBLAS_CALL(cublasCreate(&handle));
                CUBLAS_CALL(cublasSetStream_v2(handle, stream));
                handles.insert(std::pair<cudaStream_t, cublasHandle_t>(stream, handle));
            }
        }
    }

    static cudaStream_t get_stream() {
        auto it = handles.begin();
        curr_stream_index = curr_stream_index < max_num_of_streams - 2? curr_stream_index + 1: 0;
        for (int i = 0; i < curr_stream_index; ++i)
            ++it;

        return it->first;
    }

    static void synchronize() {
        for (auto it = handles.begin(); it != handles.end(); ++it) {
            CUDA_CALL(cudaStreamSynchronize(it->first));
        }
    }
};

const int StreamManager::max_num_of_streams = 10;
int StreamManager::curr_stream_index = 0;
std::unordered_map<cudaStream_t, cublasHandle_t> StreamManager::handles = std::unordered_map<cudaStream_t, cublasHandle_t>();


#endif //MOOREPENROSE_STREAM_MANAGER_H
