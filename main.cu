#include <iostream>
#include <curand_kernel.h>
#include "utils/macros.h"
#include "matrix.h"
#include "utils/stream_manager.h"
#include "utils/remove_zero_rows.h"
#include "moore_penrose.h"
#include <opencv4/opencv2/opencv.hpp>
#include "utils/file_utils.h"
#include <chrono>
#include "pinv_cpu.h"

template<typename T>
void test_gpu_moore_penrose(unsigned int n, unsigned int m) {
    StreamManager::init();

    gpu_implementation::Matrix<T> A(n, m);
    A.allocate_memory(InitMode::RANDOM);

    CUDA_CALL(cudaDeviceSynchronize());
    auto start_time = std::chrono::high_resolution_clock::now();
    gpu_implementation::Matrix<T> MP = gpu_implementation::calculate_moore_penrose_pseudo_inverse<T>(A);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    A.free_memory();
    MP.free_memory();

    std::string type = "float";
    if (std::is_same<T, double>::value)
        type = "double";
    std::string processor = "GPU";
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout << "Computation time: " << std::setw(12) << millis << std::endl;

    std::string algo = USE_SLOW_MATMUL? "Recursive(slow)" : "Recursive(fast)";
    write_results(n, m, algo, processor, type, millis);

    CUDA_CALL(cudaDeviceReset());
}



template<typename T, bool recursive = true>
void test_cpu_moore_penrose(unsigned int n, unsigned int m) {
    std::string processor = "CPU";
    std::string type = "float";
    std::string algo = recursive? "Recursive" : "SVD-based";

    std::chrono::duration<long, std::ratio<1,1000000000>> time;

    if (std::is_same<T, double>::value)
        type = "double";

    if (recursive) {
        cpu_implementation::Matrix<T> A(n, m);
        A.allocate_memory(InitMode::RANDOM);

        CUDA_CALL(cudaDeviceSynchronize());
        auto start_time = std::chrono::high_resolution_clock::now();
        cpu_implementation::Matrix<T> MP = cpu_implementation::calculate_moore_penrose_pseudo_inverse<T>(A);
        CUDA_CALL(cudaDeviceSynchronize());
        auto end_time = std::chrono::high_resolution_clock::now();
        time = end_time - start_time;

        A.free_memory();
        MP.free_memory();

    } else {
        cv::Mat B;
        cv::Mat Bpinv;

        if (std::is_same<T, float>::value) {
            B = cv::Mat(n, m, CV_32FC1);
        } else {
            B = cv::Mat(n, m, CV_64FC1);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        invert(B, Bpinv, cv::DECOMP_SVD);
        auto end_time = std::chrono::high_resolution_clock::now();
        time = end_time - start_time;
    }

    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout<< "Computation time: " << millis << " ms" << std::endl;
    write_results(n, m, algo, processor, type, millis);

    CUDA_CALL(cudaDeviceReset());
}

int main(int argc, char **argv) {

    int n = 3072 * 2;
    int m = 2048 * 2;

    char* processor = "GPU";
    char* type = "float";
    bool recursive = true;

    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        m = atoi(argv[2]);
    }
    if (argc > 3) {
        if ((strcmp(argv[3], "GPU") != 0) && (strcmp(argv[3], "CPU") != 0)) {
            std::string message = "Undefined processor type specified. Supported values are \"GPU\" and \"CPU\". Using the default\"GPU\" processor";
            std::cout << message << std::endl;
        } else
            processor = argv[3];
    }
    if (argc > 4) {
        if ((strcmp(argv[4], "float") != 0) && (strcmp(argv[4], "double") != 0)) {
            std::string message2 = "Undefined data type specified. Supported values are \"float\" and \"double\". Using the default\"float\" data type";
            std::cout << message2 << std::endl;
        } else
            type = argv[4];
    }
    if (argc > 5) {
        int r = atoi(argv[5]);
        recursive = r == 1;
        std::cout << "Command line args: " << n << " " << m << " " << processor << " " << type << " " << r << std::endl << std::endl;
    }

    if (strcmp(processor, "GPU") == 0) {
        if (!recursive) exit(0);
        if (strcmp(type, "float") == 0) {
            test_gpu_moore_penrose<float>(n, m);
        } else {
            test_gpu_moore_penrose<double>(n, m);
        }
    } else {
        if (strcmp(type, "float") == 0) {
            if (recursive)
                test_cpu_moore_penrose<float, true>(n, m);
            else
                test_cpu_moore_penrose<float, false>(n, m);
        } else {
            if (recursive)
                test_cpu_moore_penrose<double, true>(n, m);
            else
                test_cpu_moore_penrose<double, false>(n, m);
        }
    }

    return 0;
}
