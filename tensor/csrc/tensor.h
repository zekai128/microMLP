#pragma once
#include <cstddef>

struct Tensor {
    float* data;      // Pointer to GPU memory
    float* grad;      // Pointer to gradient (GPU memory)
    size_t size;      // Total number of elements
    int shape[2];     // Shape (rows, cols) - 2D only for simplicity

    // Constructor
    Tensor() : data(nullptr), grad(nullptr), size(0) {
        shape[0] = 0;
        shape[1] = 0;
    }
};

// CUDA memory operations (implemented in tensor.cu)
void tensor_allocate(Tensor* t, int rows, int cols);
void tensor_free(Tensor* t);
void tensor_to_gpu(Tensor* t, const float* cpu_data);
void tensor_to_cpu(const Tensor* t, float* cpu_data);
