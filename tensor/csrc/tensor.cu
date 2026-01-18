#include "tensor.h"
#include <cuda_runtime.h>

void tensor_allocate(Tensor* t, int rows, int cols) {
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->size = rows * cols;

    cudaMalloc(&t->data, t->size * sizeof(float));
}

void tensor_free(Tensor* t) {
    if (t->data) {
        cudaFree(t->data);
        t->data = nullptr;
    }
    if (t->grad) {
        cudaFree(t->grad);
        t->grad = nullptr;
    }
    t->size = 0;
    t->shape[0] = 0;
    t->shape[1] = 0;
}

void tensor_to_gpu(Tensor* t, const float* cpu_data) {
    cudaMemcpy(t->data, cpu_data, t->size * sizeof(float), cudaMemcpyHostToDevice);
}

void tensor_to_cpu(const Tensor* t, float* cpu_data) {
    cudaMemcpy(cpu_data, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
}
