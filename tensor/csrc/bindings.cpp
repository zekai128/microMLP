#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;

// Wrapper class to make Tensor easier to use from Python
class PyTensor {
public:
    Tensor t;

    PyTensor() {}

    void allocate(int rows, int cols) {
        tensor_allocate(&t, rows, cols);
    }

    void free() {
        tensor_free(&t);
    }

    void to_gpu(py::array_t<float> data) {
        py::buffer_info buf = data.request();
        float* ptr = static_cast<float*>(buf.ptr);
        tensor_to_gpu(&t, ptr);
    }

    py::array_t<float> to_cpu() {
        // Create numpy array to hold result
        auto result = py::array_t<float>(t.size);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        tensor_to_cpu(&t, ptr);
        return result;
    }

    size_t size() const { return t.size; }
    int rows() const { return t.shape[0]; }
    int cols() const { return t.shape[1]; }
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "tensor C++ backend";

    py::class_<PyTensor>(m, "Tensor")
        .def(py::init<>())
        .def("allocate", &PyTensor::allocate, "Allocate GPU memory")
        .def("free", &PyTensor::free, "Free GPU memory")
        .def("to_gpu", &PyTensor::to_gpu, "Copy data from CPU to GPU")
        .def("to_cpu", &PyTensor::to_cpu, "Copy data from GPU to CPU")
        .def("size", &PyTensor::size, "Get total number of elements")
        .def("rows", &PyTensor::rows, "Get number of rows")
        .def("cols", &PyTensor::cols, "Get number of cols");
}
