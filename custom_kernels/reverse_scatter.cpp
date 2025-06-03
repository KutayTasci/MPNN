#include <torch/extension.h>

// Declare your CUDA kernels
torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor mapping, torch::Tensor output);
torch::Tensor backward_cuda(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows);

// Define the operator implementations
torch::Tensor reverse_scatter_forward(torch::Tensor input, torch::Tensor mapping, torch::Tensor output) {
    return forward_cuda(input, mapping, output);
}

torch::Tensor reverse_scatter_backward(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows) {
    return backward_cuda(grad_output, mapping, input_rows);
}

// Register the operators with Torch dispatcher
TORCH_LIBRARY(reverse_scatter, m) {
    m.def("forward", reverse_scatter_forward);
    m.def("backward", reverse_scatter_backward);
}