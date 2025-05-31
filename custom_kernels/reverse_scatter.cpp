#include <torch/extension.h>

torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor mapping, torch::Tensor output);
torch::Tensor backward_cuda(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows);

torch::Tensor forward(torch::Tensor input, torch::Tensor mapping, torch::Tensor output) {
    return forward_cuda(input, mapping, output);
}

torch::Tensor backward(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows) {
    return backward_cuda(grad_output, mapping, input_rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom forward");
    m.def("backward", &backward, "Custom backward");
}