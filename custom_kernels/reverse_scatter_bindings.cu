#include <torch/extension.h>

__global__ void forward_kernel(const float* __restrict__ input,
                               const int64_t* __restrict__ mapping,
                               float* __restrict__ output,
                               int64_t N,
                               int64_t D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    if (i < N) {
        int idx = mapping[i];
        for (int d = 0; d < D; ++d) {
            output[i * D + d] += input[idx * D + d];
        }
    }
}
// Declaration of your CUDA kernel
__global__ void backward_kernel(const float* __restrict__ grad_output,
                                const int64_t* __restrict__ mapping,
                                float* __restrict__ grad_input,
                                int64_t N,
                                int64_t D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    if (i < N) {
        int idx = mapping[i];
        for (int d = 0; d < D; ++d) {
            atomicAdd(&grad_input[idx * D + d], grad_output[i * D + d]);
        }
    }
}

torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor mapping, torch::Tensor output) {
    const auto N = mapping.size(0);
    const auto D = input.size(1);
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mapping.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

torch::Tensor backward_cuda(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows) {
    const auto N = mapping.size(0);
    const auto D = grad_output.size(1);
    auto grad_input = torch::zeros({input_rows, D}, grad_output.options());

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        mapping.data_ptr<int64_t>(),
        grad_input.data_ptr<float>(),
        N,
        D
    );

    return grad_input;
}

// --- CUDA declarations ---
torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor mapping, torch::Tensor output);
torch::Tensor backward_cuda(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows);

// --- Dispatcher-visible functions ---
torch::Tensor reverse_scatter_forward(torch::Tensor input, torch::Tensor mapping, torch::Tensor output) {
    return forward_cuda(input, mapping, output);
}

torch::Tensor reverse_scatter_backward(torch::Tensor grad_output, torch::Tensor mapping, int64_t input_rows) {
    return backward_cuda(grad_output, mapping, input_rows);
}

// --- Operator registration ---
TORCH_LIBRARY(reverse_scatter, m) {
    m.def("forward", reverse_scatter_forward);
    m.def("backward", reverse_scatter_backward);
}