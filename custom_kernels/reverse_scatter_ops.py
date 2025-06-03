from torch.utils.cpp_extension import load
from torch.library import Library

# Load CUDA extension
cuda_module = load(
    name="reverse_scatter",
    sources=["custom_kernels/reverse_scatter.cpp", "custom_kernels/reverse_scatter.cu"],
    extra_cflags=["-O3"],
    verbose=True
)

# Step 1: Register ops with torch.library
lib = Library("my_ops", "DEF")

# Define forward and backward symbols
lib.define("reverse_scatter(Tensor input, Tensor mapping, Tensor output) -> Tensor")
lib.define("reverse_scatter_backward(Tensor grad_output, Tensor mapping, int input_rows) -> Tensor")

# Correctly pass CUDA functions as callables
import torch

def reverse_scatter_forward(input, mapping, output):
    # Ensure all tensors are on the same CUDA device
    device = input.device
    mapping = mapping.to(device)
    output = output.to(device)
    return cuda_module.forward(input, mapping, output)

def reverse_scatter_backward(grad_output, mapping, input_rows):
    # Ensure all tensors are on the same CUDA device
    device = grad_output.device
    mapping = mapping.to(device)
    return cuda_module.backward(grad_output, mapping, input_rows)

# Register CUDA implementations
lib.impl("reverse_scatter", reverse_scatter_forward, "CUDA")
lib.impl("reverse_scatter_backward", reverse_scatter_backward, "CUDA")