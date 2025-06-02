
import experiments.test_experiment as test_exp
import logging
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


test_exp.test_egnn(benchmark=True, model_type='cat', batch_size=32)
#test_exp.test_egnn(benchmark=True, model_type='sum', batch_size=2048)

#test_exp.test_egnn(benchmark=True, model_type='sum', batch_size=2)
#test_exp.test_egnn(benchmark=True, model_type='cat', batch_size=2)


"""
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

cuda_module = load(name="reverse_scatter",
                        sources=["custom_kernels/reverse_scatter.cpp", "custom_kernels/reverse_scatter.cu"])


class ReverseScatter(Function):
    @staticmethod
    def forward(ctx, input, mapping, output):
        ctx.save_for_backward(mapping)
        ctx.input_size = input.size(0)  # Save the input size for use in backward
        return cuda_module.forward(input.contiguous(), mapping, output.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        (mapping,) = ctx.saved_tensors

        input_size = ctx.input_size
        grad_input = cuda_module.backward(grad_output.contiguous(), mapping,input_size)
        return grad_input, None, None  # grad w.r.t. input only

# Alias for easier use
reverse_scatter_fn = ReverseScatter.apply

# Testing the custom kernel
def test_reverse_scatter():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define input parameters
    input_rows = 3
    output_rows = 5
    feature_dim = 4

    # Create input tensor
    input_tensor = torch.randn(input_rows, feature_dim, device=device, requires_grad=True)

    # Define mapping from output rows to input rows
    mapping = torch.tensor([0, 1, 2, 1, 0], device=device, dtype=torch.int64)

    # Initialize output tensor
    output_tensor = torch.zeros(output_rows, feature_dim, device=device)

    # Forward pass
    output = reverse_scatter_fn(input_tensor, mapping, output_tensor.clone())

    # Define a simple loss (sum of all elements)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Print results
    print("Input Tensor:")
    print(input_tensor)

    print("\nMapping:")
    print(mapping)

    print("\nOutput Tensor after Forward Pass:")
    print(output)

    print("\nGradient w.r.t. Input Tensor:")
    print(input_tensor.grad)

    # Verification
    # Manually compute expected output
    expected_output = torch.zeros_like(output_tensor)
    for i in range(output_rows):
        expected_output[i] = input_tensor[mapping[i]]
    assert torch.allclose(output, expected_output), "Forward pass output mismatch."

    # Manually compute expected gradients
    expected_grad = torch.zeros_like(input_tensor)
    for i in range(output_rows):
        expected_grad[mapping[i]] += 1.0  # Since loss is sum, gradient is 1 for each output element

    print("\nExpected Gradient:")
    print(expected_grad)
    assert torch.allclose(input_tensor.grad, expected_grad), "Backward pass gradient mismatch."

    print("\nTest passed successfully.")

if __name__ == "__main__":
    test_reverse_scatter()
"""