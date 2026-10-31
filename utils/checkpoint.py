import torch


def checkpoint(function, inputs, parameters, enabled):
    if not enabled:
        return function(*inputs)
    args = tuple(inputs) + tuple(parameters)
    return CheckpointFunction.apply(function, len(inputs), *args)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, input_count, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:input_count])
        ctx.input_parameters = list(args[input_count:])
        with torch.no_grad():
            return ctx.run_function(*ctx.input_tensors)

    @staticmethod
    def backward(ctx, *output_gradients):
        ctx.input_tensors = [
            tensor.detach().requires_grad_(True) for tensor in ctx.input_tensors
        ]
        with torch.enable_grad():
            shallow_copies = [tensor.view_as(tensor) for tensor in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_gradients = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_parameters,
            output_gradients,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_parameters
        del output_tensors
        return (None, None) + input_gradients
