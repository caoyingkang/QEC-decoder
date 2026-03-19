import torch
import torch.nn.functional as F


def smooth_sign(x: torch.Tensor, *, alpha: float = 100.0) -> torch.Tensor:
    """
    Smooth version of sign function. Larger `alpha` => better approximation.
    """
    return torch.tanh(alpha * x)


def smooth_min(x: torch.Tensor, *, dim: int, temp: float = 0.01) -> torch.Tensor:
    """
    Smooth version of min function along a given dimension `dim`. Smaller `temp` => better approximation.
    """
    return torch.sum(x * F.softmin(x / temp, dim=dim), dim=dim)


def matmul_GF2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication over GF(2).

    Assume that the input tensors take binary values (0 or 1). The returned dtype is 
    always `torch.int32` no matter what the input dtypes are. If the inputs are integer 
    or boolean tensors, they will be converted to float tensors before applying the 
    `@` operator. This is usually more efficient than using the `@` operator directly 
    (even on CPU). Moreover, some CUDA devices do not support integer matmul.

    See pytorch/notebooks/benchmark_matmul_GF2.ipynb for more details.
    """
    return (x.float() @ y.float()).round().int() % 2


def leave_one_out_sign_product(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    For each element of `x` along dimension `dim`, compute the product of signs of all other 
    elements along that dimension:

    `output[..., i, ...] = prod(sign(x[..., j, ...]) for j != i)`.

    Note that the returned tensor has dtype `torch.float32` and no gradients can be 
    back-propagated through it.
    """
    # Different from `torch.sign()` function, here we never get 0.0 in the output.
    x_sgn = torch.where(x < 0, -1.0, 1.0)
    return x_sgn.prod(dim, keepdim=True) * x_sgn


def leave_one_out_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    For each element of `x` along dimension `dim`, compute the minimum value of all other 
    elements along that dimension:

    `output[..., i, ...] = min(x[..., j, ...] for j != i)`.
    """
    values, indices = x.topk(2, dim=dim, largest=False)
    min1, min2 = values.split(1, dim=dim)
    ind1 = indices.narrow(dim, 0, 1)
    return min1.expand_as(x).clone().scatter_(dim, ind1, min2)
