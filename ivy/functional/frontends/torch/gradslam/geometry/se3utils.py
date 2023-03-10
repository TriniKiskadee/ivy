# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def so3_hat(omega):
    return ivy.so3_hat(
        omega
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def se3_hat(xi):
    return ivy.se3_hat(
        xi
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def so3_exp(omega):
    return ivy.so3_exp(
        omega
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def se3_exp(xi):
    return ivy.se3_exp(
        xi
    )
