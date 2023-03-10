# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def homogenize_points(pts):
    return ivy.homogenize_points(
        pts
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def unhomogenize_points(pts, *, eps=1e-6):
    return ivy.unhomogenize_points(
        pts, eps=eps
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def project_points(cam_coords, proj_mat, *, eps=1e-6):
    return ivy.project_points(
        cam_coords, proj_mat, eps=eps
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def unproject_points(pixel_coords, intrinsics_inv, depths):
    return ivy.unproject_points(
        pixel_coords, intrinsics_inv, depths
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def inverse_intrinsics(K, eps=1e-6):
    return ivy.inverse_intrinsics(
        K, eps=eps
    )