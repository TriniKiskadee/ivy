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
def quaternion_to_axisangle(quaternion):
    return ivy.quaternion_to_axisangle(
        quaternion
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def normalize_quaternion(quaternion, *, eps=1e-12):
    return ivy.normalize_quaternion(
        quaternion, eps=eps
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def quaternion_to_rotation_matrix(quaternion):
    return ivy.quaternion_to_rotation_matrix(
        quaternion
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def inverse_transform_3d(trans):
    return ivy.inverse_transform_3d(
        trans
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def compose_transforms_3d(trans1, trans2):
    return ivy.compose_transforms_3d(
        trans1, trans2
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def transform_pts_3d(pts_b, t_ab):
    return ivy.transform_pts_3d(
        pts_b, t_ab
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def transform_pts_nd_KF(pts, tform):
    return ivy.transform_pts_nd_KF(
        pts, tform
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def relative_transform_3d(trans_01, trans_02):
    return ivy.relative_transform_3d(
        trans_01, trans_02
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def relative_transformation(trans_01, trans_02, *, orthogonal_rotations=False):
    return ivy.relative_transformation(
        trans_01, trans_02, orthogonal_rotations=orthogonal_rotations
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def normalize_pixel_coords(pixel_coords, height, width):
    return ivy.normalize_pixel_coords(
        pixel_coords, height, width
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def unnormalize_pixel_coords(pixel_coords_norm, height, width):
    return ivy.unnormalize_pixel_coords(
        pixel_coords_norm, height, width
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def create_meshgrid(height, width, *, normalized_coords=True):
    return ivy.create_meshgrid(
        height, width, normalized_coords=normalized_coords
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def cam2pixel(cam_coords_src, dst_proj_src, *, eps=1e-6):
    return ivy.cam2pixel(
        cam_coords_src, dst_proj_src, eps=eps
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def pixel2cam(depth, intrinsics_inv, pixel_coords):
    return ivy.pixel2cam(
        depth, intrinsics_inv, pixel_coords
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def cam2pixel_KF(cam_coords_src, P, *, eps=1e-6):
    return ivy.cam2pixel_KF(
        cam_coords_src, P, eps=eps
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def transform_pointcloud(pointcloud, transform):
    return ivy.transform_pointcloud(
        pointcloud, transform
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def transform_normals(normals, transform):
    return ivy.transform_normals(
        normals, transform
    )
