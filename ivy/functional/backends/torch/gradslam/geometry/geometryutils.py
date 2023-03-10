""" Projective geometry utility functions. """

from typing import Optional

import torch
from kornia.geometry.linalg import compose_transformations, inverse_transformation


def homogenize_points(
    pts: torch.Tensor
):
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    return torch.nn.functional.pad(pts, (0, 1), "constant", 1.0)


def unhomogenize_points(
    pts: torch.Tensor,
    /,
    *,
    eps: float = 1e-6
) -> torch.Tensor:
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    # Get points with the last coordinate (scale) as 0 (points at infinity)
    w: torch.Tensor = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale: torch.Tensor = torch.where(torch.abs(w) > eps, 1.0 / w, torch.ones_like(w))

    return scale * pts[..., :-1]


def quaternion_to_axisangle(
    quat: torch.Tensor
) -> torch.Tensor:
    if not torch.is_tensor(quat):
        raise TypeError(
            "Expected input quat to be of type torch.Tensor."
            " Got {} instead.".format(type(quat))
        )
    if not quat.shape[-1] == 4:
        raise ValueError(
            "Last dim of input quat must be of shape 4. "
            "Got {} instead.".format(quat.shape[-1])
        )

    # Unpack quat
    qx: torch.Tensor = quat[..., 0]
    qy: torch.Tensor = quat[..., 1]
    qz: torch.Tensor = quat[..., 2]
    sin_sq_theta: torch.Tensor = qx * qx + qy * qy + qz * qz
    sin_theta: torch.Tensor = torch.sqrt(sin_sq_theta)
    cos_theta: torch.Tensor = quat[..., 3]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_sq_theta > 0.0, k_pos, k_neg)

    axisangle: torch.Tensor = torch.zeros_like(quat)[..., :3]
    axisangle[..., 0] = qx * k
    axisangle[..., 1] = qy * k
    axisangle[..., 2] = qz * k

    return axisangle


def normalize_quaternion(
    quaternion: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}.".format(quaternion.shape)
        )
    return torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor
) -> torch.Tensor:
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )

    # Normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)

    # Unpack the components of the normalized quaternion
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # Compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def inverse_transform_3d(
    trans: torch.Tensor
):
    if not torch.is_tensor(trans):
        raise TypeError(
            "Expected input trans of type torch.Tensor. Got {} instead.".format(
                type(trans)
            )
        )
    if not trans.dim() in (2, 3) and trans.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(trans.shape)
        )

    # Unpack tensor into rotation and tranlation components
    rmat: torch.Tensor = trans[..., :3, :3]
    tvec: torch.Tensor = trans[..., :3, 3]

    # Compute the inverse
    rmat_inv: torch.Tensor = torch.transpose(rmat, -1, -2)
    tvec_inv: torch.Tensor = torch.matmul(-rmat_inv, tvec)

    # Pack the inverse rotation and translation into tensor
    trans_inv: torch.Tensor = torch.zeros_like(trans)
    trans_inv[..., :3, :3] = rmat_inv
    trans_inv[..., :3, 3] = tvec_inv
    trans_inv[..., -1, -1] = 1.0

    return trans_inv


def compose_transforms_3d(
    trans1: torch.Tensor,
    trans2: torch.Tensor
) -> torch.Tensor:
    if not torch.is_tensor(trans1):
        raise TypeError(
            "Expected input trans1 of type torch.Tensor. Got {} instead.".format(
                type(trans1)
            )
        )
    if not trans1.dim() in (2, 3) and trans1.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans1.shape
            )
        )
    if not torch.is_tensor(trans2):
        raise TypeError(
            "Expected input trans2 of type torch.Tensor. Got {} instead.".format(
                type(trans2)
            )
        )
    if not trans2.dim() in (2, 3) and trans2.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans2.shape
            )
        )
    assert (
        trans1.shape == trans2.shape
    ), "Both input transformations must have the same shape."

    # Unpack into rmat, tvec
    rmat1: torch.Tensor = trans1[..., :3, :3]
    rmat2: torch.Tensor = trans2[..., :3, :3]
    tvec1: torch.Tensor = trans1[..., :3, 3]
    tvec2: torch.Tensor = trans2[..., :3, 3]

    # Compute the composition
    rmat_cat: torch.Tensor = torch.matmul(rmat1, rmat2)
    tvec_cat: torch.Tensor = torch.matmul(rmat1, tvec2) + tvec1

    # Pack into output tensor
    trans_cat: torch.Tensor = torch.zeros_like(trans1)
    trans_cat[..., :3, :3] = rmat_cat
    trans_cat[..., :3, 3] = tvec_cat
    trans_cat[..., -1, -1] = 1.0

    return trans_cat


def transform_pts_3d(
    pts_b: torch.Tensor,
    t_ab: torch.Tensor
) -> torch.Tensor:
    if not torch.is_tensor(pts_b):
        raise TypeError(
            "Expected input pts_b of type torch.Tensor. Got {} instead.".format(
                type(pts_b)
            )
        )
    if not torch.is_tensor(t_ab):
        raise TypeError(
            "Expected input t_ab of type torch.Tensor. Got {} instead.".format(
                type(t_ab)
            )
        )
    if pts_b.dim() < 2:
        raise ValueError(
            "Expected pts_b to have at least 2 dimensions. Got {} instead.".format(
                pts_b.dim()
            )
        )
    if t_ab.dim() != 2:
        raise ValueError(
            "Expected t_ab to have 2 dimensions. Got {} instead.".format(t_ab.dim())
        )
    if t_ab.shape[0] != 4 or t_ab.shape[1] != 4:
        raise ValueError(
            "Expected t_ab to have shape (4, 4). Got {} instead.".format(t_ab.shape)
        )

    # Determine if we need to homogenize the points
    if pts_b.shape[-1] == 3:
        pts_b = homogenize_points(pts_b)

    # Apply the transformation

    if pts_b.dim() == 4:
        pts_a_homo = torch.matmul(
            t_ab.unsqueeze(0).unsqueeze(0), pts_b.unsqueeze(-1)
        ).squeeze(-1)
    else:
        pts_a_homo = torch.matmul(t_ab.unsqueeze(0), pts_b.unsqueeze(-1))
    pts_a = unhomogenize_points(pts_a_homo)

    return pts_a[..., :3]


def transform_pts_nd_KF(
    pts,
    tform
):
    if not pts.shape[0] == tform.shape[0]:
        raise ValueError("Input batchsize must be the same for both  tensors")
    if not pts.shape[-1] + 1 == tform.shape[-1]:
        raise ValueError(
            "Last input dims must differ by one, i.e., "
            "pts.shape[-1] + 1 should be equal to tform.shape[-1]."
        )

    # Homogenize
    pts_homo = homogenize_points(pts)

    # Transform
    pts_homo_tformed = torch.matmul(tform.unsqueeze(1), pts_homo.unsqueeze(-1))
    pts_homo_tformed = pts_homo_tformed.squeeze(-1)

    # Unhomogenize
    return unhomogenize_points(pts_homo_tformed)


def relative_transform_3d(
    trans_01: torch.Tensor,
    trans_02: torch.Tensor
) -> torch.Tensor:
    return compose_transforms_3d(inverse_transform_3d(trans_01), trans_02)


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


def normalize_pixel_coords(
    pixel_coords: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    if not torch.is_tensor(pixel_coords):
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    if pixel_coords.shape[-1] != 2:
        raise ValueError(
            "Expected last dimension of pixel_coords to be of size 2. Got {} instead.".format(
                pixel_coords.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."

    dtype = pixel_coords.dtype
    device = pixel_coords.device

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device)

    norm: torch.Tensor = 2.0 / (wh - 1)

    return norm[:, 0] * pixel_coords - 1


def unnormalize_pixel_coords(
    pixel_coords_norm: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    if not torch.is_tensor(pixel_coords_norm):
        raise TypeError(
            "Expected pixel_coords_norm to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords_norm)
            )
        )
    if pixel_coords_norm.shape[-1] != 2:
        raise ValueError(
            "Expected last dim of pixel_coords_norm to be of shape 2. Got {} instead.".format(
                pixel_coords_norm.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."

    dtype = pixel_coords_norm.dtype
    device = pixel_coords_norm.device

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device)

    norm: torch.Tensor = 2.0 / (wh - 1)

    return 1.0 / norm[:, 0] * (pixel_coords_norm + 1)


def create_meshgrid(
    height: int, width: int, normalized_coords: Optional[bool] = True
) -> torch.Tensor:
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coords:
        xs = torch.linspace(-1, 1, height)
        ys = torch.linspace(-1, 1, width)
    else:
        xs = torch.linspace(0, height - 1, height)
        ys = torch.linspace(0, width - 1, width)
    # Generate grid (2 x H x W)
    base_grid: torch.Tensor = torch.stack((torch.meshgrid([xs, ys])))

    return base_grid.permute(1, 2, 0).unsqueeze(0)  # 1 xH x W x 2


def cam2pixel(
    cam_coords_src: torch.Tensor,
    dst_proj_src: torch.Tensor,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    assert cam_coords_src.dim() in (3, 4), "cam_coords_src must have 3 or 4 dimensions."
    assert cam_coords_src.shape[-1] == 3
    assert torch.is_tensor(dst_proj_src), "dst_proj_src must be of type torch.Tensor."
    assert (
        dst_proj_src.dim() == 2
        and dst_proj_src.shape[0] == 4
        and dst_proj_src.shape[0] == 4
    )

    _, h, w, _ = cam_coords_src.shape
    pts: torch.Tensor = transform_pts_3d(cam_coords_src, dst_proj_src)
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack([u, v], dim=-1)


def pixel2cam(
    depth: torch.Tensor, intrinsics_inv: torch.Tensor, pixel_coords: torch.Tensor
) -> torch.Tensor:
    if not torch.is_tensor(depth):
        raise TypeError(
            "Expected depth to be of type torch.Tensor. Got {} instead.".format(
                type(depth)
            )
        )
    if not torch.is_tensor(intrinsics_inv):
        raise TypeError(
            "Expected intrinsics_inv to be of type torch.Tensor. Got {} instead.".format(
                type(intrinsics_inv)
            )
        )
    if not torch.is_tensor(pixel_coords):
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    assert (
        intrinsics_inv.shape[0] == 4
        and intrinsics_inv.shape[1] == 4
        and intrinsics_inv.dim() == 2
    )

    cam_coords: torch.Tensor = transform_pts_3d(
        pixel_coords, intrinsics_inv
    )  # .permute(0, 3, 1, 2)

    return cam_coords * depth.permute(0, 2, 3, 1)


def cam2pixel_KF(
    cam_coords_src: torch.Tensor, P: torch.Tensor, eps: Optional[float] = 1e-6
) -> torch.Tensor:
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    # assert cam_coords_src.dim() > 3, 'cam_coords_src must have > 3 dimensions.'
    assert cam_coords_src.shape[-1] == 3
    assert torch.is_tensor(P), "dst_proj_src must be of type torch.Tensor."
    assert P.dim() >= 2 and P.shape[-1] == 4 and P.shape[-2] == 4

    pts: torch.Tensor = transform_pts_nd_KF(cam_coords_src, P)
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack([u, v], dim=-1)


def transform_pointcloud(pointcloud: torch.Tensor, transform: torch.Tensor):
    if not torch.is_tensor(pointcloud):
        raise TypeError(
            "pointcloud should be tensor, but was %r instead" % type(pointcloud)
        )

    if not torch.is_tensor(transform):
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not pointcloud.ndim == 2:
        raise ValueError(
            "pointcloud should have ndim of 2, but had {} instead.".format(
                pointcloud.ndim
            )
        )
    if not pointcloud.shape[1] == 3:
        raise ValueError(
            "pointcloud.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                pointcloud.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation matrix
    rmat = transform[:3, :3]
    # Translation vector
    tvec = transform[:3, 3]

    # Transpose the pointcloud (to enable broadcast of rotation to each point)
    transposed_pointcloud = torch.transpose(pointcloud, 0, 1)
    # Rotate and translate cloud
    transformed_pointcloud = torch.matmul(rmat, transposed_pointcloud) + tvec.unsqueeze(
        1
    )
    # Transpose the transformed cloud to original dimensions
    transformed_pointcloud = torch.transpose(transformed_pointcloud, 0, 1)

    return transformed_pointcloud


def transform_normals(normals: torch.Tensor, transform: torch.Tensor):
    if not torch.is_tensor(normals):
        raise TypeError("normals should be tensor, but was %r instead" % type(normals))

    if not torch.is_tensor(transform):
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not normals.ndim == 2:
        raise ValueError(
            "normals should have ndim of 2, but had {} instead.".format(normals.ndim)
        )
    if not normals.shape[1] == 3:
        raise ValueError(
            "normals.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                normals.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation
    R = transform[:3, :3]

    # apply transpose to normals
    transposed_normals = torch.transpose(normals, 0, 1)

    # transpose after transform
    transformed_normals = torch.transpose(torch.matmul(R, transposed_normals), 0, 1)

    return transformed_normals
