""" Projective geometry utility functions. """

from typing import Optional, Union

import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
    handle_array_like_without_promotion,
)
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def homogenize_points(
        pts: Union[ivy.Array, ivy.NativeArray]
):
    """
    Convert a set of points to homogeneous coordinates.

    Parameters
    ----------
    pts
        ivy.Array containing points to be homogenized.
        :math:`(N, *, K)` where :math:`N` indicates the number of points in a
        cloud if the shape is :math:`(N, K)` and indicates batchsize if the number
        of dimensions is greater than 2. Also, :math:`*` means any number of
        additional dimensions, and `K` is the dimensionality of each point.

    Returns
    -------
    ret
        ivy.Array of Homogeneous coordinates of `pts`.
        :math:`(N, *, K + 1)` where all but the last dimension are the same shape
        as `pts`.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> pts = ivy.array.rand(10, 3)
    >>> pts_homo = homogenize_points(pts)
    >>> pts_homo.shape
        torch.Size([10, 4])
    """
    if not isinstance(pts, ivy.Array):
        raise ivy.exceptions.IvyException(
            "Expected input type ivy.Array. Got {} instead".format(type(pts))
        )
    if pts.dim() < 2:
        raise ivy.exceptions.IvyException(
            "Input tensor must have at least 2 dimensions. Got {} instead.".format(
                pts.dim()
            )
        )

    pad(pts, (0, 1), "constant", 1.0)
    return pad(pts, (0, 1), "constant", 1.0)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def unhomogenize_points(
        pts: Union[ivy.Array, ivy.NativeArray],
        eps: Optional[float] = 1e-6
) -> ivy.Array:
    """Convert a set of points from homogeneous coordinates to Euclidean
    coordinates. This is usually done by taking each point :math:`(X, Y, Z, W)`
    and dividing it by the last coordinate :math:`(w)`.

    Parameters
    ----------
    pts
        Tensor containing points to be unhomogenized.
        :math:`(N, *, K)` where :math:`N` indicates the number of points in a
        cloud if the shape is :math:`(N, K)` and indicates batchsize if the number
        of dimensions is greater than 2. Also, :math:`*` means any number of
        additional dimensions, and `K` is the dimensionality of each point.

    eps
        float

    Returns
    -------
    ret
        a tensor of 'Unhomogenized' points :math:`(N, *, K-1)` where all but the
        last dimension are the same shape as `pts`.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> pts = ivy.rand(10, 3)
    >>> pts_unhomo = unhomogenize_points(x)
    >>> pts_unhomo.shape
    torch.Size([10, 2])
    """
    if not isinstance(pts, ivy.Array):
        raise ivy.exceptions.IvyException(
            "Expected input type ivy.Array. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ivy.exceptions.IvyException(
            "Input tensor must have at least 2 dimensions. Got {} instead.".format(
                pts.dim()
            )
        )

    # Get points with the last coordinate (scale) as 0 (points at infinity).
    w: ivy.Array = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale: ivy.Array = ivy.where(ivy.abs(w) > eps, 1.0 / w, ivy.ones_like(w))
    unhomo_pts = scale * pts[..., :-1]

    return current_backend(pts).unhomo_pts


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def project_points(
    cam_coords: Union[ivy.Array, ivy.NativeArray],
    proj_mat: Union[ivy.Array, ivy.NativeArray],
    eps: Optional[float] = 1e-6
) -> ivy.Array:
    """
    Projects points from the camera coordinate frame to the image (pixel) frame.
    Args:
        cam_coords (torch.Tensor): pixel coordinates (defined in the
            frame of the first camera).
        proj_mat (torch.Tensor): projection matrix between the reference
            and the non-reference camera frame.
    Returns:
        torch.Tensor: Image (pixel) coordinates corresponding to the input 3D points.
    Shapes:
        - cam_coords: :math:`(N, *, 3)` or :math:`(*, 4)` where :math:`*` indicates an arbitrary number of dimensions.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - proj_mat: :math:`(*, 4, 4)` where :math:`*` indicates an arbitrary number of dimensions.
          dimension contains a :math:`(4, 4)` camera projection matrix.
        - Output: :math:`(N, *, 2)`, where :math:`*` indicates the same dimensions as in `cam_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
    Examples::
        >>> # Case 1: Input cam_coords are homogeneous, no batchsize dimension.
        >>> cam_coords = torch.rand(10, 4)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([10, 2])
        >>> # Case 2: Input cam_coords are homogeneous and batched. Broadcast proj_mat across batch.
        >>> cam_coords = torch.rand(2, 10, 4)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 3: Input cam_coords are homogeneous and batched. A different proj_mat applied to each element.
        >>> cam_coords = torch.rand(2, 10, 4)
        >>> proj_mat = torch.rand(2, 4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 4: Similar to case 1, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(10, 3)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([10, 2])
        >>> # Case 5: Similar to case 2, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(2, 10, 3)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 6: Similar to case 3, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(2, 10, 3)
        >>> proj_mat = torch.rand(2, 4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
    """
    # Based on
    # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43
    # and Kornia.
    if not ivy.is_ivy_array(cam_coords):
        raise TypeError(
            "Expected input cam_coords to be of type ivy.Array. Got {0} instead.".format(
                type(cam_coords)
            )
        )

    if not ivy.is_ivy_array(cam_coords):
        raise ivy.exceptions.IvyException(
            "Expected input cam_coords to be of type ivy.Array. Got {0} instead.".format(
                type(cam_coords)
            )
        )

    if not ivy.is_ivy_array(proj_mat):
        raise ivy.exceptions.IvyException(
            "Expected input proj_mat to be of type ivy.is_ivy_array. Got {0} instead.".format(
                type(proj_mat)
            )
        )
    if cam_coords.dim() < 2:
        raise ivy.exceptions.IvyException(
            "Input cam_coords must have at least 2 dims. Got {0} instead.".format(
                cam_coords.dim()
            )
        )
    if cam_coords.shape[-1] not in (3, 4):
        raise ivy.exceptions.IvyException(
            "Input cam_coords must have shape (*, 3), or (*, 4). Got {0} instead.".format(
                cam_coords.shape
            )
        )
    if proj_mat.dim() < 2:
        raise ivy.exceptions.IvyException(
            "Input proj_mat must have at least 2 dims. Got {0} instead.".format(
                proj_mat.dim()
            )
        )
    if proj_mat.shape[-1] != 4 or proj_mat.shape[-2] != 4:
        raise ivy.exceptions.IvyException(
            "Input proj_mat must have shape (*, 4, 4). Got {0} instead.".format(
                proj_mat.shape
            )
        )
    if proj_mat.dim() > 2 and proj_mat.dim() != cam_coords.dim():
        raise ivy.exceptions.IvyException(
            "Input proj_mat must either have 2 dimensions, or have equal number of dimensions to cam_coords. "
            "Got {0} instead.".format(proj_mat.dim())
        )
    if proj_mat.dim() > 2 and proj_mat.shape[0] != cam_coords.shape[0]:
        raise ivy.exceptions.IvyException(
            "Batch sizes of proj_mat and cam_coords do not match. Shapes: {0} and {1} respectively.".format(
                proj_mat.shape, cam_coords.shape
            )
        )

    # Determine whether to homogenize `cam_coords`.
    to_homogenize = cam_coords.shape[-1] == 3

    pts_homo = None
    if to_homogenize:
        pts_homo = homogenize_points(cam_coords)
    else:
        pts_homo = cam_coords

    # Determine whether `proj_mat` needs to be expanded to match dims of `cam_coords`.
    to_expand_proj_mat = (proj_mat.dim() == 2) and (pts_homo.dim() > 2)
    if to_expand_proj_mat:
        while proj_mat.dim() < pts_homo.dim():
            proj_mat = proj_mat.unsqueeze(0)

    # Whether to perform simple matrix multiplaction instead of batch matrix multiplication.
    need_bmm: bool = pts_homo.dim() > 2

    if not need_bmm:
        pts: torch.Tensor = torch.matmul(proj_mat.unsqueeze(0), pts_homo.unsqueeze(-1))
    else:
        pts: torch.Tensor = torch.matmul(proj_mat.unsqueeze(-3), pts_homo.unsqueeze(-1))

    # Remove the extra dimension resulting from torch.matmul()
    pts = pts.squeeze(-1)
    # Unhomogenize and stack.
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack((u, v), dim=-1)