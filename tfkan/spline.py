import tensorflow as tf


def B_batch(x: tf.Tensor, grid: tf.Tensor, k: int = 0, extend: bool = True):
    """
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D tf.Tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D tf.Tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D tf.Tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    """

    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for _ in range(k_extend):
            grid = tf.concat([grid[:, [0]] - h, grid], dim=1)
            grid = tf.concat([grid, grid[:, [-1]] + h], dim=1)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D tf.tensor)
            shape (number of splines, number of samples)
        grid : 2D tf.tensor)
            shape (number of splines, number of grid points)
        coef : 2D tf.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D tf.tensor
            shape (number of splines, number of samples)

    Example
    -------
    """
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = tf.einsum("ij,ijk->ik", coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D tf.tensor
            shape (number of splines, number of samples)
        y_eval : 2D tf.tensor
            shape (number of splines, number of samples)
        grid : 2D tf.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    """
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = tf.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    return coef