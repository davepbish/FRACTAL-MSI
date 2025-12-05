import numpy as np


def affine_transform(
    coords: np.ndarray, matrix: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """Perform an affine transform of coordinates.

    Args:
        coords: array of points with shape (3, ...)
        matrix: the transformation matrix
        inverse: do an inverse transformation
    Returns:
        transformed coords
    """
    if inverse:
        matrix = np.linalg.inv(matrix)
    # swap x,y (numpy array index) to y,x (coords) and back
    return np.dot(coords[:, [1, 0, 2]], matrix)[:, [1, 0, 2]]


def pixel_centers(
    shape: np.ndarray | tuple[int, int], px: float, py: float | None = None
) -> np.ndarray:
    """Gets center coordinate of pixels.

    Args:
        shape: array or array shape
        px: pixel width
        py: pixel height (defaults to px)

    Returns:
        array of (x, y, z) coordinates
    """
    if isinstance(shape, np.ndarray):
        shape = shape.shape[:2]
    if py is None:
        py = px

    xs = np.arange(shape[0]) * px + px / 2.0
    ys = np.arange(shape[1]) * py + py / 2.0

    X, Y = np.meshgrid(xs, ys)
    return np.stack((X.flat, Y.flat, np.ones(Y.size)), axis=1)


def pixel_indicies(shape: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Indicies of pixels.

    Args:
        shape: array or array shape

    Returns:
        array of (x, y) indicices
    """
    if isinstance(shape, np.ndarray):
        shape = shape.shape
    shape = shape[:2]

    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    return np.stack((X.flat, Y.flat), axis=1)


def pixel_indicies_from_centers(
    centers: np.ndarray,
    px: float,
    py: float | None = None,
) -> np.ndarray:
    """Gets the index for each pixel center.

    Args:
        centers: centers from pixel_centers (shape (N, 3))
        px: pixel width
        py: pixel height (defaults to px)

    Returns:
        indices of pixels
    """
    if py is None:
        py = px

    xi = (centers[:, 0] / px).astype(int)
    yi = (centers[:, 1] / py).astype(int)

    return np.stack((xi, yi), axis=1)


def valid_indicies(
    indicies: np.ndarray, shape: np.ndarray | tuple[int, int]
) -> np.ndarray:
    """Creates a mask for indicies valid for shape.

    Args:
        indicies: from pixel_indicies_from_centers
        shape: array or array shape

    Returns:
        mask of valid indicies
    """
    if isinstance(shape, np.ndarray):
        shape = shape.shape[:2]

    return np.logical_and(
        np.logical_and(indicies[:, 0] >= 0, indicies[:, 1] >= 0),
        np.logical_and(indicies[:, 0] < shape[0], indicies[:, 1] < shape[1]),
    )


def map_transformed_image(
    from_array: np.ndarray,
    from_pixel_size: tuple[float, float],
    to_array: np.ndarray,
    to_pixel_size: tuple[float, float],
    transform: np.ndarray,
    inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Maps from_array indicies to ro_array using the given transform.

    Args:
        from_array: array map from
        from_pixel_size: size of pixels in from_array
        to_array: array map to
        to_pixel_size: size of pixels in to_array
        transform: affine transform from from_array to to_array
        inverse: transform is from to_array to from_array

    Returns:
        idx of from_array, idx mapped to to_array
    """

    from_idx = pixel_indicies(from_array.shape)

    if isinstance(from_pixel_size, float):
        centers = pixel_centers(from_array.shape, from_pixel_size)
    else:
        centers = pixel_centers(
            from_array.shape, from_pixel_size[0], from_pixel_size[1]
        )

    remapped = affine_transform(centers, transform, inverse=inverse)

    if isinstance(to_pixel_size, float):
        to_idx = pixel_indicies_from_centers(remapped, to_pixel_size)
    else:
        to_idx = pixel_indicies_from_centers(
            remapped, to_pixel_size[0], to_pixel_size[1]
        )

    return from_idx, to_idx
