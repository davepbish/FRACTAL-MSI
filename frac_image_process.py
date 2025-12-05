import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pewlib import Laser, io
from pewlib.config import SpotConfig
from pewlib.process.filters import rolling_median

import register


def read_tiff_image(
    path: Path, channel: int = 0
) -> tuple[np.ndarray, tuple[float, float]]:
    """Read in micrograph from TIFF.

    Args:
        path: path to tiff file
        channel: which channel tiff page (to read)

    Returns:
        image, pixel size
    """
    from PIL import Image

    # TIFF exports from QuPath, Zen
    image = Image.open(path)
    image.seek(channel)
    exif = image.getexif()

    resunits = {
        1: 1.0,  # µm
        3: 10000.0,  # cm
    }

    px = float(resunits[exif[296]] / exif[282])
    py = float(resunits[exif[296]] / exif[283])

    return np.array(image), (px, py)


def read_czi_image(
    path: Path | str, channel: int = 0
) -> tuple[np.ndarray, tuple[float, float]]:
    """Read in micrograph from Zeiss CZI.

    Args:
        path: path to tiff file
        channel: which channel to read

    Returns:
        image, pixel size
    """
    import aicspylibczi

    czi = aicspylibczi.CziFile(path)

    items = czi.meta.findall(".//Scaling/Items")
    if len(items) != 1:
        raise ValueError("too many items")
    px, py, pz = [float(item.text) * 1e6 for item in items[0].findall(".//Value")]
    img = czi.read_mosaic(C=channel, scale_factor=1)[0, :, :]

    return img, (px, py)


def read_transform(path: Path | str) -> np.ndarray:
    """Read in the micrograph to laser transform.

    Expects file in format as exported by pewpew.
    """
    trans = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=3).T
    pos = np.loadtxt(path, delimiter=",", skiprows=4)
    trans[2][0] += pos[0]
    trans[2][1] += pos[1]
    return trans


def read_laser_image(
    path: Path | str, element: str | None = None
) -> tuple[np.ndarray, tuple[float, float], str]:
    """Read in laser image.

    Expects a laser .npz, as exported by pewpew.

    Args:
        path: to the .npz file
        element: which element to read

    Returns:
        image data, pixel size, element read
    """
    laser = io.npz.load(path)
    px, py = laser.config.get_pixel_width(), laser.config.get_pixel_height()
    if element is None:
        element = str(laser.elements[0])

    return laser.get(element, flat=True), (px, py), element


def map_image_to(
    img: np.ndarray, idx: np.ndarray, img_to: np.ndarray, idx_to: np.ndarray
) -> np.ndarray:
    """Remap an image from the `idx` coordinates to the `idx_to`.

    Mapping is performed by summing overlapping pixels in the newimage.
    Both `idx` and `idx_to` can be generated using `register.map_transformed_image`.

    Args:
        img: the image to be re-mapped
        idx: indicies of pixels in `img`
        img_to: the new image, used to determine size
        idx_to: indicies of the pixels in `img_to`

    Returns:
        re-mapped image
    """
    mapped = np.zeros_like(img_to)
    np.add.at(mapped, (idx_to[:, 0], idx_to[:, 1]), img[(idx[:, 0], idx[:, 1])])
    # mapped[mapped == 0] = np.nan
    return mapped


def mean_normalised_image(
    img: np.ndarray, idx: np.ndarray, img_from: np.ndarray, idx_from: np.ndarray
) -> np.ndarray:
    """Creates an image where each pixel group is normalised to its mean.

    Pixel groups are pixels in `img` that overlap those in `img_from`. The result can
    be multiplied by the a re-mapped image (`map_image_to`) to recover quantitative values.
    Both `idx` and `idx_to` can be generated using `register.map_transformed_image`.

    Args:
        img: the micrograph
        idx: indicies of pixels in `img`
        img_from: the laser image
        idx_from: indicies of pixels in `img_from`

    Returns:
        combined image where  laser pixels are scaled by the mean of overlapping micrograph pixels
    """
    counts = np.zeros_like(img_from)
    np.add.at(counts, (idx_from[:, 0], idx_from[:, 1]), 1)
    totals = map_image_to(img, idx, img_from, idx_from)

    mean = np.divide(totals, counts, where=counts > 0)
    mean[counts == 0] = 0.0

    mean_mapped = map_image_to(mean, idx_from, img_to=img, idx_to=idx)

    normed = np.divide(img, mean_mapped, where=mean_mapped > 0)
    normed[mean_mapped == 0] = np.nan
    return normed


def histogram_normalise_image(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map the CDF of ``img`` to that of ``ref``.

    Rescales values in ``img`` to match the distribution of values found in ``ref``.

    Args:
        img: image to scale
        ref: reference image to read CDF

    Returns:
        `img` rescaled to match CDF in `ref`
    """
    _, idx, counts = np.unique(img, return_counts=True, return_inverse=True)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]

    ref_unique, counts = np.unique(ref, return_counts=True)
    ref_cdf = np.cumsum(counts)
    ref_cdf = ref_cdf / ref_cdf[-1]

    interpolated = np.interp(cdf, ref_cdf, ref_unique)

    normed = interpolated[idx].reshape(img.shape)
    return normed


def trim_nans(x: np.ndarray, name: str):
    """Trim all NaN rows  and collumns from structured array.

    Args:
        x: structured array
        name: name of layer to use as NaN mask

    Returns:
        array with NaNs trimmed
    """
    nan_rows = np.all(np.isnan(x[name]), axis=1)
    x = x[~nan_rows]
    nan_cols = np.all(np.isnan(x[name]), axis=0)
    x = x[:, ~nan_cols]
    return x


def main():
    parser = argparse.ArgumentParser(
        "fracanal",
        description="Analysis software for combined immunohistological and LA-ICP-MS FRACTAL samples",
    )
    parser.add_argument(
        "micrograph", type=Path, help="path to microscope image (.tiff, .czi)"
    )
    parser.add_argument("laser", type=Path, help="path to laser image (.npz)")
    parser.add_argument("transform", type=Path, help="path to transform file")
    parser.add_argument("--channel", type=int, default=0, help="micrograph channel")
    parser.add_argument("--element", type=str, help="laser image element to use")
    parser.add_argument(
        "--smooth", type=float, nargs="?", const=1.0, help="smooth the laser image"
    )
    parser.add_argument(
        "--filter", type=float, nargs="?", const=9.0, help="mean filter the laser image"
    )
    parser.add_argument("--show", action="store_true", help="show the registered image")
    parser.add_argument(
        "--save", action="store_true", help="save the un- and registered images as PNGs"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="path to save the un- and registered images"
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = Path("combined.npz")
    elif args.output.suffix != ".npz":
        parser.error("output must be a npz file")

    if args.micrograph.suffix.lower() not in [".czi", ".tif", ".tiff"]:
        parser.error("micrograph must be a CZI or TIFF")

    print("reading microscope image...", flush=True)
    if args.micrograph.suffix.lower() == ".czi":
        micro, (mx, my) = read_czi_image(args.micrograph, channel=args.channel)
    else:
        micro, (mx, my) = read_tiff_image(args.micrograph, channel=args.channel)
    print(f"\tpixel size: {mx}, {my}", flush=True)

    micro = micro.astype(np.float32)

    print("reading laser image...", flush=True)
    laser, (lx, ly), element = read_laser_image(args.laser, element=args.element)
    print(f"\tpixel size: {lx}, {ly}", flush=True)
    if args.filter is not None:
        print(f"median filtering laser image, MAD={args.filter}...", flush=True)
        laser = rolling_median(laser, 5, args.filter)
    if args.smooth is not None:
        print(f"smoothing laser image, σ={args.smooth:.1f}...", flush=True)
        from scipy.ndimage import gaussian_filter

        laser = gaussian_filter(laser, args.smooth)

    print("reading transform...", flush=True)
    trans = read_transform(args.transform)

    print("registering images using transform...", flush=True)
    midx, lidx = register.map_transformed_image(micro, (mx, my), laser, (lx, ly), trans)
    valid = register.valid_indicies(lidx, laser)

    print("mapping laser to micrograph...", flush=True)
    laser_mapped = map_image_to(laser, lidx[valid], micro, midx[valid])

    print("mean normalising micrograph per laser pixel...", flush=True)
    normed_micro = mean_normalised_image(micro, midx[valid], laser, lidx[valid])
    combined = laser_mapped * normed_micro

    print("cdf mapping micro to laser histogram...", flush=True)
    hist_map = histogram_normalise_image(micro, laser_mapped[~np.isnan(laser_mapped)])

    if args.show:
        rgb = np.zeros((*laser_mapped.shape, 3))
        rgb[:, :, 0] = laser_mapped / np.nanpercentile(laser_mapped, 99.5)
        rgb[:, :, 1] = micro / np.nanpercentile(micro, 99.5)
        np.clip(rgb, 0.0, 1.0, out=rgb)

        plt.imshow(rgb)
        plt.show()
        exit()

    if args.save:
        rgb = np.zeros((*laser_mapped.shape, 3))
        rgb[:, :, 0] = laser_mapped / np.nanpercentile(laser_mapped, 99.5)
        rgb[:, :, 1] = micro / np.nanpercentile(micro, 99.5)
        np.clip(rgb, 0.0, 1.0, out=rgb)

        zeros = np.zeros_like(rgb[:, :, 0])

        plt.imsave("laser_map.png", rgb)
        plt.imsave("laser.png", np.stack((rgb[:, :, 0], zeros, zeros), axis=2))
        plt.imsave("fluoro.png", np.stack((zeros, rgb[:, :, 1], zeros), axis=2))

    new_name = f"{element}_channel{args.channel}"
    data = np.empty(
        combined.shape,
        dtype=[
            (new_name, np.float32),
            ("cdf", np.float32),
            ("mapped", np.float32),
            ("micro", np.float32),
        ],
    )
    data[new_name] = combined
    data["mapped"] = laser_mapped
    data["micro"] = micro
    data["cdf"] = hist_map

    data = trim_nans(data, data.dtype.names[0])  # type: ignore

    print(f"writing to {args.output}...", flush=True)
    io.npz.save(
        args.output,
        Laser(
            data=data,
            config=SpotConfig(mx, my),
            info={
                "Name": f"combined {new_name}",
                "Micrograph": args.micrograph.stem,
                "Channel": str(args.channel),
                "Laser": args.laser.stem,
            },
        ),
    )


if __name__ == "__main__":
    main()
