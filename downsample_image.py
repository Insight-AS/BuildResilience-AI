"""
Downsample a GeoTIFF in memory without creating a separate output file.

Features:
- Supports arbitrary GeoTIFF dimensions.
- Automatically downsamples images exceeding MAX_PIXELS.
- Preserves CRS and affine transform.
- Avoids creating a separate downsampled TIFF on disk.
- Returns the resized raster array and updated metadata.
"""

import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio import Affine


MAX_PIXELS = 20_000_000


def downsample_geotiff(input_file: str):
    """
    Read a GeoTIFF and downsample it in memory if it exceeds MAX_PIXELS.

    Returns:
        data: NumPy array containing the raster bands.
        profile: Updated raster metadata.
    """

    with rasterio.open(input_file) as src:

        original_width = src.width
        original_height = src.height
        original_pixels = original_width * original_height

        print(f"Original resolution: {original_width} x {original_height}")
        print(f"Original pixel count: {original_pixels:,}")

        # Determine target resolution
        if original_pixels > MAX_PIXELS:

            scale = (MAX_PIXELS / original_pixels) ** 0.5

            new_width = max(1, int(original_width * scale))
            new_height = max(1, int(original_height * scale))

            print("Image exceeds pixel limit. Downsampling in memory...")

        else:

            new_width = original_width
            new_height = original_height

            print("Image is within pixel limit. No downsampling required.")

        # Read resized raster into RAM
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear,
        )

        # Preserve and update geospatial metadata
        profile = src.profile.copy()

        profile.update(
            width=new_width,
            height=new_height,
            transform=src.transform
            * Affine.scale(
                original_width / new_width,
                original_height / new_height,
            ),
        )

        print(f"Processed resolution: {new_width} x {new_height}")
        print(f"Processed pixel count: {new_width * new_height:,}")
        print("GeoTIFF processed in memory. No output file created.")

        return data, profile


if __name__ == "__main__":

    input_file = r"C:\Users\speed\Downloads\493701.tif"

    data, profile = downsample_geotiff(input_file)

    print("\nProcessing completed successfully!")
    print(f"Array shape: {data.shape}")
    print(f"CRS: {profile.get('crs')}")
