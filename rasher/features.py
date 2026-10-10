"""Deterministic image-only descriptors. No filenames or labels become features."""
import numpy as np
from PIL import Image


def extract_features(path, grid=4):
    """Return RGB/texture descriptors using a fixed 96x72 image.

    Args:
        path: Readable image path.
        grid: Spatial subdivisions per axis, either 2 or 4.
    Returns:
        Float64 vector: six RGB moments and two gradient means per cell.
    Raises:
        ValueError: Unsupported grid size.
        OSError: Unreadable image.
    """
    if grid not in (2,4): raise ValueError('grid must be 2 or 4')
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB').resize((96,72),Image.Resampling.BILINEAR),dtype=float)/255
    vector=[]
    for row in np.array_split(a,grid,axis=0):
        for cell in np.array_split(row,grid,axis=1):
            gray=cell.mean(axis=2)
            vector.extend(cell.mean(axis=(0,1)))
            vector.extend(cell.std(axis=(0,1)))
            vector.extend([np.abs(np.diff(gray,axis=0)).mean(),np.abs(np.diff(gray,axis=1)).mean()])
    return np.asarray(vector)
