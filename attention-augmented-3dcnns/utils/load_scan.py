import SimpleITK as sitk
import numpy as np


def load_scan(path, new_spacing=(1, 1, 1)):
    """
    Load a .mhd CT scan, resample to new_spacing, and normalize HU values.
    Args:
        path (str): Path to the .mhd file.
        new_spacing (tuple): Desired voxel spacing (z, y, x) in mm.
    Returns:
        np.ndarray: Resampled and normalized CT volume (float32, [0, 1])
        np.ndarray: Origin (3,)
        np.ndarray: Spacing (3,)
    """
    # Load image
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)  # z, y, x
    origin = np.array(list(reversed(itk_img.GetOrigin())))  # x, y, z -> z, y, x
    spacing = np.array(list(reversed(itk_img.GetSpacing())))

    # Resample
    resize_factor = spacing / np.array(new_spacing)
    new_shape = np.round(img.shape * resize_factor).astype(int)
    real_resize_factor = new_shape / img.shape
    new_spacing = spacing / real_resize_factor

    img = sitk.GetImageFromArray(img)
    img.SetSpacing(tuple(list(reversed(spacing))))
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(list(reversed(new_spacing))))
    resampler.SetSize([int(sz) for sz in new_shape[::-1]])
    resampler.SetInterpolator(sitk.sitkLinear)
    img = resampler.Execute(img)
    img = sitk.GetArrayFromImage(img)

    # Normalize HU
    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400  # [-1000, 400] -> [0, 1]
    img = img.astype(np.float32)

    return img, origin, new_spacing 