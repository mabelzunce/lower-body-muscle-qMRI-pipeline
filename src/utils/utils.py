import SimpleITK as sitk
import numpy as np
import multiprocessing

def apply_bias_correction(image: np.ndarray, shrink_factor=4) -> np.ndarray:
    """
    Applies an N4 bias correction filter to an image in a numpy array.

    Parameters:
        image (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Bias-corrected image.
    """
    
    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image_lowres = sitk_image #In cas shrinking is appleid
    # ---- (optional but recommended) build a mask ----
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

    # Shrink the image:
    if shrink_factor > 1:
        sitk_image_lowres = sitk.Shrink(sitk_image, [shrink_factor] * sitk_image.GetDimension())
        mask = sitk.Shrink(
            mask, [shrink_factor] * mask.GetDimension()
        )  

    # Initialize the N4 bias field correction filter
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.SetNumberOfThreads(multiprocessing.cpu_count())
    # Apply the filter
    corrected_image = n4_filter.Execute(sitk_image_lowres, mask)

    log_bias_field = n4_filter.GetLogBiasFieldAsImage(sitk_image)

    corrected_image_full_resolution = sitk.Cast(sitk_image, sitk.sitkFloat32) / sitk.Exp(log_bias_field)

    # Convert the corrected image back to a numpy array
    corrected_array = sitk.GetArrayFromImage(corrected_image_full_resolution)

    return corrected_array
