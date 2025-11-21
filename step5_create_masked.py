"""
Step 5: Create a masked stippled image.
Applies the block letter mask to the stippled image to demonstrate selection bias.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply a mask to a stippled image to create the "biased estimate".
    
    This function removes stipples (data points) where the mask is dark,
    demonstrating how selection bias systematically removes data in a pattern.
    
    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image as 2D array (height, width) with values in [0, 1]
        Black stipples (0.0) on white background (1.0)
    mask_img : np.ndarray
        Mask image as 2D array (height, width) with values in [0, 1]
        Dark areas (0.0) indicate where to remove stipples
        Light areas (1.0) indicate where to keep stipples
    threshold : float
        Threshold value to determine mask regions (0.0 to 1.0).
        Pixels in mask_img below this threshold are considered "masked"
        and stipples in those regions will be removed (set to white).
        Default 0.5.
    
    Returns
    -------
    masked_stipple : np.ndarray
        2D array (height × width) with values in [0, 1]
        Stippled image with masked regions removed (set to white)
    
    Notes
    -----
    The masking operation works as follows:
    - Where mask_img < threshold: set pixels to 1.0 (white, removing stipples)
    - Where mask_img >= threshold: keep original stipple_img values
    
    This creates the visual effect of the letter "erasing" stipples,
    demonstrating how selection bias systematically removes data points.
    """
    # Ensure inputs have the same shape
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"stipple_img and mask_img must have the same shape. "
            f"Got stipple_img: {stipple_img.shape}, mask_img: {mask_img.shape}"
        )
    
    # Create a copy of the stippled image
    masked_stipple = stipple_img.copy()
    
    # Create a boolean mask: True where mask is dark (below threshold)
    # These are the regions where we want to remove stipples
    mask_regions = mask_img < threshold
    
    # Set masked regions to white (1.0), effectively removing stipples
    masked_stipple[mask_regions] = 1.0
    
    return masked_stipple
