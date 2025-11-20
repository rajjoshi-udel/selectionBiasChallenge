"""
Importance map computation for blue noise stippling.
This function identifies which regions of the image should receive more stipples.
"""

import numpy as np


def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.4,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
    mid_tone_boost: float = 0.4,
    mid_tone_sigma: float = 0.2,
):
    """
    Importance map computation that downweights extreme tones (very dark and very light)
    using smooth functions, while boosting mid-tones.
    
    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image in [0, 1]
    extreme_downweight : float
        Strength of downweighting for extreme tones (0.0 = no downweighting, 1.0 = maximum downweighting)
    extreme_threshold_low : float
        Threshold below which tones are considered "very dark" and get downweighted
    extreme_threshold_high : float
        Threshold above which tones are considered "very light" and get downweighted
    extreme_sigma : float
        Width of the smooth transition for extreme downweighting (smaller = sharper transition)
    mid_tone_boost : float
        Strength of mid-tone emphasis (0.0 = no boost, 1.0 = strong boost)
    mid_tone_sigma : float
        Width of the mid-tone Gaussian bump (smaller = narrower, larger = wider)
    
    Returns
    -------
    importance : np.ndarray
        Importance map in [0, 1]; higher = more stipples (dark areas and mid-tones get higher importance)
    """
    I = np.clip(gray_img, 0.0, 1.0)
    
    # Invert brightness: dark areas should get more dots (higher importance)
    I_inverted = 1.0 - I
    
    # Create smooth downweighting mask for extreme tones
    # Downweight very dark regions (I < extreme_threshold_low)
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    
    # Downweight very light regions (I > extreme_threshold_high)
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    
    # Combine both masks
    extreme_mask = np.maximum(dark_mask, light_mask)
    
    # Apply smooth downweighting
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    
    # Add smooth gradual mid-tone boost (Gaussian centered on 0.65)
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    
    # Boost importance in mid-tone regions
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    
    # Normalize to [0,1]
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance

