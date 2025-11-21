"""
Step 4: Create a block letter matching image dimensions.
Generates a block letter (default "S") that will serve as the selection bias pattern.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Create a block letter matching the specified image dimensions.
    
    The letter will be black (0.0) on a white background (1.0), representing
    the selection bias pattern that will systematically remove data points.
    
    Parameters
    ----------
    height : int
        Height of the output image in pixels
    width : int
        Width of the output image in pixels
    letter : str
        The letter to draw (default "S" for Selection Bias)
    font_size_ratio : float
        Ratio of letter size to image height (0.0 to 1.0).
        Default 0.9 means the letter will be 90% of the image height.
    
    Returns
    -------
    block_letter : np.ndarray
        2D array (height × width) with values in [0, 1]
        The letter is black (0.0) on white background (1.0)
    """
    # Create a white image
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Calculate font size based on image height
    target_font_size = int(height * font_size_ratio)
    
    # Try to load a bold font from common system locations
    font = None
    font_paths = [
        # macOS fonts
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        # Generic fallback
        "Arial-Bold",
        "Helvetica-Bold",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, target_font_size)
            break
        except (IOError, OSError):
            continue
    
    # If no TrueType font found, use default font and scale up
    if font is None:
        try:
            # Try to use a default font with size
            font = ImageFont.load_default()
        except:
            font = None
    
    # Get text bounding box to center the letter
    if font is not None:
        # Get bounding box using textbbox (newer PIL/Pillow method)
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(letter, font=font)
    else:
        # If no font available, create a simple geometric "S"
        # This is a fallback approach
        return _create_geometric_s(height, width)
    
    # Calculate position to center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw the letter in black (0)
    draw.text((x, y), letter, fill=0, font=font)
    
    # Convert to numpy array and normalize to [0, 1]
    block_letter = np.array(img, dtype=np.float32) / 255.0
    
    return block_letter


def _create_geometric_s(height: int, width: int) -> np.ndarray:
    """
    Create a geometric "S" shape as a fallback when fonts are unavailable.
    
    This creates a simple block "S" using geometric shapes.
    
    Parameters
    ----------
    height : int
        Height of the output image in pixels
    width : int
        Width of the output image in pixels
    
    Returns
    -------
    block_letter : np.ndarray
        2D array (height × width) with values in [0, 1]
        The letter is black (0.0) on white background (1.0)
    """
    # Create a white image
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Define S shape using rectangles and curves
    # Calculate dimensions relative to image size
    margin_x = width * 0.2
    margin_y = height * 0.1
    stroke_width = int(min(width, height) * 0.15)
    
    # Top horizontal bar
    top_y = margin_y
    top_bar = [
        margin_x, top_y,
        width - margin_x, top_y + stroke_width
    ]
    draw.rectangle(top_bar, fill=0)
    
    # Left vertical section (top part)
    left_top = [
        margin_x, top_y,
        margin_x + stroke_width, height * 0.4
    ]
    draw.rectangle(left_top, fill=0)
    
    # Middle horizontal bar
    mid_y = height * 0.4
    mid_bar = [
        margin_x, mid_y,
        width - margin_x, mid_y + stroke_width
    ]
    draw.rectangle(mid_bar, fill=0)
    
    # Right vertical section (bottom part)
    right_bottom = [
        width - margin_x - stroke_width, mid_y,
        width - margin_x, height - margin_y
    ]
    draw.rectangle(right_bottom, fill=0)
    
    # Bottom horizontal bar
    bottom_y = height - margin_y - stroke_width
    bottom_bar = [
        margin_x, bottom_y,
        width - margin_x, height - margin_y
    ]
    draw.rectangle(bottom_bar, fill=0)
    
    # Convert to numpy array and normalize to [0, 1]
    block_letter = np.array(img, dtype=np.float32) / 255.0
    
    return block_letter
