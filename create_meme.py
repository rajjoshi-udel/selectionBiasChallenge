"""
Create the final statistics meme.
Assembles all four panels (Reality, Your Model, Selection Bias, Estimate)
into a professional four-panel meme demonstrating selection bias.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white"
) -> None:
    """
    Create a four-panel statistics meme demonstrating selection bias.
    
    Assembles the original image, stippled version, block letter mask, and
    masked result into a professional-looking meme with labels.
    
    Parameters
    ----------
    original_img : np.ndarray
        Original grayscale image (Reality panel)
    stipple_img : np.ndarray
        Stippled version of the image (Your Model panel)
    block_letter_img : np.ndarray
        Block letter mask image (Selection Bias panel)
    masked_stipple_img : np.ndarray
        Masked stippled image (Estimate panel)
    output_path : str
        Path where the output PNG file will be saved
    dpi : int
        Resolution of the output image in dots per inch.
        Higher values produce better quality but larger files.
        Default 150 (good for screen and print).
    background_color : str
        Background color for the figure.
        Can be any matplotlib color name (e.g., "white", "pink", "lightgray").
        Default "white".
    
    Returns
    -------
    None
        Saves the meme to the specified output_path
    
    Notes
    -----
    The meme uses a 1×4 layout with panels labeled:
    - Panel 1: "Reality" (original image)
    - Panel 2: "Your Model" (stippled image)
    - Panel 3: "Selection Bias" (block letter)
    - Panel 4: "Estimate" (masked stippled image)
    """
    # Define the panel labels
    labels = ["Reality", "Your Model", "Selection Bias", "Estimate"]
    images = [original_img, stipple_img, block_letter_img, masked_stipple_img]
    
    # Create figure with 1 row and 4 columns
    fig = plt.figure(figsize=(20, 5), facecolor=background_color)
    gs = GridSpec(1, 4, figure=fig, wspace=0.05, hspace=0.05)
    
    # Create each panel
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(gs[0, i])
        
        # Display the image
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        
        # Add a border around the image
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Keep axes visible for the border
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title/label above each panel
        ax.set_title(
            label,
            fontsize=18,
            fontweight='bold',
            pad=15,
            color='black'
        )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor=background_color,
        edgecolor='none'
    )
    
    # Close the figure to free memory
    plt.close(fig)
    
    print(f"✅ Statistics meme saved to: {output_path}")
    print(f"   Resolution: {dpi} DPI")
    print(f"   Background: {background_color}")
