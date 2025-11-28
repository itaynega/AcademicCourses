import numpy as np, cv2, math
from skimage import data, color
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt

def return_names_IDs():
    # TODO: Replace 'None' with your full name as string
    name1 = "Kfir Goldring"
    # TODO: Replace 'None' with your student ID as string
    id1 = "211873575"
    
    # If working in pairs, fill in your partner's details as well
    name2 = "Itay Nega"
    id2 = "208109678"
    
    # Return a formatted string with both students' names and IDs
    # NOTE: This will be used for assignment identification and grading
    return name1, id1, name2, id2


def random_filtered_image(rng_seed=0, r_low=None, r_high=None):
    """
    Randomly select an image from skimage.data,
    apply a random frequency filter (LP, HP, BP),
    and return the original image, the filtered image, 
    and the filter type.
    
    rng_seed : int
        Random seed for reproducibility.
    r_low, r_high : int or None
        Radii for band-pass filter. If None, chosen randomly.
    """
    rng = np.random.default_rng(rng_seed)
    
    dataset = [
        data.camera(),
        data.coins(),
        data.checkerboard(),
        data.astronaut(),  # RGB
    ]

    idx = rng.integers(len(dataset))   # choose random index
    img = dataset[idx]
    
    # --- Step 2: Convert to grayscale if needed ---
    if img.ndim == 3:  # RGB image
        img = color.rgb2gray(img)
    
    img = img.astype(np.float32)
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    if r_low is None:
        r_low = rng.integers(low=5, high=min(rows, cols)//6)
    if r_high is None:
        r_high = rng.integers(low=r_low+10, high=min(rows, cols)//4)
    cv, cu = np.ogrid[:rows, :cols]
    dist_sq = (cv - crow)**2 + (cu - ccol)**2
    
    LP_mask = (dist_sq <= r_high**2).astype(np.uint8)
    HP_mask = (dist_sq >= r_low**2).astype(np.uint8)
    BP_mask = ((dist_sq >= r_low**2) & (dist_sq <= r_high**2)).astype(np.uint8)
    
    filters = {"LP": LP_mask, "HP": HP_mask, "BP": BP_mask}
    
    filter_name = rng.choice(list(filters.keys()))
    mask = filters[filter_name]
    
    F_filtered = F_shifted * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
    
    return img_filtered, filter_name

def what_filter(img):
    # receive filtered image, return the type of filter done on it
    img = img.astype(np.float32)
    F = np.fft.fft2(img)#2D spectrum
    F_shift = np.fft.fftshift(F)                        # shift spectrum so that DC is the center and high freq in the edges
    mag = np.abs(F_shift)
    mag_normalized = mag/mag.max()                      # normalize to get magnitude in (0,1)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2                   # dc position
    
    if mag_normalized[crow,ccol] > 1e-5: return "LP"    # non-negligible DC power-->LP
    corners = [ mag_normalized[0, 0],
                mag_normalized[0, -1],
                mag_normalized[-1, 0],
                mag_normalized[-1, -1]]                 # these are the 4 high freq edges of the spectrum. non negligible power-->HP
    
    if np.any(np.array(corners) > 1e-5): return "HP"
    
    return "BP"