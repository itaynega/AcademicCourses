import numpy as np
import cv2
import os
def return_names_IDs():
    # TODO: Replace 'None' with your full name as string
    name1 = "Itay Nega" 
    # TODO: Replace 'None' with your student ID as string
    id1 = "208109678"
    
    # If working in pairs, fill in your partner's details as well
    name2 = "Kfir Goldring"
    id2 = "211873575"
    
    # Return a formatted string with both students' names and IDs
    # NOTE: This will be used for assignment identification and grading
    return name1, id1, name2, id2

def block_permutation(image, rng, n_blocks=4):
    """
    Randomly permute non-overlapping blocks of a grayscale image.

    The image is divided into an n_blocks × n_blocks grid of tiles 
    (as evenly as possible). These tiles are then shuffled randomly 
    and reassembled to form the output image. 
    
    This preserves the global histogram but destroys most spatial 
    relationships between pixels across blocks.

    Parameters
    ----------
    image : np.ndarray (H, W)
        Input grayscale image.
    rng : np.random.Generator
        A NumPy random generator used to shuffle blocks 
        (ensures reproducibility if seeded).
    n_blocks : int, default=4
        Number of blocks along each axis. The image will be split 
        into n_blocks × n_blocks tiles.

    Returns
    -------
    out : np.ndarray (H, W)
        The block-permuted image, same shape and dtype as input.
    """
    h, w = image.shape
    row_edges = np.linspace(0, h, n_blocks + 1, dtype=int)
    col_edges = np.linspace(0, w, n_blocks + 1, dtype=int)

    # Extract blocks
    tiles = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            r0, r1 = row_edges[i], row_edges[i+1]
            c0, c1 = col_edges[j], col_edges[j+1]
            tiles.append(image[r0:r1, c0:c1].copy())

    # Shuffle
    rng.shuffle(tiles)

    # Reconstruct
    out = np.empty_like(image)
    k = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            r0, r1 = row_edges[i], row_edges[i+1]
            c0, c1 = col_edges[j], col_edges[j+1]
            block = tiles[k]
            out[r0:r1, c0:c1] = block[:(r1-r0), :(c1-c0)]
            k += 1

    return out

def _to_gray_uint8(img):
    """
    Converts input image to 2D grayscale uint8.
    Supports RGB (H,W,3) or grayscale float/int.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def random_modification(
    img=None,
    dataset=None,
    rng=None,
    mod_type=None,                
    n_blocks=4,                    
    noise_sigma=25.0,             
    brightness_range=(-50, 50),   
    contrast_range=(0.5, 1.5),    
):
    """
    Pick a grayscale image (given or random from dataset) and apply a modification.

    Parameters
    ----------
    img : np.ndarray or None
    dataset : list[np.ndarray] or None
    rng : np.random.Generator or None
    mod_type : str or None
        One of {"brightness","contrast","noise","block_permutation",
                 "invert","flip_h","flip_v","half_flip"}.
        If None, a random one is chosen.
    n_blocks : int
        Grid size for block permutation (n_blocks x n_blocks).
    noise_sigma : float
        Std dev for Gaussian noise.
    brightness_range : tuple(int,int)
        Inclusive range for brightness offset sampling.
    contrast_range : tuple(float,float)
        Range for contrast scaling factor sampling.

    Returns
    -------
    original : np.ndarray (uint8, HxW)
    modified : np.ndarray (uint8, HxW)
    chosen_mod : str
    """
    if rng is None:
        rng = np.random.default_rng()

    if img is None:
        if dataset is None:
            raise ValueError("Either img or dataset must be provided.")
        idx = int(rng.integers(low=0, high=len(dataset)))
        img = dataset[idx]

    original = _to_gray_uint8(img)

    # --- Choose/validate modification ---
    mods = [
        "brightness",
        "contrast",
        "noise",
        "block_permutation",
        "invert",
        "flip_h",
        "flip_v",
        "half_flip",
    ]
    if mod_type is None:
        chosen_mod = rng.choice(mods)
    else:
        if mod_type not in mods:
            raise ValueError(f"mod_type must be one of {mods}, got {mod_type!r}")
        chosen_mod = mod_type

    # --- Apply modification ---
    modified = original.copy()

    if chosen_mod == "brightness":
        lo, hi = brightness_range
        offset = int(rng.integers(lo, hi + 1))
        modified = np.clip(modified.astype(np.int16) + offset, 0, 255).astype(np.uint8)

    elif chosen_mod == "contrast":
        lo, hi = contrast_range
        alpha = float(rng.uniform(lo, hi))
        modified = np.clip(modified.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    elif chosen_mod == "noise":
        noise = rng.normal(0.0, noise_sigma, size=modified.shape).astype(np.float32)
        modified = np.clip(modified.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif chosen_mod == "block_permutation":
        modified = block_permutation(modified, rng, n_blocks=int(n_blocks))

    elif chosen_mod == "invert":
        modified = 255 - modified

    elif chosen_mod == "flip_h":
        modified = np.fliplr(modified)

    elif chosen_mod == "flip_v":
        modified = np.flipud(modified)

    elif chosen_mod == "half_flip":
        h, w = modified.shape
        modified[:, w // 2 :] = np.fliplr(modified[:, w // 2 :])

    return original, modified, chosen_mod


def manual_entropy(img):
    hist = manual_histogram(img)              
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total       
    p = p[p > 0]                              
    H = -np.sum(p * np.log2(p))
    return float(H)

def manual_histogram(img):
    hist = np.zeros(256, dtype=int)
    flat_img = img.flatten()
    for pixel in flat_img:
        hist[pixel] += 1

    return hist

def is_diagonal(matrix):
    return np.all(matrix == np.diag(np.diag(matrix)))


def manual_joint_histogram(img1, img2):
    joint_hist = np.zeros((256, 256), dtype=int)
    flat_img1 = img1.flatten()
    flat_img2 = img2.flatten()
    for i in range(len(flat_img1)):
        joint_hist[int(flat_img1[i]), int(flat_img2[i])] += 1
    return joint_hist




def identify_image(orig_img,mod_img):
    brightness = True
    contrast = True
    noise = True
    block_permutation = True
    invert = True
    flip_h = True
    flip_v = True
    half_flip = True

    orig_arr = np.array(orig_img.astype(np.float32))
    orig_arr_flat = orig_arr.flatten()

    mod_arr = np.array(mod_img.astype(np.float32))
    mod_arr_flat = mod_arr.flatten()
    diff_flag = False
    mult_flag = False

    for i in range(len(orig_arr_flat)):
        if orig_arr_flat[i] != 0 and mod_arr_flat[i] != 0 and orig_arr_flat[i] != 255 and mod_arr_flat[i] != 255:
            # brightness
            if not diff_flag and brightness:
                diff = orig_arr_flat[i] - mod_arr_flat[i]
                diff_flag = True
            else:
                if orig_arr_flat[i] - mod_arr_flat[i] != diff:
                    brightness = False
            # contrast
            if not mult_flag and contrast:
                mult = mod_arr_flat[i] / orig_arr_flat[i]
                mult_flag = True
            else:
                if not (abs(mod_arr_flat[i] - orig_arr_flat[i]*mult) < 5): # 5 is the margin of error
                    contrast = False
        
        # invert
        if orig_arr_flat[i] + mod_arr_flat[i] != 255:
            invert = False

    # flip_h
    sum_img = orig_arr + mod_arr
    for i in range(sum_img.shape[1]//2):
        if (sum_img[:,i] != sum_img[:, sum_img.shape[1]-1-i]).any():
            flip_h = False

    # flip_v
    for i in range(sum_img.shape[0]//2):
        if (sum_img[i,:] != sum_img[sum_img.shape[0]-1-i,:]).any():
            flip_v = False

    # half flip
    mid_col = sum_img.shape[1]//2
    sum_img_right = sum_img[:, mid_col:]
    for i in range(sum_img_right.shape[1]//2):
        if (sum_img_right[:,i] != sum_img_right[:, sum_img_right.shape[1]-1-i]).any():
            half_flip = False
    

    if brightness:
        return "brightness"
    if contrast:
        return "contrast"
    if invert:
        return "invert"
    if flip_h:
        return "flip_h"
    if flip_v:
        return "flip_v"
    if half_flip:
        return "half_flip"
    
    #block permutation:
    orig_hist=manual_histogram(orig_img)
    mod_hist=manual_histogram(mod_img)
    joint_hist=manual_joint_histogram(orig_img, mod_img)
    if is_diagonal(joint_hist) or (orig_hist!=mod_hist).any():
        block_permutation = False

    # noise
    entropy_orig = manual_entropy(orig_img)
    entropy_mod = manual_entropy(mod_img)
    if entropy_orig == entropy_mod:
        noise = False

    if noise:
        return "noise"
    else:
        return "block_permutation"
