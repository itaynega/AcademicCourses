from calendar import c
import numpy as np, cv2, math
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import cv2
def return_names_IDs():
    name1 = None  
    id1 = None    
    
    # If working in pairs, fill in your partner's details as well
    name2 = None  
    id2 = None   
    # Return a formatted string with both students' names and IDs
    # NOTE: This will be used for assignment identification and grading
    return name1, id1, name2, id2

def _rotate_and_translate(pts: np.ndarray, angle_deg: float, center: Tuple[int,int]) -> np.ndarray:
    M = cv2.getRotationMatrix2D((0, 0), angle_deg, 1.0)
    pts = np.asarray(pts, np.float32)
    pts_rot = cv2.transform(pts[None, ...], M)[0]
    pts_rot[:, 0] += center[0]
    pts_rot[:, 1] += center[1]
    return pts_rot.astype(np.int32)

def _iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_w, inter_h = max(0, xB - xA), max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter_area / float(areaA + areaB - inter_area)

def _rotate_and_translate(points: np.ndarray, angle: float, center: Tuple[int, int]) -> np.ndarray:
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    return (points @ rot.T + np.array(center)).astype(np.int32)

def _rotate_and_translate(points: np.ndarray, angle: float, center: Tuple[int, int]) -> np.ndarray:
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    return (points @ rot.T + np.array(center)).astype(np.int32)

def generate_shape(
    shape: str = "circle",
    h: int = 256,
    w: int = 256,
    size: float = 50,
    angle: float = 0.0,
    bg: int = 0,
    color: int = 255,
    blur_sigma_range: Tuple[float, float] = (0.0, 2.5), 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a clean image with a single shape: 'circle', 'rectangle', or 'triangle'.
    Ensures the shape fits entirely in the image, even after rotation.
    Applies random Gaussian blur with sigma in blur_sigma_range.
    """
    rng = rng or np.random.default_rng()
    img = np.full((h, w), bg, dtype=np.uint8)

    # --- Compute safe padding ---
    if shape == "circle":
        safe_pad = int(size) + 1
    elif shape == "rectangle":
        rect_w = size * 1.5
        rect_h = size
        safe_pad = int(np.sqrt(rect_w**2 + rect_h**2) / 2) + 1
    else:
        safe_pad = int(np.sqrt(2) * size) + 1

    cx = int(rng.integers(safe_pad, w - safe_pad))
    cy = int(rng.integers(safe_pad, h - safe_pad))

    if shape == "circle":
        r = int(size)
        cv2.circle(img, (cx, cy), r, color, -1)

    elif shape == "rectangle":
        rect_w = size * 1.5
        rect_h = size
        rect = np.array([
            [-rect_w / 2, -rect_h / 2],
            [ rect_w / 2, -rect_h / 2],
            [ rect_w / 2,  rect_h / 2],
            [-rect_w / 2,  rect_h / 2]
        ], dtype=np.float32)
        pts = _rotate_and_translate(rect, angle, (cx, cy))
        cv2.fillPoly(img, [pts], color)

    elif shape == "triangle":
        tri = np.array([
            [0, -1.15 * size],
            [ size, 0.65 * size],
            [-size, 0.65 * size]
        ], dtype=np.float32)
        pts = _rotate_and_translate(tri, angle, (cx, cy))
        cv2.fillPoly(img, [pts], color)

    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # --- Apply random Gaussian blur ---
    sigma = float(rng.uniform(*blur_sigma_range))
    if sigma > 0:
        # kernel size: ensure it's odd and large enough for sigma
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    return img, shape

def my_canny(img: np.ndarray, low_thresh: float = 100, high_thresh: float = 200):
    gaussian_kernel = (1/16) * np.array([[1, 2, 1],
                                         [2, 4, 2],
                                         [1, 2, 1]], np.float32)
    smoothed = cv2.filter2D(img, -1, gaussian_kernel)

    dx_kernel = np.array([[-1, 0, 1]], np.float32)
    dy_kernel = dx_kernel.T
    Ix = cv2.filter2D(smoothed.astype(np.float32), -1, dx_kernel)
    Iy = cv2.filter2D(smoothed.astype(np.float32), -1, dy_kernel)
    
    magnitude = np.sqrt(Ix**2 + Iy**2)
    orientation = np.arctan2(Iy, Ix)    # radians, range [-π, π]
    
    nms_result = np.zeros_like(magnitude)
    
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            angle_deg = np.degrees(orientation[i, j]) % 180
            
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                neighbor1 = magnitude[i, j-1]
                neighbor2 = magnitude[i, j+1]
            elif 22.5 <= angle_deg < 67.5:
                neighbor1 = magnitude[i-1, j-1]
                neighbor2 = magnitude[i+1, j+1]
            elif 67.5 <= angle_deg < 112.5:
                neighbor1 = magnitude[i-1, j]
                neighbor2 = magnitude[i+1, j]
            else:  # 112.5 <= angle_deg < 157.5
                neighbor1 = magnitude[i-1, j+1]
                neighbor2 = magnitude[i+1, j-1]
            
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                nms_result[i, j] = magnitude[i, j]

    nms_nonzero = nms_result[nms_result > 0]
    if len(nms_nonzero) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    
    high_thresh_adaptive = np.percentile(nms_nonzero, 70)
    low_thresh_adaptive = 0.4 * high_thresh_adaptive
    
    canny_image = np.zeros_like(img, dtype=np.uint8)
    
    strong_edges = (nms_result >= high_thresh_adaptive)
    canny_image[strong_edges] = 255
    
    for _ in range(10):
        changed = False
        for i in range(1, nms_result.shape[0]-1):
            for j in range(1, nms_result.shape[1]-1):
                if (low_thresh_adaptive <= nms_result[i, j] < high_thresh_adaptive) and (canny_image[i, j] != 255):
                    neighbors = canny_image[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 255):
                        canny_image[i, j] = 255
                        changed = True
        
        if not changed:
            break
    
    return canny_image


def shape_identifier(img: np.ndarray):
    canny_image = my_canny(img)
    ys, xs = np.where(canny_image > 0)
    row_start_index = ys.min()
    row_end_index = ys.max()
    column_start_index = xs.min()
    column_end_index = xs.max()
    
    obj_height = row_end_index - row_start_index + 1
    obj_width = column_end_index - column_start_index + 1
    
    circle_threshold = 5
    triangle_threshold = 10
    
    if abs(obj_height - obj_width) < circle_threshold:
        flag = "circle"
    elif abs(obj_height - 0.5*np.tan(np.pi/3)*obj_width) < triangle_threshold or abs(obj_width - 0.5*np.tan(np.pi/3)*obj_height) < triangle_threshold:
        flag = "triangle"
    else:
        flag = "rectangle"
    
    return flag

def shape_identifier_test():
    shapes = ["circle", "rectangle", "triangle"]
    print("--------------------------------")
    for shape in shapes:
        print(f"Testing {shape} shape")
        correct_count = 0
        for i in range(100):
            img, shape = generate_shape(shape=shape)
            if shape == "triangle":
                pass
            student_shape = shape_identifier(img)
            if student_shape == shape:
                correct_count += 1
            else:
                print(f"Student shape: {student_shape}, Correct shape: {shape}")
        print(f"Identifier accuracy: {correct_count/100*100}%\n")

if __name__ == "__main__":
    shape_identifier_test()