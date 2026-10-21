import cv2
import numpy as np

# quality_analyzer.py
def measure_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    dark_ch = np.min(image / 255.0, axis=2)
    haze_level = np.mean(dark_ch)
    noise_est = np.std(gray - cv2.GaussianBlur(gray, (5,5), 0))
    return brightness, contrast, sharpness, haze_level, noise_est

# param_selector.py
def select_parameters(brightness, contrast, sharpness, haze_level, noise):
    clahe_clip = 5.0 if brightness < 80 else (3.5 if brightness < 120 else 1.5)
    bilateral_s = 75 if noise > 15 else 50
    apply_dehaze = haze_level > 0.4
    apply_sharp = sharpness < 100
    return clahe_clip, bilateral_s, apply_dehaze, apply_sharp

# dehazer.py (simplified for now)
def dark_channel_prior(image, patch_size=15):
    # This is a placeholder. You would implement actual DCP here.
    # For now, just return a slightly enhanced image to simulate dehazing
    return cv2.convertScaleAbs(image, alpha=1.2, beta=10)

# enhancer.py
def apply_clahe(image, clip_limit):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def apply_bilateral(image, sigma):
    return cv2.bilateralFilter(image, 9, sigma, sigma)

def apply_sharpening(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(image, -1, kernel)

# feature_eval.py
def count_orb(image, n=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n)
    kp, _ = orb.detectAndCompute(gray, None)
    return len(kp)

# The Adaptive Pipeline combining the above
def adaptive_pipeline(img):
    brightness, contrast, sharpness, haze_level, noise = measure_quality(img)
    clahe_clip, bilateral_s, apply_dehaze, apply_sharp = select_parameters(brightness, contrast, sharpness, haze_level, noise)
    
    enhanced = img.copy()
    
    if apply_dehaze:
        enhanced = dark_channel_prior(enhanced)
        
    enhanced = apply_clahe(enhanced, clahe_clip)
    enhanced = apply_bilateral(enhanced, bilateral_s)
    
    if apply_sharp:
        enhanced = apply_sharpening(enhanced)
        
    return enhanced
