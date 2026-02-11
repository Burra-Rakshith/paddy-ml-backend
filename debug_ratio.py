
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())

# Mock main.py dependencies if needed?
# Actually main.py loads model on import.
# We can just extract the function code to a separate file or copy-paste it here to test logic.
# Copy-pasting is safer to avoid side effects of importing main.

def is_paddy_leaf(img):
    """
    Simple heuristic to check if image has paddy leaf characteristics (Green/Yellow dominant).
    """
    try:
        # Convert to HSV
        img_hsv = img.convert('HSV')
        img_np = np.array(img_hsv)
        
        # Extract channels
        H = img_np[:,:,0]
        S = img_np[:,:,1]
        V = img_np[:,:,2]
        
        # Define Green/Yellow range (approximate for PIL HSV 0-255)
        # Green/Yellow hue is roughly 20 to 130
        # Saturation and Value should be sufficient to avoid black/white/gray
        leaf_mask = (H > 20) & (H < 130) & (S > 25) & (V > 25)
        
        leaf_pixels = np.sum(leaf_mask)
        total_pixels = img_np.shape[0] * img_np.shape[1]
        
        ratio = leaf_pixels / total_pixels
        print(f"Leaf pixel ratio: {ratio:.4f}")
        
        # Threshold: at least 10% of the image should be "leaf-colored"
        return ratio > 0.10
    except Exception as e:
        print(f"Error in heuristic check: {e}")
        return True # Fallback to model if check fails

# Test Green Image
green_img = Image.new('RGB', (224, 224), color=(0, 128, 0))
print("Testing Green Image (0, 128, 0):")
print(f"Result: {is_paddy_leaf(green_img)}")

# Test Bright Green
bright_green_img = Image.new('RGB', (224, 224), color=(0, 255, 0))
print("Testing Bright Green Image (0, 255, 0):")
print(f"Result: {is_paddy_leaf(bright_green_img)}")

# Test Yellow
yellow_img = Image.new('RGB', (224, 224), color=(255, 255, 0))
print("Testing Yellow Image (255, 255, 0):")
print(f"Result: {is_paddy_leaf(yellow_img)}")
