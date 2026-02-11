
import numpy as np
from PIL import Image
import io

def checks_leaf_characteristics(img):
    """
    Checks if the image has characteristics of a paddy leaf.
    1. Color Analysis: Check for dominance of green/yellow/brown hues common in leaves.
    """
    # Convert to HSV for better color segmentation
    img_hsv = img.convert('HSV')
    img_np = np.array(img_hsv)
    
    # H: 0-255, S: 0-255, V: 0-255
    # Green range approx: H(35-85) -> Pillow H is 0-255 map to 0-360? No, Pillow is 0-255.
    # OpenCV H is 0-179. Pillow H is 0-255.
    # Green is roughly 60-180 degrees -> 42-127 in 0-255 scale.
    # Yellow is roughly 40-60 degrees -> 28-42.
    # Brown/Dead leaf might be orange/reddish.
    
    # Let's count "leaf-like" pixels.
    # Saturation > 20 (avoid grays/whites/blacks being counted as color)
    # Value > 20 (avoid very dark)
    
    H = img_np[:,:,0]
    S = img_np[:,:,1]
    V = img_np[:,:,2]
    
    # Simple Green/Yellow filter (broad range for leaves)
    # 20 (yellow-ish) to 130 (green/cyan)
    leaf_mask = (H > 20) & (H < 130) & (S > 25) & (V > 25)
    
    leaf_pixels = np.sum(leaf_mask)
    total_pixels = img_np.shape[0] * img_np.shape[1]
    
    ratio = leaf_pixels / total_pixels
    print(f"Leaf-like pixel ratio: {ratio:.2f}")
    
    return ratio > 0.10 # Reduced to 10% to be safe, but mostly non-leaves should be very low if background is plain.
    # Wait, if background is white/black, and leaf is small, ratio might be small.
    # But usually leaf images are close-ups.
    
# Create a dummy image (Random Noise - should be low)
noise_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
print("Testing Noise Image:")
checks_leaf_characteristics(noise_img)

# Create a dummy "Green" image
green_img = Image.new('RGB', (224, 224), color=(0, 128, 0))
print("Testing Green Image:")
checks_leaf_characteristics(green_img)

# Create a dummy "Red" image
red_img = Image.new('RGB', (224, 224), color=(255, 0, 0))
print("Testing Red Image:")
checks_leaf_characteristics(red_img)

# Create a dummy "Face" like image (skin tone approx)
# Skin is often orange-ish. H around 10-20.
skin_img = Image.new('RGB', (224, 224), color=(255, 200, 150))
print("Testing Skin-tone Image:")
checks_leaf_characteristics(skin_img)
