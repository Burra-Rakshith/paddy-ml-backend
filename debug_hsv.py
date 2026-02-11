
from PIL import Image
import numpy as np

# Create green image
green_img = Image.new('RGB', (1, 1), color=(0, 128, 0))
hsv_img = green_img.convert('HSV')
hsv_val = hsv_img.getpixel((0, 0))
print(f"Green (0, 128, 0) HSV: {hsv_val}")

# Create bright green
bright_green_img = Image.new('RGB', (1, 1), color=(0, 255, 0))
hsv_val_bright = bright_green_img.convert('HSV').getpixel((0, 0))
print(f"Bright Green (0, 255, 0) HSV: {hsv_val_bright}")

# Create yellow
yellow_img = Image.new('RGB', (1, 1), color=(255, 255, 0))
hsv_val_yellow = yellow_img.convert('HSV').getpixel((0, 0))
print(f"Yellow (255, 255, 0) HSV: {hsv_val_yellow}")
