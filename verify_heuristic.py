
import requests
import numpy as np
from PIL import Image
import io

def test_image(name, img_data):
    url = 'http://127.0.0.1:8000/predict'
    files = {'image': ('test.png', img_data, 'image/png')}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            data = response.json()
            print(f"[{name}] Response: {data['disease']}, Confidence: {data['confidence']}")
            if data['disease'] == "Not a Paddy Leaf":
                print(f"[{name}] SUCCESS: Rejected as expected.")
            else:
                print(f"[{name}] FAILURE: Accepted as {data['disease']}")
        else:
            print(f"[{name}] Error: {response.status_code}")
    except Exception as e:
        print(f"[{name}] Exception: {e}")

# 1. Random Noise (should be rejected)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)
test_image("Random Noise", img_byte_arr.getvalue())

# 2. Skin Tone (should be rejected)
# Skin is often orange-ish. RGB approx (255, 200, 150)
skin_img = Image.new('RGB', (224, 224), color=(255, 200, 150))
skin_byte_arr = io.BytesIO()
skin_img.save(skin_byte_arr, format='PNG')
skin_byte_arr.seek(0)
test_image("Skin Tone", skin_byte_arr.getvalue())

# 3. Green Square (should be accepted by heuristic, then maybe model predicts something)
green_img = Image.new('RGB', (224, 224), color=(0, 128, 0))
green_byte_arr = io.BytesIO()
green_img.save(green_byte_arr, format='PNG')
green_byte_arr.seek(0)
print(f"[Green Square] sending...")
# This might return "Healthy Rice Leaf" or something, but hopefully heuristic passes.
# We just want to check if heuristic BLOCKS it or not.
# If it returns "Not a Paddy Leaf" with 0.0 confidence, then heuristic blocked it (which would be wrong for a green leaf proxy, but maybe okay for a solid block).
# Wait, solid green block matches heuristic (ratio 1.0 > 0.1). So it should pass heuristic.
# Then model predicts.
test_image("Green Square", green_byte_arr.getvalue())
