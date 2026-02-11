
import requests
import numpy as np
from PIL import Image
import io

# Create a random noise image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

url = 'http://127.0.0.1:8000/predict'
files = {'image': ('test.png', img_byte_arr, 'image/png')}

try:
    response = requests.post(url, files=files)
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data}")
        if data['disease'] == "Not a Paddy Leaf":
            print("SUCCESS: Correctly identified as 'Not a Paddy Leaf'")
        else:
            print(f"FAILURE: Identified as {data['disease']} with confidence {data['confidence']}")
    else:
        print(f"Error: Status code {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Exception: {e}")
