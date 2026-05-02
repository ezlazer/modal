import requests
import base64
import os
import sys

# 1. Load your image
if len(sys.argv) < 2:
    print("Usage: python process.py <image_path>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
with open(INPUT_FILE, "rb") as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# 2. Prepare Payload
payload = {
    "image": image_b64,
    "keep_background": False,   # set True to retain background (skips PhotoRoom + silhouette cut)
    "depth_ratio": 0.30,        # Z extent as fraction of max(X,Y) extent
    "alpha_threshold": 160,     # higher threshold excludes semi-transparent stragglers
}

# 3. Send to Modal (The URL you got from 'modal deploy')
MODAL_URL = "https://info-38385--depth-to-glb-v3-generate-3d.modal.run"
headers = {"Authorization": "Bearer yGaiBLs46frDXusfw7aQqIOHtZVx5txA"}
response = requests.post(MODAL_URL, json=payload, headers=headers)

# 4. Save Result
print(f"Status code: {response.status_code}")
print(f"Response headers: {dict(response.headers)}")
print(f"Response text: {response.text[:2000] if response.text else 'Empty response'}")

if response.status_code != 200:
    print(f"Error: {response.text}")
    exit(1)

result = response.json()
glb_data = base64.b64decode(result["glb"])

output_file = os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".glb"
with open(output_file, "wb") as f:
    f.write(glb_data)

print(f"✅ GLB received and saved as {output_file}!")