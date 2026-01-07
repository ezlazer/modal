import modal
import io
import os

# 1. Define the Serverless Environment
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]", 
        "torch",
        "torchvision",
        "transformers",
        "rembg[gpu]",
        "trimesh",
        "numpy<2.0",
        "scipy",
        "Pillow",
        "huggingface_hub",
        "onnxruntime-gpu" # Explicitly ensure GPU runtime for rembg
    )
    # Bake models into the image to speed up cold starts
    .run_commands(
        # Download u2net for rembg
        "python -c 'from rembg import new_session; new_session(\"u2net\")'",
        # Pre-download depth model
        "python -c 'from transformers import AutoModelForDepthEstimation, AutoImageProcessor; "
        "model_id = \"depth-anything/Depth-Anything-V2-Large-hf\"; "
        "AutoModelForDepthEstimation.from_pretrained(model_id); "
        "AutoImageProcessor.from_pretrained(model_id)'"
    )
    # Suppress TensorRT warnings since we are happy with CUDA fallback
    .env({
        "ORT_TENSORRT_FP16_ENABLE": "0",
        "REMBG_ONNX_PROVIDERS": "CUDAExecutionProvider,CPUExecutionProvider"
    })
)

app = modal.App("depth-to-glb-service", image=image)

# 2. Define the GPU Class
# UPDATED: 'scaledown_window' replaces 'container_idle_timeout'
@app.cls(gpu="A10", timeout=600, scaledown_window=300, enable_memory_snapshot=True)
class Model:
    @modal.enter()
    def _load_models(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from rembg import new_session

        print("🔄 Loading models to GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "depth-anything/Depth-Anything-V2-Large-hf"

        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=torch.float16).to(self.device)
        self.rembg_session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print("✅ Models loaded successfully.")

    @modal.method()
    def process(self, image_bytes, depth_scale=1.0, mesh_detail=0.5, smooth_iters=0,
                alpha_threshold=64, edge_erosion=1, depth_blur=0.5, max_depth_gradient=0.4,
                edge_blend_distance=10):
        import torch
        import numpy as np
        import trimesh
        import trimesh.transformations as transformations
        from PIL import Image
        from rembg import remove
        from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt

        # --- Lazy Load Check ---
        # This prevents the AttributeError by ensuring models exist before usage
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
            self._load_models()

        print("Processing request...")
        original_img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Auto-Downsize
        max_dimension = 1080
        if max(original_img_pil.size) > max_dimension:
            print(f"Downsizing from {original_img_pil.size}...")
            original_img_pil.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

        # Remove Background
        print("Removing background...")
        rgba_high_res = remove(original_img_pil, session=self.rembg_session)

        # Estimate Depth
        inputs = self.processor(images=original_img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Prepare Mesh
        width, height = original_img_pil.size
        aspect_ratio = width / height
        target_width = int(width * mesh_detail)
        target_height = int(height * mesh_detail)

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        depth_array = prediction.squeeze().cpu().numpy().astype(np.float32)

        # Normalize (ensure float32 for scipy compatibility)
        depth_min, depth_max = depth_array.min(), depth_array.max()
        normalized_depth = ((depth_array - depth_min) / (depth_max - depth_min)).astype(np.float32)

        # Apply Gaussian blur to smooth depth transitions
        if depth_blur > 0:
            normalized_depth = gaussian_filter(normalized_depth, sigma=depth_blur)

        # Depth gradient clamping to prevent steep edges
        if max_depth_gradient > 0:
            grad_y, grad_x = np.gradient(normalized_depth)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            steep_mask = gradient_magnitude > max_depth_gradient

            # For steep areas, blend toward local average to soften edges
            if np.any(steep_mask):
                # Apply additional smoothing only to steep gradient areas
                smoothed = gaussian_filter(normalized_depth, sigma=2.0)
                normalized_depth = np.where(steep_mask, smoothed, normalized_depth)

        # Edge depth blending - fade depth toward median near boundaries
        if edge_blend_distance > 0:
            # Get alpha mask at depth resolution
            alpha_resized = np.array(rgba_high_res.resize((target_width, target_height), Image.NEAREST))[:, :, 3]
            alpha_mask = alpha_resized > alpha_threshold

            # Distance from each pixel to nearest transparent pixel (edge)
            distance_from_edge = distance_transform_edt(alpha_mask).astype(np.float32)

            # Create smooth blend weight (0 at edge, 1 at edge_blend_distance)
            blend_weight = np.clip(distance_from_edge / edge_blend_distance, 0, 1)

            # Blend toward median depth at edges
            subject_depths = normalized_depth[alpha_mask]
            if len(subject_depths) > 0:
                median_depth = np.median(subject_depths)
                normalized_depth = normalized_depth * blend_weight + median_depth * (1 - blend_weight)

        # Generate Grid
        x_range = aspect_ratio
        x = np.linspace(-x_range, x_range, target_width)
        y = np.linspace(1, -1, target_height)
        xv, yv = np.meshgrid(x, y)
        
        u = np.linspace(0, 1, target_width)
        v = np.linspace(1, 0, target_height)
        uv_u, uv_v = np.meshgrid(u, v)

        z = normalized_depth * depth_scale
        vertices = np.column_stack((xv.flatten(), yv.flatten(), z.flatten()))
        uvs = np.column_stack((uv_u.flatten(), uv_v.flatten()))

        # Faces
        indices = []
        for r in range(target_height - 1):
            for c in range(target_width - 1):
                i = r * target_width + c
                indices.append([i, i + 1, i + target_width])
                indices.append([i + 1, i + target_width + 1, i + target_width])
        faces = np.array(indices)

        # Texture Visuals
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=rgba_high_res)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)

        # Cleanup Transparency with edge erosion
        mask_img = rgba_high_res.resize((target_width, target_height), Image.NEAREST)
        alpha_array = np.array(mask_img)[:, :, 3]

        # Apply edge erosion to pull boundaries inward and remove fringe pixels
        if edge_erosion > 0:
            alpha_mask = alpha_array > alpha_threshold
            eroded_mask = binary_erosion(alpha_mask, iterations=edge_erosion)
            alpha_array = np.where(eroded_mask, alpha_array, 0)

        alpha_values = alpha_array.flatten()
        vertex_mask = alpha_values > alpha_threshold
        face_mask = vertex_mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()

        # Smoothing
        if smooth_iters > 0:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)

        # Re-orient (Upright)
        matrix = transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        mesh.apply_transform(matrix)

        # Export to Bytes
        output_io = io.BytesIO()
        mesh.export(output_io, file_type="glb")
        return output_io.getvalue()

# 3. Define the Web Endpoint
from fastapi import Header, HTTPException

@app.function(secrets=[modal.Secret.from_name("api-auth")])
@modal.fastapi_endpoint(method="POST")
def generate_3d(item: dict, authorization: str = Header(None)):
    import base64
    import hmac

    # Validate API key
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured")

    # Extract Bearer token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_key = authorization[7:]  # Strip "Bearer "
    if not hmac.compare_digest(provided_key, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # We expect a JSON body: {"image": "base64_string", "scale": 1.5, ...}
    image_data = base64.b64decode(item["image"])
    scale = item.get("scale", 1.0)
    detail = item.get("detail", 0.5)
    smooth = item.get("smooth", 0)
    alpha_threshold = item.get("alpha_threshold", 64)
    edge_erosion = item.get("edge_erosion", 1)
    depth_blur = item.get("depth_blur", 0.5)
    max_depth_gradient = item.get("max_depth_gradient", 0.4)
    edge_blend_distance = item.get("edge_blend_distance", 10)

    # Call the GPU class
    glb_bytes = Model().process.remote(
        image_data, scale, detail, smooth,
        alpha_threshold, edge_erosion, depth_blur, max_depth_gradient,
        edge_blend_distance
    )

    # Return the GLB as base64
    return {"glb": base64.b64encode(glb_bytes).decode("utf-8")}