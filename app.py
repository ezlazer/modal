import modal
import io
import os

# 1. Define the Serverless Environment
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchvision",
        "rembg[gpu]",
        "trimesh",
        "numpy<2.0",
        "scipy",
        "Pillow",
        "huggingface_hub",
        "onnxruntime-gpu",  # Explicitly ensure GPU runtime for rembg
    )
    .run_commands(
        # Download u2net for rembg
        "python -c 'from rembg import new_session; new_session(\"u2net\")'",
        # Pre-download Sapiens depth model (1B) from HuggingFace
        "python -c 'from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id=\"facebook/sapiens-depth-1b-torchscript\", "
        "filename=\"sapiens_1b_render_people_epoch_88_torchscript.pt2\")'"
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
        from huggingface_hub import hf_hub_download
        from rembg import new_session

        try:
            print("🔄 Loading models to GPU...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {self.device}")

            print("Loading Meta Sapiens Depth 1B model...")
            # Download and load TorchScript model
            model_path = hf_hub_download(
                repo_id="facebook/sapiens-depth-1b-torchscript",
                filename="sapiens_1b_render_people_epoch_88_torchscript.pt2"
            )
            self.depth_model = torch.jit.load(model_path)
            self.depth_model.eval()
            self.depth_model.to(self.device)
            print("Loading rembg session...")
            self.rembg_session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            print("✅ Models loaded successfully.")
        except Exception as e:
            import traceback
            print(f"❌ Model loading failed: {e}")
            print(traceback.format_exc())
            raise

    @modal.method()
    def process(self, image_bytes, depth_scale=1.0, mesh_detail=0.5, smooth_iters=0,
                alpha_threshold=64, edge_erosion=0, depth_blur=1.0, max_depth_gradient=0.2,
                edge_blend_distance=25, extrude=True):
        import torch
        import numpy as np
        import trimesh
        import trimesh.transformations as transformations
        from PIL import Image
        from rembg import remove
        from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt

        # --- Lazy Load Check ---
        # This prevents the AttributeError by ensuring models exist before usage
        if not hasattr(self, 'depth_model'):
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

        # Estimate Depth using Meta Sapiens 1B (optimized for human images)
        # Sapiens requires 1024x768 (H x W) input resolution
        print("Estimating depth with Meta Sapiens 1B...")

        # Store original dimensions
        width, height = original_img_pil.size
        aspect_ratio = width / height

        # Preprocess for Sapiens: resize to 1024x768 and normalize
        sapiens_h, sapiens_w = 1024, 768
        img_resized = original_img_pil.resize((sapiens_w, sapiens_h), Image.BILINEAR)

        # Convert to tensor and normalize (ImageNet normalization)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_np - mean) / std

        # Convert to tensor: [H, W, C] -> [1, C, H, W]
        input_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        input_tensor = input_tensor.to(self.device)

        # Run Sapiens depth inference
        with torch.inference_mode():
            depth_output = self.depth_model(input_tensor)
            # Resize back to original image dimensions
            depth_output = torch.nn.functional.interpolate(
                depth_output, size=(height, width), mode="bilinear", align_corners=False
            )

        predicted_depth = depth_output.squeeze().cpu().numpy()
        print(f"Sapiens raw depth range: {predicted_depth.min():.4f} to {predicted_depth.max():.4f}")

        # Prepare Mesh
        target_width = int(width * mesh_detail)
        target_height = int(height * mesh_detail)

        # Resize depth map to target mesh resolution
        depth_tensor = torch.from_numpy(predicted_depth).unsqueeze(0).unsqueeze(0).float()
        depth_resized = torch.nn.functional.interpolate(
            depth_tensor,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        depth_array = depth_resized.squeeze().numpy().astype(np.float32)

        # Normalize and invert depth for relief effect
        # Sapiens outputs relative depth (closer = higher values)
        depth_min, depth_max = depth_array.min(), depth_array.max()
        depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
        # Normalize to 0-1 range
        normalized_depth = ((depth_array - depth_min) / depth_range).astype(np.float32)
        print(f"Sapiens normalized depth range: {depth_min:.4f} to {depth_max:.4f}")

        # Get alpha mask at depth resolution
        alpha_resized = np.array(rgba_high_res.resize((target_width, target_height), Image.NEAREST))[:, :, 3]
        alpha_mask = alpha_resized > alpha_threshold

        # DILATE DEPTH (Extrapolate to edges)
        # Instead of fading/smoothing at the edge (which causes streaks), we want to preserve 
        # the foreground depth all the way to the cut-out line.
        # We fill invalid (background) pixels with the nearest valid (foreground) pixel's depth.
        
        # Invert mask: valid pixels are 0, invalid are 1 for distance transform
        invalid_mask = ~alpha_mask
        nearest_indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
        # Map every pixel to its nearest valid pixel from the original mask
        normalized_depth = normalized_depth[tuple(nearest_indices)]
        
        # Apply Gaussian blur to smooth depth transitions (internal smoothing)
        if depth_blur > 0:
            normalized_depth = gaussian_filter(normalized_depth, sigma=depth_blur)

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
        
        # Create standard grid faces
        indices = []
        for r in range(target_height - 1):
            for c in range(target_width - 1):
                i = r * target_width + c
                indices.append([i, i + 1, i + target_width])
                indices.append([i + 1, i + target_width + 1, i + target_width])
        faces = np.array(indices)
        
        # Filter faces to ensure they refer to valid vertices
        # This handles any discrepancies in size/indexing
        if len(vertices) > 0:
            valid_faces = faces.max(axis=1) < len(vertices)
            faces = faces[valid_faces]

        # Texture Visuals (Create initial mesh)
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=rgba_high_res)
        # Verify process=False to keep vertices aligned with our grid
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

        # Cleanup Transparency with edge erosion
        # Reuse the alpha logic
        # Ensure alpha_array matches VERTEX count
        # If sizes differ, we must resize mask to match vertices size (target_width * target_height)
        
        expected_size = len(vertices)
        if len(alpha_resized.flatten()) != expected_size:
             print(f"WARNING: Alpha mask size {len(alpha_resized.flatten())} != Vertices {expected_size}. Resizing mask.")
             # Fallback to recalculate based on current target dims (which defined vertices)
             mask_img = rgba_high_res.resize((target_width, target_height), Image.NEAREST)
             alpha_array = np.array(mask_img)[:, :, 3]
        else:
             alpha_array = alpha_resized

        # Apply edge erosion to pull boundaries inward and remove fringe pixels
        if edge_erosion > 0:
             alpha_mask_eroded = binary_erosion(alpha_array > alpha_threshold, iterations=edge_erosion)
             alpha_array = np.where(alpha_mask_eroded, alpha_array, 0)
        
        vertex_mask = alpha_array.flatten() > alpha_threshold
        
        # Try-catch the cleanup process to return a valid mesh even if filtering fails
        try:
             # Ensure we don't crash if bad faces slipped through (double check)
             # We use the MINIMUM size to be absolutely safe against any mismatch
             safe_limit = min(len(vertices), len(vertex_mask))
             
             current_faces = mesh.faces
             valid_idx_ref = current_faces.max(axis=1) < safe_limit
             current_faces = current_faces[valid_idx_ref]
             
             # Now access vertex_mask safe in the knowledge that indices are < len(vertex_mask)
             # AND that indices refer to valid vertices < len(vertices)
             face_mask = vertex_mask[current_faces].all(axis=1)
             
             # Update mesh
             mesh.faces = current_faces[face_mask]
             mesh.remove_unreferenced_vertices()
        except IndexError as e:
             print(f"WARNING: Cleanup step failed with {e}. Skipping transparency cleanup.")
             # Continue with the mesh as-is (extruded sides still present)

        # Smoothing
        if smooth_iters > 0:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
            except IndexError as e:
                print(f"WARNING: Smoothing failed with {e}. Skipping smoothing.")

        # EXTRUSION (Solidify)
        # Find naked edges (boundary loops) and extrude them backwards
        if extrude:
          try:
            # unique_edges counts how many times an edge is shared. Boundary edges appear once.
            # However, trimesh.grouping.group_rows is slow.
            # mesh.outline() returns the boundary as paths, but we want the edges to construct faces.

            # Use trimesh's built-in boundary finding
            boundary_edges = mesh.edges_unique[
                trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
            ]
            
            if len(boundary_edges) > 0:
                # We need to orient these edges correctly or just create double-sided walls
                # For simplicity, we create quads connecting the edge to a back-projected edge
                
                # Retrieve vertex coordinates
                v_start_indices = boundary_edges[:, 0]
                v_end_indices = boundary_edges[:, 1]
                
                v_start = mesh.vertices[v_start_indices]
                v_end = mesh.vertices[v_end_indices]
                
                # Create corresponding back vertices (z = 0 or min_z)
                # Let's align back plane with the minimum Z of the object to simulate a flat back
                min_z = mesh.vertices[:, 2].min()
                # Or just 0 if we assume the plane is at 0. Let's use proportional thickness.
                # Actually, let's offset by a fixed amount relative to scale or just go to 0.
                target_z = min_z - (0.05 * depth_scale) # Slight offset behind min
                
                v_start_back = v_start.copy()
                v_start_back[:, 2] = target_z
                
                v_end_back = v_end.copy()
                v_end_back[:, 2] = target_z
                
                # New vertices for the rim
                # Note: This is a hacky way to add geometry. A cleaner way is to append vertices to the mesh.
                # But treating them as a separate mesh and joining is safer.
                
                # Let's create a list of side faces (quads -> 2 tris)
                side_vertices = []
                side_faces = []
                
                # We can just construct a ribbon.
                # Vertices: 4 per edge (Start, End, StartBack, EndBack)
                # This duplicates vertices but ensures normals are crisp (flat shading for sides)
                
                existing_v_count = len(mesh.vertices)
                
                # Flatten the quad generation
                # For each edge i:
                # V0: v_start[i], V1: v_end[i], V2: v_end_back[i], V3: v_start_back[i]
                # Faces: 0-1-2, 0-2-3 (check winding order)
                
                num_edges = len(boundary_edges)
                
                # Prepare data arrays
                # current boundary vertices (top)
                v_top_0 = v_start
                v_top_1 = v_end
                # back vertices (bottom)
                v_bot_0 = v_start_back
                v_bot_1 = v_end_back
                
                # Stack them: [N,3]
                # We will create a new mesh for the sides to avoid complex indexing math
                # Then concat
                
                # 4 vertices per edge segment
                sides_v = np.concatenate([v_top_0, v_top_1, v_bot_1, v_bot_0]) # [4*N, 3] logic
                # Actually let's reshape to organize
                # 0:start, 1:end, 2:end_back, 3:start_back
                
                # We need a predictable order.
                # Let's just create a separate mesh for sides using TextureVisuals with a solid color (e.g., black or average edge color)
                # To keep it simple, we'll map the sides to a dark pixel in the texture or just white.
                
                # Constructing indices
                # N edges. 
                # vertices 0..N-1 are starts
                # vertices N..2N-1 are ends
                # vertices 2N..3N-1 are end_backs
                # vertices 3N..4N-1 are start_backs
                
                s_v = np.concatenate([v_start, v_end, v_end_back, v_start_back])
                offset = 0
                f_A = np.arange(num_edges)
                f_B = np.arange(num_edges) + num_edges
                f_C = np.arange(num_edges) + 2*num_edges
                f_D = np.arange(num_edges) + 3*num_edges
                
                # Two triangles per quad
                # Tri 1: Top-Start -> Top-End -> Bot-End
                t1 = np.column_stack([f_A, f_B, f_C])
                # Tri 2: Top-Start -> Bot-End -> Bot-Start
                t2 = np.column_stack([f_A, f_C, f_D])
                
                s_f = np.concatenate([t1, t2])
                
                # UVs for sides: Map to (0,0) or a specific color. 
                # Let's map to the edge of the texture (somewhere transparent or edge color).
                # Ideally, we stretch the edge texture down (streak), but users hated streaks.
                # So let's map to a solid color.
                # We can just set UVs to (0,0) and hope (0,0) is a good color, or find a controlled spot.
                # Better: Use vertex colors? Trimesh handles mixed visuals poorly sometimes.
                # Let's just assume 0.5, 0.5 is safe or map to the same UV as the top vertex to EXTEND texture (streaks) - wait, plan said no streaks.
                # Plan said "Solid side walls". Solid usually implies a color.
                # Let's picking the UV of the top vertex for the top, and same UV for bottom. This creates streaks.
                # Users wanted "clean edges". 
                # Maybe simple black/grey sides? 
                # Let's try mapping to a generic grey point if possible, or just replicate edge UVs which creates streaks but valid ones.
                # "Streaky edges" in the prompt image looked like bad displacement.
                # Clean extrusion usually looks like streaks of the edge color.
                # Let's use the edge UVs for now, as that's standard extrusion.
                
                # Fetch UVs for boundary vertices (with fallback for index errors)
                try:
                    if mesh.visual.uv is not None and len(mesh.visual.uv) > max(v_start_indices.max(), v_end_indices.max()):
                        uv_start = mesh.visual.uv[v_start_indices]
                        uv_end = mesh.visual.uv[v_end_indices]
                    else:
                        # Fallback: use center of texture
                        uv_start = np.full((num_edges, 2), 0.5)
                        uv_end = np.full((num_edges, 2), 0.5)
                except (IndexError, AttributeError):
                    # Fallback: use center of texture for solid color sides
                    uv_start = np.full((num_edges, 2), 0.5)
                    uv_end = np.full((num_edges, 2), 0.5)

                s_uv = np.concatenate([uv_start, uv_end, uv_end, uv_start])
                
                side_mesh = trimesh.Trimesh(vertices=s_v, faces=s_f, visual=trimesh.visual.TextureVisuals(uv=s_uv, image=rgba_high_res))
                
                # Merge
                mesh = trimesh.util.concatenate([mesh, side_mesh])
                
                # Back face? (optional, maybe overkill for now, sides are usually enough)
                # To make it watertight calculate a hull or fill holes? complex.
                # Sides are good enough for "relief" look.
          except (IndexError, ValueError) as e:
            print(f"WARNING: Extrusion failed with {e}. Skipping extrusion.")

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
    edge_erosion = item.get("edge_erosion", 0)
    depth_blur = item.get("depth_blur", 1.0)
    max_depth_gradient = item.get("max_depth_gradient", 0.2)
    edge_blend_distance = item.get("edge_blend_distance", 25)

    # Call the GPU class
    try:
        glb_bytes = Model().process.remote(
            image_data, scale, detail, smooth,
            alpha_threshold, edge_erosion, depth_blur, max_depth_gradient,
            edge_blend_distance, extrude=item.get("extrude", True)
        )
    except Exception as e:
        import traceback
        error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # Return the GLB as base64
    return {"glb": base64.b64encode(glb_bytes).decode("utf-8")}