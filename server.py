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
        "transformers>=4.45",
        "accelerate",
        "requests",
        "trimesh",
        "numpy<2.0",
        "scipy",
        "Pillow",
        "huggingface_hub",
        "opencv-python-headless",
    )
    .run_commands(
        # Pre-download Depth Anything V2 Large (generic depth: humans + pets + objects)
        "python -c 'from transformers import pipeline; "
        "pipeline(\"depth-estimation\", model=\"depth-anything/Depth-Anything-V2-Large-hf\")'",
        # Pre-download Sapiens-Normal-1B (high-frequency face/hair geometry for portraits)
        "python -c 'from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id=\"facebook/sapiens-normal-1b-torchscript\", "
        "filename=\"sapiens_1b_normal_render_people_epoch_115_torchscript.pt2\")'",
    )
    # Bump BUILD_VERSION to invalidate the cached image hash AND any class memory
    # snapshots derived from it. Required whenever process() signature changes,
    # because Modal's enable_memory_snapshot=True can otherwise restore a stale
    # pickled class definition and reject new kwargs.
    .env({"BUILD_VERSION": "3"})
)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
NORMAL_MODEL_REPO = "facebook/sapiens-normal-1b-torchscript"
NORMAL_MODEL_FILE = "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"

app = modal.App("depth-to-glb-service", image=image)


# 2. Define the GPU Class
@app.cls(gpu="A10", timeout=600, scaledown_window=300, enable_memory_snapshot=True,
         secrets=[modal.Secret.from_name("photoroom-api-key")])
class Model:
    @modal.enter()
    def _load_models(self):
        import torch
        from transformers import pipeline
        from huggingface_hub import hf_hub_download

        try:
            print("🔄 Loading models to GPU...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {self.device}")

            print(f"Loading Depth Anything V2 Large from {DEPTH_MODEL_ID}...")
            self.depth_pipe = pipeline(
                "depth-estimation",
                model=DEPTH_MODEL_ID,
                device=0 if torch.cuda.is_available() else -1,
            )

            print(f"Loading Sapiens-Normal-1B from {NORMAL_MODEL_REPO}...")
            normal_path = hf_hub_download(repo_id=NORMAL_MODEL_REPO, filename=NORMAL_MODEL_FILE)
            self.normal_model = torch.jit.load(normal_path)
            self.normal_model = self.normal_model.to(self.device).eval()

            print("Loading OpenCV Haar face detector...")
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            if self.face_detector.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

            print("✅ Models loaded successfully.")
        except Exception as e:
            import traceback
            print(f"❌ Model loading failed: {e}")
            print(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _remove_background(self, pil_img):
        """Send to PhotoRoom; return RGBA PIL at original resolution."""
        import requests
        from PIL import Image

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        resp = requests.post(
            "https://sdk.photoroom.com/v1/segment",
            headers={"x-api-key": os.environ["PHOTOROOM_API_KEY"]},
            files={"image_file": ("image.png", buf, "image/png")},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"PhotoRoom API error {resp.status_code}: {resp.text}")
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")

    def _estimate_depth(self, pil_img):
        """Run Depth Anything V2 on a PIL RGB image. Returns float32 (H, W).

        Convention: larger = closer (DA V2 outputs inverse depth).
        """
        import numpy as np

        out = self.depth_pipe(pil_img)
        # HF pipeline returns {"depth": PIL.Image, "predicted_depth": tensor}
        if "predicted_depth" in out:
            arr = out["predicted_depth"].squeeze().detach().cpu().float().numpy()
        else:
            arr = np.array(out["depth"]).astype(np.float32)

        # Resize to input size if needed
        if arr.shape != (pil_img.size[1], pil_img.size[0]):
            import torch
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(
                t, size=(pil_img.size[1], pil_img.size[0]), mode="bilinear", align_corners=False
            )
            arr = t.squeeze().numpy().astype(np.float32)

        print(f"DA V2 raw depth range: {arr.min():.4f} to {arr.max():.4f}")
        return arr

    def _normalize_foreground_depth(self, depth, mask, mad_k=4.0):
        """Robust normalize via median ± k*MAD on foreground pixels.

        MAD outlier rejection drops DA V2's "dark-region → far" artifacts
        (pupils, nostrils, eyebrows) that would otherwise dominate the
        normalized range and squash subtle facial topology.
        """
        import numpy as np

        if mask is None or not mask.any():
            fg = depth.flatten()
        else:
            fg = depth[mask]
        med = float(np.median(fg))
        mad = float(np.median(np.abs(fg - med))) + 1e-8
        lo = med - mad_k * mad
        hi = med + mad_k * mad
        d_range = hi - lo if hi > lo else 1.0
        normed = ((depth - lo) / d_range).astype(np.float32)
        normed = np.clip(normed, 0.0, 1.0)
        if mask is not None:
            normed = np.where(mask, normed, 0.0).astype(np.float32)
        print(
            f"Foreground depth: median={med:.4f}, MAD={mad:.4f}, "
            f"clipped to [{lo:.4f}, {hi:.4f}] (k={mad_k})"
        )
        return normed

    def _enhance_detail(self, depth, mask, detail_boost, base_sigma):
        """Base/detail decomposition: low-pass = global shape, residual = local detail.

        Boosts the residual to make face features (nose, eyes, lips) pop without
        exaggerating the silhouette. Standard bas-relief sharpening technique.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter

        if detail_boost <= 1.001:
            return depth
        d = depth.astype(np.float32)
        base = gaussian_filter(d, sigma=base_sigma)
        detail = d - base
        enhanced = base + detail * float(detail_boost)
        if mask is not None and mask.any():
            fg = enhanced[mask]
            lo, hi = float(fg.min()), float(fg.max())
        else:
            lo, hi = float(enhanced.min()), float(enhanced.max())
        rng = hi - lo if hi > lo else 1.0
        enhanced = (enhanced - lo) / rng
        enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            enhanced = np.where(mask, enhanced, 0.0).astype(np.float32)
        print(f"Detail boost x{detail_boost}, base sigma={base_sigma}")
        return enhanced

    def _compress_bas_relief(self, depth, mask, tau):
        """Gradient-domain compression + Poisson reconstruction (DCT, Neumann BCs).

        See Weyrich et al. 2007. Compresses extreme depth jumps (extended limbs)
        while preserving subtle features (nose, eye sockets).

        tau: compression strength. Smaller = more aggressive compression.
             tau >= 0.999 → identity (skip).
        """
        import numpy as np
        from scipy.fft import dctn, idctn

        if tau >= 0.999:
            return depth.astype(np.float32)

        H, W = depth.shape
        d = np.where(mask, depth, 0.0).astype(np.float64) if mask is not None else depth.astype(np.float64)

        gx = np.diff(d, axis=1, prepend=d[:, :1])
        gy = np.diff(d, axis=0, prepend=d[:1, :])
        mag = np.hypot(gx, gy) + 1e-8
        factor = (np.log1p(mag / tau) * tau) / mag
        gx2 = gx * factor
        gy2 = gy * factor

        div = (
            np.diff(gx2, axis=1, append=gx2[:, -1:])
            + np.diff(gy2, axis=0, append=gy2[-1:, :])
        )

        f = dctn(div, type=2, norm="ortho")
        ii = np.arange(H)[:, None]
        jj = np.arange(W)[None, :]
        denom = 2 * np.cos(np.pi * ii / H) + 2 * np.cos(np.pi * jj / W) - 4
        denom[0, 0] = 1.0  # gauge freedom
        z = idctn(f / denom, type=2, norm="ortho")
        z[0, 0] = 0

        if mask is not None and mask.any():
            z_fg = z[mask]
            lo, hi = float(z_fg.min()), float(z_fg.max())
        else:
            lo, hi = float(z.min()), float(z.max())
        rng = hi - lo if hi > lo else 1.0
        z = (z - lo) / rng
        z = np.clip(z, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            z = np.where(mask, z, 0.0).astype(np.float32)
        return z

    # ------------------------------------------------------------------
    # Normal-fusion helpers (Sapiens-Normal + DA V2 high-freq blending)
    # ------------------------------------------------------------------
    def _solve_poisson_neumann(self, div):
        """Solve ∇²z = div with Neumann BCs via DCT-II (closed form, ~30ms).

        Same math as the unused _compress_bas_relief, factored out.
        """
        import numpy as np
        from scipy.fft import dctn, idctn

        H, W = div.shape
        f = dctn(div, type=2, norm="ortho")
        ii = np.arange(H)[:, None]
        jj = np.arange(W)[None, :]
        denom = 2 * np.cos(np.pi * ii / H) + 2 * np.cos(np.pi * jj / W) - 4
        denom[0, 0] = 1.0  # gauge freedom
        z = idctn(f / denom, type=2, norm="ortho")
        z[0, 0] = 0
        return z.astype(np.float32)

    def _detect_face(self, pil_img):
        """Returns list of (x, y, w, h) face bounding boxes in image pixel coords.

        Caller decides what to do based on len(boxes): 0 = no fusion, 1 = fuse,
        >1 = multi-face → skip fusion (Sapiens is single-subject, ringing risk).
        """
        import cv2
        import numpy as np
        try:
            arr = np.array(pil_img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            min_dim = max(40, int(min(arr.shape[:2]) * 0.05))
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(min_dim, min_dim),
            )
            faces = list(faces) if faces is not None else []
            print(f"Face detection: {len(faces)} face(s) found")
            return faces
        except Exception as e:
            print(f"WARNING: face detection failed: {e}; falling back to no-fusion.")
            return []

    def _estimate_normals(self, pil_img):
        """Run Sapiens-Normal-1B. Returns (H, W, 3) unit-normal float32 in image coords."""
        import torch
        import numpy as np
        from PIL import Image

        target_h, target_w = 1024, 768
        # Letterbox preserving aspect
        scale = min(target_w / pil_img.width, target_h / pil_img.height)
        new_w, new_h = max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale))
        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        pad_l = (target_w - new_w) // 2
        pad_t = (target_h - new_h) // 2
        padded = Image.new("RGB", (target_w, target_h), (127, 127, 127))
        padded.paste(resized, (pad_l, pad_t))

        img_np = np.array(padded).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_np - mean) / std
        t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.inference_mode():
            out = self.normal_model(t)
        out = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

        # Crop letterbox
        out = out[pad_t:pad_t + new_h, pad_l:pad_l + new_w]

        # Resize each channel back to original image size (preserve sign)
        ow, oh = pil_img.size
        out_rs = np.empty((oh, ow, 3), dtype=np.float32)
        for c in range(3):
            chan = out[..., c]
            # Map [-1,1] → [0,255], resize, back. Loses minor precision but acceptable.
            chan_u8 = np.clip((chan + 1) * 127.5, 0, 255).astype(np.uint8)
            chan_resized = np.array(Image.fromarray(chan_u8).resize((ow, oh), Image.BILINEAR))
            out_rs[..., c] = chan_resized.astype(np.float32) / 127.5 - 1.0

        # Re-unitize per-pixel
        n = np.linalg.norm(out_rs, axis=2, keepdims=True) + 1e-6
        out_rs = out_rs / n
        print(f"Normals: range nx[{out_rs[...,0].min():.2f},{out_rs[...,0].max():.2f}] "
              f"ny[{out_rs[...,1].min():.2f},{out_rs[...,1].max():.2f}] "
              f"nz[{out_rs[...,2].min():.2f},{out_rs[...,2].max():.2f}]")
        return out_rs

    def _fuse_depth_normal(self, depth, normal, face_bbox, sigma_blend_px,
                           bbox_pad_frac=0.5):
        """Fuse Sapiens normals with DA V2 ONLY within an expanded face bounding box.

        Restricting to face bbox avoids two problems:
        - Whole-image Poisson with mask creates ringing at silhouette (icicle artifacts).
        - Sapiens normals are nonsensical on non-face regions (hair, body, pets).

        face_bbox is (x, y, w, h) in image pixel coords. We pad by bbox_pad_frac
        on all sides, crop, run Poisson on the crop, blend, and feather back.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter

        H, W = depth.shape
        fx, fy, fw, fh = face_bbox
        # Expand bbox by pad fraction (cover hair/forehead/chin context)
        pad_x = int(fw * bbox_pad_frac)
        pad_y = int(fh * bbox_pad_frac)
        x0 = max(0, fx - pad_x)
        y0 = max(0, fy - pad_y)
        x1 = min(W, fx + fw + pad_x)
        y1 = min(H, fy + fh + pad_y)
        if x1 - x0 < 32 or y1 - y0 < 32:
            print("Fusion skipped: face crop too small.")
            return depth

        d_crop = depth[y0:y1, x0:x1].astype(np.float64)
        n_crop = normal[y0:y1, x0:x1]
        nx, ny, nz = n_crop[..., 0], n_crop[..., 1], n_crop[..., 2]
        nz_safe = np.where(np.abs(nz) < 0.05, np.sign(nz) * 0.05 + 1e-6, nz)
        # Height-field math: surface z=f(u,v), normal = (-fu, -fv, 1)/|N|
        # → fu = -nx/nz, fv = -ny/nz. Both signs negative.
        gx = -nx / nz_safe
        gy = -ny / nz_safe

        # Light gaussian smooth on gradients suppresses Poisson ringing
        gx = gaussian_filter(gx, sigma=1.5)
        gy = gaussian_filter(gy, sigma=1.5)

        div = (
            np.diff(gx, axis=1, append=gx[:, -1:])
            + np.diff(gy, axis=0, append=gy[-1:, :])
        )
        z_fine = self._solve_poisson_neumann(div)

        # Scale-match z_fine to depth crop (mean + std)
        z_mu, z_sd = float(z_fine.mean()), float(z_fine.std()) + 1e-8
        d_mu, d_sd = float(d_crop.mean()), float(d_crop.std()) + 1e-8
        z_aligned = (z_fine - z_mu) * (d_sd / z_sd) + d_mu

        # Frequency blend: low-freq from DA V2, high-freq from Sapiens
        sig = max(1.0, float(sigma_blend_px))
        d_lo = gaussian_filter(d_crop.astype(np.float32), sigma=sig)
        z_lo = gaussian_filter(z_aligned.astype(np.float32), sigma=sig)
        fused_crop = d_lo + (z_aligned - z_lo)

        # Feather the crop back into the full image with a smooth edge so
        # there's no visible seam between fused and DA V2 regions.
        out = depth.copy().astype(np.float32)
        ch, cw = fused_crop.shape
        feather = np.ones((ch, cw), dtype=np.float32)
        edge_px = max(8, min(ch, cw) // 8)
        # Linear ramp at all 4 edges
        for i in range(edge_px):
            w = (i + 1) / edge_px
            feather[i, :] *= w
            feather[ch - 1 - i, :] *= w
            feather[:, i] *= w
            feather[:, cw - 1 - i] *= w
        out[y0:y1, x0:x1] = (
            fused_crop * feather + d_crop.astype(np.float32) * (1.0 - feather)
        )

        # Renormalize whole-image to [0,1] so silhouette cut etc still works
        fg_min, fg_max = float(out.min()), float(out.max())
        rng = fg_max - fg_min if fg_max > fg_min else 1.0
        out = np.clip((out - fg_min) / rng, 0.0, 1.0).astype(np.float32)
        print(f"Fusion: bbox=({x0},{y0})-({x1},{y1}) ({cw}×{ch}), sigma_px={sig:.1f}, "
              f"d μ/σ=({d_mu:.3f},{d_sd:.3f}), z μ/σ=({z_mu:.3f},{z_sd:.3f})")
        return out

    def _extrapolate_to_background(self, depth, mask):
        """Fill background pixels with nearest foreground depth (avoids edge streaks)."""
        from scipy.ndimage import distance_transform_edt

        if mask is None or mask.all():
            return depth
        invalid = ~mask
        nearest = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        return depth[tuple(nearest)]

    def _smooth_silhouette_edge_depth(self, depth, mask, edge_px):
        """Feather depth toward smooth body-shape at the silhouette boundary.

        DA V2 depth is unreliable at object boundaries (wispy hair, fuzzy
        segmentation edges). Without this step, boundary pixels with outlier
        depths get triangulated into icicle-like z-spikes radiating from the
        silhouette in the final mesh.

        Within `edge_px` of the boundary, blends original depth toward a
        heavily-smoothed version (low-frequency body shape). At the boundary
        itself depth equals the smoothed value; deeper than `edge_px` inside
        the figure depth is unchanged so interior detail is preserved.
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        if mask is None or not mask.any() or edge_px <= 0:
            return depth
        inside_dist = distance_transform_edt(mask)
        smooth = gaussian_filter(depth.astype(np.float32), sigma=float(edge_px) * 1.5)
        t = np.clip(1.0 - inside_dist / float(edge_px), 0.0, 1.0).astype(np.float32)
        out = depth.astype(np.float32) * (1.0 - t) + smooth * t
        print(f"Silhouette depth feather: edge_px={edge_px}")
        return out.astype(np.float32)

    def _build_grid_mesh(self, depth, rgba_pil, aspect_ratio, depth_ratio, mesh_detail):
        """Build a textured grid mesh where each pixel sits at its true depth.

        For sub-surface crystal engraving: NO alpha fade, NO back plane. Each
        foreground pixel becomes a 3D vertex at (x, y, z=-depth*z_extent). The
        silhouette cut step (called after this) trims to the figure outline.
        """
        import numpy as np
        import torch
        import trimesh
        from PIL import Image

        height_full, width_full = depth.shape
        target_w = max(2, int(width_full * mesh_detail))
        target_h = max(2, int(height_full * mesh_detail))

        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
        depth_t = torch.nn.functional.interpolate(
            depth_t, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        depth_resized = depth_t.squeeze().numpy().astype(np.float32)
        depth_resized = np.clip(depth_resized, 0.0, 1.0)

        alpha_resized = np.array(rgba_pil.resize((target_w, target_h), Image.NEAREST))[:, :, 3]

        # Z extent calibrated to be a fixed fraction of XY extent
        x_extent = max(2 * aspect_ratio, 2.0)
        z_extent = depth_ratio * x_extent

        x_range = aspect_ratio
        x = np.linspace(-x_range, x_range, target_w)
        y = np.linspace(1, -1, target_h)
        xv, yv = np.meshgrid(x, y)

        u = np.linspace(0, 1, target_w)
        v = np.linspace(1, 0, target_h)
        uv_u, uv_v = np.meshgrid(u, v)

        # Each pixel sits at its true 3D position. No alpha fade.
        z = -depth_resized * z_extent
        vertices = np.column_stack((xv.flatten(), yv.flatten(), z.flatten()))
        uvs = np.column_stack((uv_u.flatten(), uv_v.flatten()))

        # Vectorized face generation
        rr, cc = np.meshgrid(np.arange(target_h - 1), np.arange(target_w - 1), indexing="ij")
        i = rr * target_w + cc
        t1 = np.stack([i, i + 1, i + target_w], axis=-1).reshape(-1, 3)
        t2 = np.stack([i + 1, i + target_w + 1, i + target_w], axis=-1).reshape(-1, 3)
        faces = np.concatenate([t1, t2], axis=0)

        visual = trimesh.visual.TextureVisuals(uv=uvs, image=rgba_pil)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

        print(
            f"Grid mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}, "
            f"X-extent={2*aspect_ratio:.3f}, Y-extent=2.000, Z-extent={z_extent:.3f}"
        )
        return mesh, alpha_resized, z_extent

    def _cut_to_silhouette(self, mesh, alpha_array, alpha_threshold, smooth_iters=2,
                           open_iters=2, erode_iters=1):
        """Cut the rectangular mesh down to the figure silhouette.

        Pipeline:
        1. binary_opening (erode → dilate) removes thin 1-2px features (wispy
           hair strands) that would otherwise mesh into icicle spikes.
        2. binary_closing (dilate → erode) fills small gaps inside the figure.
        3. binary_fill_holes plugs interior voids.
        4. binary_erosion (erode_iters) peels off the outermost ring of
           anti-aliased / depth-unreliable boundary pixels.

        Net: the main figure stays intact; only thin protruding features and
        the very outermost feathered boundary are removed.
        """
        from scipy.ndimage import (
            binary_closing, binary_erosion, binary_fill_holes, binary_opening,
        )

        mask_bool = alpha_array > alpha_threshold
        if open_iters > 0:
            mask_bool = binary_opening(mask_bool, iterations=int(open_iters))
        if smooth_iters > 0:
            mask_bool = binary_closing(mask_bool, iterations=int(smooth_iters))
        mask_bool = binary_fill_holes(mask_bool)
        if erode_iters > 0:
            mask_bool = binary_erosion(mask_bool, iterations=int(erode_iters))

        vertex_mask = mask_bool.flatten()
        try:
            safe_limit = min(len(mesh.vertices), len(vertex_mask))
            faces = mesh.faces
            valid = faces.max(axis=1) < safe_limit
            faces = faces[valid]
            face_mask = vertex_mask[faces].all(axis=1)
            mesh.faces = faces[face_mask]
            # Intentionally NOT calling mesh.remove_unreferenced_vertices() —
            # trimesh's TextureVisuals.uv array can fall out of sync after vertex
            # renumbering, causing IndexError in _extrude_boundary when it
            # indexes mesh.visual.uv[v_start_idx]. Leaving unreferenced vertices
            # in place adds ~5-10% file size but keeps UVs consistent with
            # mesh.vertices. Renderers ignore unreferenced vertices.
            print(
                f"Silhouette cut: kept {face_mask.sum()}/{len(face_mask)} faces, "
                f"smooth_iters={smooth_iters}"
            )
        except IndexError as e:
            print(f"WARNING: Silhouette cut failed with {e}. Skipping.")
        return mesh

    def _extrude_boundary(self, mesh, rgba_pil, z_extent):
        """Add a side ribbon connecting the boundary loop to a back plane."""
        import numpy as np
        import trimesh

        try:
            boundary_edges = mesh.edges_unique[
                trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
            ]
            if len(boundary_edges) == 0:
                return mesh

            v_start_idx = boundary_edges[:, 0]
            v_end_idx = boundary_edges[:, 1]
            # Sanity guard: stale boundary indices vs vertex/UV arrays. If they
            # don't agree (e.g. after vertex renumbering elsewhere), skip cleanly
            # rather than crash so the request still returns 200.
            max_vidx = int(max(v_start_idx.max(), v_end_idx.max()))
            if max_vidx >= len(mesh.vertices):
                print(f"WARNING: boundary index {max_vidx} >= len(mesh.vertices)={len(mesh.vertices)}; skipping extrusion.")
                return mesh
            v_start = mesh.vertices[v_start_idx]
            v_end = mesh.vertices[v_end_idx]

            min_z = float(mesh.vertices[:, 2].min())
            target_z = min_z - 0.05 * z_extent

            v_start_back = v_start.copy(); v_start_back[:, 2] = target_z
            v_end_back = v_end.copy(); v_end_back[:, 2] = target_z

            num_edges = len(boundary_edges)
            s_v = np.concatenate([v_start, v_end, v_end_back, v_start_back])
            f_A = np.arange(num_edges)
            f_B = np.arange(num_edges) + num_edges
            f_C = np.arange(num_edges) + 2 * num_edges
            f_D = np.arange(num_edges) + 3 * num_edges
            t1 = np.column_stack([f_A, f_B, f_C])
            t2 = np.column_stack([f_A, f_C, f_D])
            s_f = np.concatenate([t1, t2])

            try:
                if mesh.visual.uv is not None and len(mesh.visual.uv) > max(v_start_idx.max(), v_end_idx.max()):
                    uv_start = mesh.visual.uv[v_start_idx]
                    uv_end = mesh.visual.uv[v_end_idx]
                else:
                    uv_start = np.full((num_edges, 2), 0.5)
                    uv_end = np.full((num_edges, 2), 0.5)
            except (IndexError, AttributeError):
                uv_start = np.full((num_edges, 2), 0.5)
                uv_end = np.full((num_edges, 2), 0.5)
            s_uv = np.concatenate([uv_start, uv_end, uv_end, uv_start])

            side_mesh = trimesh.Trimesh(
                vertices=s_v, faces=s_f,
                visual=trimesh.visual.TextureVisuals(uv=s_uv, image=rgba_pil),
            )

            # Back plane: clone the front mesh's vertices/faces at target_z with
            # reversed winding. Closes the slab so viewers don't see the inside
            # of the front face from the back side.
            back_v = mesh.vertices.copy()
            back_v[:, 2] = target_z
            back_f = mesh.faces[:, ::-1]  # reverse winding → outward-facing normals
            back_uv = mesh.visual.uv.copy() if mesh.visual.uv is not None else None
            back_mesh = trimesh.Trimesh(
                vertices=back_v, faces=back_f,
                visual=trimesh.visual.TextureVisuals(uv=back_uv, image=rgba_pil),
                process=False,
            )

            return trimesh.util.concatenate([mesh, side_mesh, back_mesh])
        except (IndexError, ValueError) as e:
            print(f"WARNING: Extrusion failed with {e}. Skipping.")
            return mesh

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    @modal.method()
    def process(self, image_bytes,
                keep_background: bool = False,
                depth_ratio: float = 0.30,
                detail_boost: float = 1.0,
                detail_base_sigma: float = 12.0,
                bilateral_d: int = 9,
                bilateral_sigma_color: float = 0.05,
                bilateral_sigma_space: float = 20.0,
                use_normal_fusion: bool = True,
                normal_blend_sigma: float = 60.0,
                silhouette_smooth_iters: int = 2,
                silhouette_open_iters: int = 2,
                silhouette_erode_iters: int = 1,
                silhouette_edge_smooth_px: float = 12.0,
                mesh_detail: float = 0.5,
                smooth_iters: int = 0,
                alpha_threshold: int = 160,
                depth_blur: float = 0.0,
                extrude: bool = False):
        import time
        import numpy as np
        import trimesh
        import trimesh.transformations as transformations
        from PIL import Image
        from scipy.ndimage import gaussian_filter

        if not hasattr(self, "depth_pipe"):
            self._load_models()

        timings = {}
        t0 = time.time()
        print("Processing request...")
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(original.size) > 1080:
            print(f"Downsizing from {original.size}...")
            original.thumbnail((1080, 1080), Image.LANCZOS)
        width, height = original.size
        aspect_ratio = width / height

        # 1. Background handling
        t = time.time()
        if keep_background:
            print("keep_background=True: skipping PhotoRoom; using original RGB as texture.")
            rgba = original.convert("RGBA")
            alpha_mask = np.ones((height, width), dtype=bool)
        else:
            rgba = self._remove_background(original)
            if rgba.size != (width, height):
                rgba = rgba.resize((width, height), Image.LANCZOS)
            alpha_mask = np.array(rgba)[:, :, 3] > alpha_threshold
        timings["background"] = time.time() - t

        # 2. Depth estimation
        t = time.time()
        depth_raw = self._estimate_depth(original)
        timings["depth"] = time.time() - t

        # 3. Foreground-only normalization
        t = time.time()
        depth_norm = self._normalize_foreground_depth(depth_raw, alpha_mask)
        timings["normalize"] = time.time() - t

        # 4. EDT fill so background pixels get nearest-foreground depth (avoids
        # huge silhouette gradient that would otherwise dominate the mesh).
        t = time.time()
        depth_continuous = self._extrapolate_to_background(depth_norm, alpha_mask)
        timings["edt_fill"] = time.time() - t

        # 5. Bilateral filter: edge-preserving smoothing. Smooths within flat
        # regions (hair, cheek) but preserves sharp transitions (face contour,
        # nose ridge). Eliminates the stepped/banded hair artifacts that median
        # + gaussian produced.
        depth_processed = depth_continuous
        if bilateral_d >= 3:
            t = time.time()
            import cv2
            depth_processed = cv2.bilateralFilter(
                depth_processed.astype(np.float32),
                d=int(bilateral_d),
                sigmaColor=float(bilateral_sigma_color),
                sigmaSpace=float(bilateral_sigma_space),
            )
            timings["bilateral"] = time.time() - t

        # 5a. Normal fusion: bbox-restricted. Sapiens normals fuse Sapiens
        # high-frequency face geometry with DA V2's overall shape ONLY within
        # the face bounding box (with margin). Outside the bbox, DA V2 is used
        # unchanged. Skipped when 0 or >1 faces are detected (Sapiens is
        # single-subject; multi-face inputs use DA V2 alone).
        if use_normal_fusion:
            t = time.time()
            faces = self._detect_face(original)
            timings["face_detect"] = time.time() - t
            if len(faces) == 1:
                fx, fy, fw, fh = faces[0]
                # Map detected bbox (at original PIL size) to depth-array coords
                ow, oh = original.size
                dh, dw = depth_processed.shape
                sx, sy = dw / ow, dh / oh
                bbox_in_depth = (
                    int(fx * sx), int(fy * sy),
                    int(fw * sx), int(fh * sy),
                )
                t = time.time()
                normals = self._estimate_normals(original)
                # Resize normals to depth resolution if needed
                if normals.shape[:2] != depth_processed.shape:
                    import cv2
                    normals = cv2.resize(
                        normals, (dw, dh), interpolation=cv2.INTER_LINEAR
                    )
                    n_norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6
                    normals = normals / n_norm
                timings["normals"] = time.time() - t
                t = time.time()
                sig_px = float(normal_blend_sigma) * (max(dh, dw) / 1080.0)
                depth_processed = self._fuse_depth_normal(
                    depth_processed, normals, bbox_in_depth, sig_px,
                )
                timings["fusion"] = time.time() - t
            elif len(faces) > 1:
                print(f"Fusion skipped: {len(faces)} faces detected (Sapiens is single-subject).")
            else:
                print("Fusion skipped: no face detected (DA V2 only).")

        # 6. Detail enhancement: boost local features (nose, eyes, lips). For
        # crystal engraving we keep the natural depth gradient (no bas-relief
        # compression) and only amplify high-frequency residual.
        t = time.time()
        sigma_px = max(1.0, float(detail_base_sigma) * (max(depth_processed.shape) / 1080.0))
        depth_processed = self._enhance_detail(
            depth_processed, alpha_mask, float(detail_boost), sigma_px
        )
        timings["detail_boost"] = time.time() - t

        # 6.5. Silhouette-edge depth feather: flattens boundary z so wispy
        # hair / fuzzy alpha pixels can't form icicle spikes after the cut.
        if silhouette_edge_smooth_px > 0:
            t = time.time()
            edge_px = max(
                1, int(float(silhouette_edge_smooth_px) * (max(depth_processed.shape) / 1080.0))
            )
            depth_processed = self._smooth_silhouette_edge_depth(
                depth_processed, alpha_mask, edge_px
            )
            timings["edge_smooth"] = time.time() - t

        depth_filled = depth_processed

        # 6. Optional gaussian smoothing
        if depth_blur > 0:
            depth_filled = gaussian_filter(depth_filled, sigma=depth_blur)

        # 7. Build textured grid mesh — each foreground pixel at its true 3D position.
        t = time.time()
        mesh, alpha_resized, z_extent = self._build_grid_mesh(
            depth_filled, rgba, aspect_ratio, depth_ratio, mesh_detail,
        )
        timings["mesh_build"] = time.time() - t

        # 8. Cut rectangular mesh down to figure silhouette using closing+fill_holes
        # (no erosion, so no fragmentation). Boundary becomes the natural figure
        # outline, so the next extrusion step draws clean side walls instead of
        # a useless rectangle. Skipped when keep_background=True.
        if not keep_background:
            t = time.time()
            mesh = self._cut_to_silhouette(
                mesh, alpha_resized, alpha_threshold,
                smooth_iters=int(silhouette_smooth_iters),
                open_iters=int(silhouette_open_iters),
                erode_iters=int(silhouette_erode_iters),
            )
            timings["silhouette_cut"] = time.time() - t

        # 9. Optional Laplacian smoothing
        if smooth_iters > 0:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
            except IndexError as e:
                print(f"WARNING: Smoothing failed with {e}. Skipping.")

        # 10. Boundary extrusion: figure-silhouette side wall + back ribbon
        if extrude:
            t = time.time()
            mesh = self._extrude_boundary(mesh, rgba, z_extent)
            timings["extrude"] = time.time() - t

        # 11. Re-orient upright
        mesh.apply_transform(transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))

        # 12. Export
        t = time.time()
        out = io.BytesIO()
        mesh.export(out, file_type="glb")
        timings["export"] = time.time() - t
        timings["total"] = time.time() - t0
        print("Stage timings (s): " + ", ".join(f"{k}={v:.3f}" for k, v in timings.items()))
        return out.getvalue()


# 3. Define the Web Endpoint
import fastapi


@app.function(secrets=[modal.Secret.from_name("api-auth")])
@modal.fastapi_endpoint(method="POST")
def generate_3d(item: dict, authorization: str = fastapi.Header(None)):
    import base64
    import hmac

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise fastapi.HTTPException(status_code=500, detail="API_KEY not configured")

    if not authorization or not authorization.startswith("Bearer "):
        raise fastapi.HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    provided_key = authorization[7:]
    if not hmac.compare_digest(provided_key, api_key):
        raise fastapi.HTTPException(status_code=401, detail="Invalid API key")

    image_data = base64.b64decode(item["image"])

    try:
        glb_bytes = Model().process.remote(
            image_data,
            keep_background=bool(item.get("keep_background", False)),
            depth_ratio=float(item.get("depth_ratio", item.get("relief_depth_ratio", 0.30))),
            detail_boost=float(item.get("detail_boost", 1.0)),
            detail_base_sigma=float(item.get("detail_base_sigma", 12.0)),
            bilateral_d=int(item.get("bilateral_d", 9)),
            bilateral_sigma_color=float(item.get("bilateral_sigma_color", 0.05)),
            bilateral_sigma_space=float(item.get("bilateral_sigma_space", 20.0)),
            use_normal_fusion=bool(item.get("use_normal_fusion", True)),
            normal_blend_sigma=float(item.get("normal_blend_sigma", 60.0)),
            silhouette_smooth_iters=int(item.get("silhouette_smooth_iters", 2)),
            silhouette_open_iters=int(item.get("silhouette_open_iters", 2)),
            silhouette_erode_iters=int(item.get("silhouette_erode_iters", 1)),
            silhouette_edge_smooth_px=float(item.get("silhouette_edge_smooth_px", 12.0)),
            mesh_detail=float(item.get("detail", item.get("mesh_detail", 0.5))),
            smooth_iters=int(item.get("smooth", item.get("smooth_iters", 0))),
            alpha_threshold=int(item.get("alpha_threshold", 160)),
            depth_blur=float(item.get("depth_blur", 0.0)),
            extrude=bool(item.get("extrude", False)),
        )
    except Exception as e:
        import traceback
        error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise fastapi.HTTPException(status_code=500, detail=error_msg)

    return {"glb": base64.b64encode(glb_bytes).decode("utf-8")}
