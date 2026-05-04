"""Depth-Anything-V3 → bas-relief GLB service for crystal engraving.

Minimal implementation per AGENT_BRIEF.md. New Modal app, separate from the
V2-based depth-to-glb-service in server.py. Same HTTP contract:

    POST /generate_3d   Bearer <API_KEY>
    Request:  {"image": "<base64 JPEG/PNG>", "keep_background": false}
    Response: {"glb": "<base64 GLB>"}
"""

import io
import os

import modal


# DA3 wheel is not on PyPI — install from a pinned commit.
DA3_GIT = "git+https://github.com/ByteDance-Seed/depth-anything-3.git@41736238f5bced4debf3f2a12375d2466874866d"
DEPTH_MODEL_ID = "depth-anything/da3mono-large"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "fastapi[standard]",
        "torch>=2.1",
        "torchvision",
        "xformers",
        "transformers",
        "huggingface_hub",
        "requests",
        "trimesh",
        "Pillow",
        "numpy<2.0",
        "scipy",
    )
    .pip_install(DA3_GIT)
    .run_commands(
        "python -c 'from depth_anything_3.api import DepthAnything3; "
        f'DepthAnything3.from_pretrained("{DEPTH_MODEL_ID}")\'',
    )
    .env({"BUILD_VERSION": "21"})
)

app = modal.App("depth-to-glb-v3", image=image)


@app.cls(
    gpu="A10",
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("photoroom-api-key")],
)
class Model:
    @modal.enter()
    def _load_models(self):
        import torch
        from depth_anything_3.api import DepthAnything3

        print("Loading Depth Anything V3 (mono-large)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = DepthAnything3.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = self.depth_model.to(self.device).eval()
        print(f"Model loaded on {self.device}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _remove_background(self, pil_img):
        """PhotoRoom segmentation. Returns RGBA PIL at PhotoRoom's resolution."""
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

    def _smooth_alpha(self, rgba_pil, sigma_px):
        """Gaussian-blur PhotoRoom's alpha to smooth stair-step boundary noise.

        PhotoRoom's segmentation is accurate but has sub-pixel jaggedness
        along silhouettes (sawtooth pattern from the model's bilinear
        upsampling). Binary thresholding (alpha > 160) downstream preserves
        that jaggedness as mesh-boundary serrations. Pre-blurring the float
        alpha pushes the binary iso-contour to the smoother gaussian-soft
        level set — every downstream alpha read sees a smooth-bordered mask
        for free, without extra morphology iterations to tune.
        """
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter

        if sigma_px <= 0:
            return rgba_pil
        arr = np.array(rgba_pil)
        a = gaussian_filter(arr[:, :, 3].astype(np.float32), sigma=float(sigma_px))
        arr[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
        print(f"Alpha smooth: sigma_px={sigma_px}")
        return Image.fromarray(arr)

    def _crop_to_alpha_bbox(self, rgba_pil, alpha_threshold, padding_frac=0.02):
        """Crop RGBA to the foreground bounding box with a small padding band.

        Done BEFORE the working-resolution downsize so:
          - The 1080 cap applies to the figure crop, not the padded photo →
            more depth pixels per cm of subject.
          - The GLB texture carries no transparent margin → smaller file.
          - The mesh's coordinate frame (X spans [-aspect, +aspect]) reflects
            the figure aspect, so the figure fills its bounding crystal.

        Padding = max(8 px, padding_frac * max(bbox_w, bbox_h)) gives the edge
        feather room to operate without clipping the silhouette.
        """
        import numpy as np

        alpha = np.array(rgba_pil)[:, :, 3]
        mask = alpha > alpha_threshold
        if not mask.any():
            print("Alpha bbox: empty mask; skipping crop.")
            return rgba_pil
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        pad = max(8, int(padding_frac * max(bw, bh)))
        full_w, full_h = rgba_pil.size
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(full_w, x1 + pad)
        y1p = min(full_h, y1 + pad)
        cropped = rgba_pil.crop((x0p, y0p, x1p, y1p))
        print(f"Alpha bbox crop: {full_w}x{full_h} → {cropped.size[0]}x{cropped.size[1]} "
              f"(bbox=({x0},{y0})-({x1},{y1}), pad={pad}px)")
        return cropped

    def _estimate_depth(self, pil_img, process_res=1024):
        """Run DA3MONO-LARGE on a single PIL image. Returns float32 (H, W).

        DA3 outputs *relative depth where larger = farther*. The brief assumes
        the V2 convention (larger = closer, i.e. inverse depth), so we flip the
        sign here. Downstream foreground-robust normalization will rescale into
        [0, 1] with 1.0 = closest.
        """
        import numpy as np
        import torch

        with torch.inference_mode():
            pred = self.depth_model.inference(
                [pil_img],
                process_res=process_res,
                export_dir=None,
            )
        depth = np.asarray(pred.depth[0], dtype=np.float32)

        # Flip to "larger = closer" convention.
        depth = -depth

        # Resize to match input pixel grid (model may run at process_res internally).
        if depth.shape != (pil_img.size[1], pil_img.size[0]):
            t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(
                t,
                size=(pil_img.size[1], pil_img.size[0]),
                mode="bilinear",
                align_corners=False,
            )
            depth = t.squeeze().numpy().astype(np.float32)

        print(f"DA3 depth (sign-flipped) range: {depth.min():.4f} to {depth.max():.4f}")
        return depth

    def _normalize_foreground_depth(self, depth, mask, mad_k=4.0):
        """Robust normalize via median ± k·MAD on foreground pixels.

        MAD outlier rejection drops dark-region depth artifacts (pupils,
        nostrils) so they don't dominate the [0, 1] range and squash facial
        topology. Background pixels (outside mask) are pinned to 0.
        """
        import numpy as np

        fg = depth[mask] if (mask is not None and mask.any()) else depth.flatten()
        med = float(np.median(fg))
        mad = float(np.median(np.abs(fg - med))) + 1e-8
        lo = med - mad_k * mad
        hi = med + mad_k * mad
        d_range = hi - lo if hi > lo else 1.0
        normed = np.clip((depth - lo) / d_range, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            normed = np.where(mask, normed, 0.0).astype(np.float32)
        print(
            f"Foreground depth: median={med:.4f}, MAD={mad:.4f}, "
            f"clipped to [{lo:.4f}, {hi:.4f}] (k={mad_k})"
        )
        return normed

    def _extrapolate_to_background(self, depth, mask):
        """Fill background pixels with nearest foreground depth.

        Run before the silhouette-edge feather so the gaussian kernel at the
        boundary is not pulled toward 0 by background pixels (which would
        round/soften the silhouette outline rather than just denoising it).
        """
        from scipy.ndimage import distance_transform_edt

        if mask is None or mask.all():
            return depth
        invalid = ~mask
        nearest = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        return depth[tuple(nearest)]

    def _suppress_dark_region_depth(self, depth, rgb_pil, mask, lum_threshold):
        """Replace depth at dark FOREGROUND pixels with nearest reliable pixel's depth.

        DA3 (and every monocular depth model) has a "dark = far" bias —
        shadows under chins, dark cloth, cast shadows all get false-deep
        readings even though they're at the figure's surface. Image-guided
        smoothing can't fix this: the dark region IS a real luminance edge,
        so the guide tells the filter "preserve the discontinuity here."

        This helper sidesteps that: it identifies dark foreground pixels as
        UNRELIABLE, then overwrites their depth with the depth of the nearest
        reliable (bright + foreground) pixel via EDT. The V-neck dent gets
        replaced with surrounding chest depth; the chin shadow gets replaced
        with surrounding face depth. Bright pixels (face, polo body) are
        untouched, so face geometry is preserved.
        """
        import numpy as np
        from PIL import Image
        from scipy.ndimage import distance_transform_edt

        if lum_threshold <= 0 or mask is None or not mask.any():
            return depth

        L_pil = rgb_pil.convert("L")
        if L_pil.size != (depth.shape[1], depth.shape[0]):
            L_pil = L_pil.resize(
                (depth.shape[1], depth.shape[0]), Image.BILINEAR,
            )
        L = np.asarray(L_pil, dtype=np.float32) / 255.0

        dark = L < float(lum_threshold)
        suspicious = dark & mask
        if not suspicious.any():
            print(f"Dark-region suppress: 0 dark fg pixels (lum<{lum_threshold})")
            return depth

        reliable = mask & ~dark
        if not reliable.any():
            print("Dark-region suppress: no reliable pixels; skipping")
            return depth

        invalid = ~reliable
        nearest = distance_transform_edt(
            invalid, return_distances=False, return_indices=True,
        )
        out = np.where(suspicious, depth[tuple(nearest)], depth)
        n = int(suspicious.sum())
        print(f"Dark-region suppress: {n} dark fg pixels replaced (lum<{lum_threshold})")
        return out.astype(np.float32)

    def _image_guided_depth_smooth(self, depth, rgb_pil, radius_px, eps):
        """Edge-preserving depth smoothing guided by the RGB image (gray).

        Targets monocular depth's "dark = far" artifacts: shadows, polo
        V-necks, chin/collar recesses get false-deep readings from DA3 because
        the network uses image luminance as a depth cue. Smoothing depth
        within image-uniform regions (gray luminance as guide) averages those
        spurious deep readings against neighboring skin/cloth. True geometric
        edges (jaw, nose ridge) have RGB discontinuities and are preserved.

        Pure numpy + scipy implementation of He et al. 2010 guided filter
        (gray formulation), so we don't depend on cv2.ximgproc — opencv-contrib
        and DA3's transitive opencv-python collide on cv2 module ownership and
        the contrib bindings get shadowed at import time.
        """
        import numpy as np
        from PIL import Image
        from scipy.ndimage import uniform_filter

        if radius_px <= 0:
            return depth

        # Build gray guide at depth resolution (BT.601 luminance, normalized).
        guide_pil = rgb_pil.convert("L")
        if guide_pil.size != (depth.shape[1], depth.shape[0]):
            guide_pil = guide_pil.resize(
                (depth.shape[1], depth.shape[0]), Image.BILINEAR,
            )
        I = np.asarray(guide_pil, dtype=np.float32) / 255.0
        p = depth.astype(np.float32)
        r = int(radius_px)
        win = 2 * r + 1

        def box(x):
            return uniform_filter(x, size=win, mode="reflect")

        mean_I = box(I)
        mean_p = box(p)
        corr_Ip = box(I * p)
        corr_II = box(I * I)
        var_I = corr_II - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + float(eps))
        b = mean_p - a * mean_I
        out = box(a) * I + box(b)
        print(f"Image-guided depth smooth: radius={radius_px}px, eps={eps}")
        return out.astype(np.float32)

    def _smooth_silhouette_edge_depth(self, depth, mask, edge_px):
        """Feather depth toward smooth body-shape at the silhouette boundary.

        DA3 depth is unreliable at object boundaries (model output is
        bilinearly upsampled from a smaller process_res, so boundary pixels
        leak depth from across the edge). Without this step, those pixels'
        outlier depths get triangulated into icicle-like z-spikes radiating
        from the silhouette in the final mesh.

        Within `edge_px` of the boundary, blends original depth toward a
        heavily-smoothed version (low-frequency body shape). At the boundary
        depth equals the smoothed value; deeper than `edge_px` inside the
        figure, depth is unchanged so interior detail is preserved.
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

    def _build_grid_mesh(self, depth, rgba_texture, alpha_work, aspect_ratio, depth_ratio):
        """Build a textured grid mesh — one vertex per depth pixel.

        Texture and mesh resolutions are decoupled: `rgba_texture` is the
        full-resolution PIL image embedded in the GLB, while the mesh runs at
        `depth.shape` (working ≤1080-px resolution). UVs are normalized in
        [0, 1] so any texture size works with the same vertex grid.

        UV layout matches the brief: u = x_pixel / W, v = 1 − y_pixel / H.
        Coordinate frame *before* the final upright rotation:
          X ∈ [−aspect_ratio, +aspect_ratio]
          Y ∈ [−1, +1]   (top of image at y = +1)
          Z = −depth · z_extent     (closer = more negative z)
        """
        import numpy as np
        import trimesh

        height, width = depth.shape
        # alpha_work is already at depth resolution (caller's responsibility).
        alpha = alpha_work

        x_extent = max(2 * aspect_ratio, 2.0)
        z_extent = depth_ratio * x_extent

        x = np.linspace(-aspect_ratio, aspect_ratio, width)
        y = np.linspace(1.0, -1.0, height)
        xv, yv = np.meshgrid(x, y)

        u = np.linspace(0.0, 1.0, width)
        v = np.linspace(1.0, 0.0, height)
        uv_u, uv_v = np.meshgrid(u, v)

        z = np.clip(depth, 0.0, 1.0) * z_extent
        vertices = np.column_stack((xv.flatten(), yv.flatten(), z.flatten()))
        uvs = np.column_stack((uv_u.flatten(), uv_v.flatten()))

        rr, cc = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing="ij")
        i = rr * width + cc
        t1 = np.stack([i, i + 1, i + width], axis=-1).reshape(-1, 3)
        t2 = np.stack([i + 1, i + width + 1, i + width], axis=-1).reshape(-1, 3)
        faces = np.concatenate([t1, t2], axis=0)

        visual = trimesh.visual.TextureVisuals(uv=uvs, image=rgba_texture)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

        tex_w, tex_h = rgba_texture.size
        print(
            f"Grid mesh: {len(vertices)} verts, {len(faces)} faces, "
            f"texture={tex_w}x{tex_h}, "
            f"X-extent={2*aspect_ratio:.3f}, Y-extent=2.000, Z-extent={z_extent:.3f}"
        )
        return mesh, alpha

    def _cut_to_silhouette(self, mesh, alpha, alpha_threshold,
                           open_iters=2, erode_iters=2):
        """Trim the rectangular mesh to the figure outline.

        Boundary defense (in mask domain — depth is left untouched):
          1. binary_opening removes 1-2 px protrusions (wispy hair) that
             would otherwise mesh into icicle spikes at the silhouette.
          2. binary_closing + fill_holes plug interior alpha gaps.
          3. binary_erosion peels off the outermost anti-aliased ring where
             PhotoRoom's alpha is fractional and DA3's depth is unreliable
             (model output is bilinearly upsampled from a smaller process_res
             so boundary pixels have leaked depth from across the edge).
        """
        from scipy.ndimage import (
            binary_closing, binary_erosion, binary_fill_holes, binary_opening,
        )

        mask_bool = alpha > alpha_threshold
        if open_iters > 0:
            mask_bool = binary_opening(mask_bool, iterations=int(open_iters))
        mask_bool = binary_closing(mask_bool, iterations=1)
        mask_bool = binary_fill_holes(mask_bool)
        if erode_iters > 0:
            mask_bool = binary_erosion(mask_bool, iterations=int(erode_iters))

        vertex_mask = mask_bool.flatten()
        faces = mesh.faces
        valid = faces.max(axis=1) < min(len(mesh.vertices), len(vertex_mask))
        faces = faces[valid]
        face_mask = vertex_mask[faces].all(axis=1)
        mesh.faces = faces[face_mask]
        # Intentionally keep unreferenced vertices so trimesh's UV array stays
        # in sync with mesh.vertices indices. Renderers ignore them.
        print(f"Silhouette cut: kept {face_mask.sum()}/{len(face_mask)} faces")
        return mesh

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    @modal.method()
    def process(
        self,
        image_bytes,
        keep_background: bool = False,
        depth_ratio: float = 0.30,
        alpha_threshold: int = 128,
        silhouette_open_iters: int = 0,
        silhouette_erode_iters: int = 0,
        silhouette_edge_smooth_px: float = 12.0,
        alpha_smooth_sigma_px: float = 4.0,
        depth_smooth_radius_px: float = 8.0,
        depth_smooth_eps: float = 0.01,
        dark_lum_threshold: float = 0.30,
    ):
        import time

        import numpy as np
        import trimesh.transformations as transformations
        from PIL import Image

        if not hasattr(self, "depth_model"):
            self._load_models()

        timings = {}
        t0 = time.time()

        # 1. Decode at full resolution.
        original_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        full_w, full_h = original_full.size

        # 2. Background. PhotoRoom is run at FULL resolution so the GLB texture
        # carries crisp detail.
        t = time.time()
        if keep_background:
            rgba_full = original_full.convert("RGBA")
        else:
            rgba_full = self._remove_background(original_full)
            if rgba_full.size != (full_w, full_h):
                rgba_full = rgba_full.resize((full_w, full_h), Image.LANCZOS)
        timings["background"] = time.time() - t

        # 2a. Pre-smooth alpha to round PhotoRoom's stair-step boundary noise
        # before any binary thresholding (bbox crop, working mask, silhouette cut).
        if not keep_background and alpha_smooth_sigma_px > 0:
            t = time.time()
            rgba_full = self._smooth_alpha(rgba_full, alpha_smooth_sigma_px)
            timings["alpha_smooth"] = time.time() - t

        # 2b. Crop to alpha bounding box BEFORE the working-resolution downsize.
        # Skipped when keep_background=True (no foreground to crop to). After
        # this step, aspect_ratio reflects the figure crop, not the original photo.
        if not keep_background:
            t = time.time()
            rgba_full = self._crop_to_alpha_bbox(rgba_full, alpha_threshold)
            timings["alpha_crop"] = time.time() - t
        tex_w, tex_h = rgba_full.size
        aspect_ratio = tex_w / tex_h

        # 2c. Working-resolution copy for depth + mesh (cost-bounded ≤1080 px),
        # built from the (cropped) RGBA so the depth model gets more figure
        # pixels per cm. Texture stays at the cropped full resolution.
        rgba_work_pil = rgba_full.copy()
        if max(rgba_work_pil.size) > 1080:
            rgba_work_pil.thumbnail((1080, 1080), Image.LANCZOS)
        work_w, work_h = rgba_work_pil.size
        original_work = rgba_work_pil.convert("RGB")
        if keep_background:
            alpha_mask = np.ones((work_h, work_w), dtype=bool)
        else:
            alpha_mask = np.array(rgba_work_pil)[:, :, 3] > alpha_threshold
        print(f"Texture size: {tex_w}x{tex_h}, working size: {work_w}x{work_h}, "
              f"aspect={aspect_ratio:.3f}")

        # 3. DA3 monocular depth at working resolution (sign-flipped to "larger=closer").
        t = time.time()
        depth_raw = self._estimate_depth(original_work)
        timings["depth"] = time.time() - t

        # 4. Foreground-only robust normalization → [0, 1] with bg = 0.
        t = time.time()
        depth = self._normalize_foreground_depth(depth_raw, alpha_mask)
        timings["normalize"] = time.time() - t

        # 4b. EDT-fill background with nearest foreground depth, so the upcoming
        # gaussian smoothing at the silhouette is not pulled toward 0 (which
        # would round/soften the outline rather than just denoising it).
        if not keep_background:
            t = time.time()
            depth = self._extrapolate_to_background(depth, alpha_mask)
            timings["edt_fill"] = time.time() - t

        # 4b1. Dark-region depth suppress — overwrite depth at dark FG pixels
        # with nearest reliable (bright FG) pixel's depth. Direct fix for
        # monocular depth's "dark = far" bias (V-neck eating, chin shadows).
        # Must run BEFORE the guided smooth so the smooth doesn't lock in the
        # bad values.
        if not keep_background and dark_lum_threshold > 0:
            t = time.time()
            depth = self._suppress_dark_region_depth(
                depth, original_work, alpha_mask, lum_threshold=dark_lum_threshold,
            )
            timings["dark_suppress"] = time.time() - t

        # 4b2. Image-guided depth smoothing — additional edge-preserving cleanup
        # within image-uniform regions; useful even after dark-region suppress.
        if depth_smooth_radius_px > 0:
            t = time.time()
            radius = max(1, int(depth_smooth_radius_px *
                                (max(depth.shape) / 1080.0)))
            depth = self._image_guided_depth_smooth(
                depth, original_work, radius_px=radius, eps=depth_smooth_eps,
            )
            timings["depth_smooth"] = time.time() - t

        # 4c. Silhouette-edge depth feather — anti-icicle, depth-domain.
        # Within edge_px of the boundary, blend toward heavy-gaussian smoothed
        # depth so noisy boundary pixels can't form sawtooth z spikes.
        if silhouette_edge_smooth_px > 0 and not keep_background:
            t = time.time()
            edge_px = max(1, int(float(silhouette_edge_smooth_px) *
                                 (max(depth.shape) / 1080.0)))
            depth = self._smooth_silhouette_edge_depth(depth, alpha_mask, edge_px)
            timings["edge_smooth"] = time.time() - t

        # 5. Build textured grid mesh — full-res rgba_full as texture,
        # working-res alpha_mask (uint8 0/255) as the silhouette source.
        t = time.time()
        alpha_work_arr = (alpha_mask.astype(np.uint8) * 255)
        mesh, alpha_arr = self._build_grid_mesh(
            depth, rgba_full, alpha_work_arr, aspect_ratio, depth_ratio,
        )
        timings["mesh_build"] = time.time() - t

        # 6. Silhouette cut (skipped when keep_background=True).
        if not keep_background:
            t = time.time()
            mesh = self._cut_to_silhouette(
                mesh, alpha_arr, alpha_threshold,
                open_iters=silhouette_open_iters,
                erode_iters=silhouette_erode_iters,
            )
            timings["silhouette_cut"] = time.time() - t

        # 7. Reorient: -90° about X with z-flipped depth (foreground at +Z
        # pre-rotation). After rotation: head at -Z (top of screen), face at
        # +Y (toward camera at +Y in user's editor), right at +X.
        mesh.apply_transform(transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))

        # 8. Export.
        t = time.time()
        out = io.BytesIO()
        mesh.export(out, file_type="glb")
        timings["export"] = time.time() - t
        timings["total"] = time.time() - t0
        print("Stage timings (s): " + ", ".join(f"{k}={v:.3f}" for k, v in timings.items()))
        return out.getvalue()


# ---------------------------------------------------------------------- web
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
        raise fastapi.HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    if not hmac.compare_digest(authorization[7:], api_key):
        raise fastapi.HTTPException(status_code=401, detail="Invalid API key")

    image_data = base64.b64decode(item["image"])

    try:
        glb_bytes = Model().process.remote(
            image_data,
            keep_background=bool(item.get("keep_background", False)),
            depth_ratio=float(item.get("depth_ratio", 0.30)),
            alpha_threshold=int(item.get("alpha_threshold", 128)),
            silhouette_open_iters=int(item.get("silhouette_open_iters", 0)),
            silhouette_erode_iters=int(item.get("silhouette_erode_iters", 0)),
            silhouette_edge_smooth_px=float(item.get("silhouette_edge_smooth_px", 12.0)),
            alpha_smooth_sigma_px=float(item.get("alpha_smooth_sigma_px", 4.0)),
            depth_smooth_radius_px=float(item.get("depth_smooth_radius_px", 8.0)),
            depth_smooth_eps=float(item.get("depth_smooth_eps", 0.01)),
            dark_lum_threshold=float(item.get("dark_lum_threshold", 0.30)),
        )
    except Exception as e:
        import traceback

        error_msg = f"Processing error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        raise fastapi.HTTPException(status_code=500, detail=error_msg)

    return {"glb": base64.b64encode(glb_bytes).decode("utf-8")}
