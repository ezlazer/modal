# Build a depth-to-GLB service (image → 3D bas-relief mesh)

## Goal
Build a single-image service that converts a portrait/figure photo into a textured GLB mesh suitable for sub-surface crystal engraving (and other bas-relief use cases). The mesh must be a 3D depth-displaced surface — not a flat card with parallax — and the silhouette must match the figure outline, not the input image's rectangle.

You will replicate the behavior of an existing Modal-hosted service. The reference service is implemented in Python on Modal with an A10 GPU. You may target a different host, but the input/output contract and quality bar must match.

---

## HTTP contract

### Endpoint
`POST /generate_3d` (or equivalent)

### Auth
Bearer token (`Authorization: Bearer <API_KEY>`), constant-time compared.

### Request body (JSON)
```json
{
  "image": "<base64-encoded JPEG/PNG bytes>",
  "keep_background": false,
}
```

### Response (JSON)
```json
{ "glb": "<base64-encoded GLB binary>" }
```

### Errors
- `401` on missing or bad token.
- `500` with traceback string in `detail` on processing failure.

---

## Input image expectations
- JPEG or PNG, RGB or RGBA.
- Typical input: portrait of one or more people, sometimes with pets.

---

## Output GLB expectations
- Single textured mesh 
- Texture is the *foreground RGBA* (alpha encodes transparency outside the figure when `keep_background=false`).
- UV layout: `u = x_pixel / W`, `v = 1 − y_pixel / H` (top of image at v=1).
- Coordinate frame after final rotation:
  - X spans `[−aspect_ratio, +aspect_ratio]`.
  - Y spans `[−1, +1]` *before* the final upright rotation; *after* the `-90°` rotation around X (applied at the end), Y is up and Z is forward.
  - Z extent: `z_extent = depth_ratio * (2 * aspect_ratio)`. Foreground sits at `z = -depth * z_extent` (closer = more negative z) before the rotation.
- Silhouette outline: when `keep_background=false`, the rectangular grid is trimmed to the figure outline. No rectangle frame visible in the GLB.

---

## Processing pipeline (must reproduce in order)

1. **Decode** RGB load & Compute `aspect_ratio = W / H`.

2. **Background removal.**
   - When `keep_background=true`: skip; alpha = all-ones; RGBA texture = original RGB.
   - When `keep_background=false`: call PhotoRoom `https://sdk.photoroom.com/v1/segment` (header `x-api-key`). Resize PhotoRoom output back to working resolution if it differs. Alpha mask is `alpha > alpha_threshold`.

3. **Downsize** downsize so `max(W,H) ≤ 1080`.

4. **Monocular depth.** Use Depth Anything V3. Output is **inverse depth** (larger value = closer to camera). All downstream math assumes this convention; if you swap models, normalize to it.

5. **Foreground-only robust normalization.**
   - Take depth values where the alpha mask is true.
   - Background pixels (outside mask) get value `0` at this stage.

6. **Reorient.** Apply `rotation_matrix(radians(-90), [1,0,0])` so the mesh is upright (Y-up, looking down −Z).

7. **Export.** `mesh.export(file_type="glb")`. Return base64 in the JSON response.

Print stage timings to stdout for observability.

---

## Parameter reference

| Param | Default | Purpose | Notes |
|---|---|---|---|
| `keep_background` | `false` | Skip PhotoRoom and silhouette cut | Output is then a full rectangle |


---

## Quality bar — visual checks

A correct implementation should pass these on a typical group portrait:
1. **No silhouette spikes.** No thin triangles projecting in z from the figure outline. 
2. **Faces have crisp features.** eye sockets, nose ridge, lip line are visible in the displaced surface (not just the texture).
3. **Hair is smooth, not stepped/banded.** No staircase artifacts from naïve median+gaussian filtering.
4. **Silhouette outline matches figure** — no visible rectangular frame; no fragmented islands of mesh in the background.
5. **Texture aligns with geometry.** UVs correspond to the displaced grid, so the texture rendered onto the mesh matches the input photo with no shift.
6. **GLB loads in standard viewers** (Three.js, Blender, gltf-viewer, Chrome `model-viewer`) without warnings about non-manifold or missing UVs.

---

## Performance targets (A10 GPU, 1080-px input, defaults)
Cold-start can amortize via persistent model loading (Modal `enable_memory_snapshot=True` or equivalent on your host).

---

## Out of scope (for the first cut)
- Multi-image / multi-view depth.
- Animated GLB / morph targets.
- Per-vertex colors (texturing only).
- Production observability beyond stdout timings.
