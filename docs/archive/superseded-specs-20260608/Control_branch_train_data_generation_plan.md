Use Kubric, but do **not** make the first anchor dataset Kubric-only unless you are deliberately doing a narrow “physics-only smoke run.”

The right Phase 1 anchor plan is:

```text
Kubric anchors:
  true paired counterfactual physics
  true Q2 affected-region supervision
  best for collision/support/trajectory changes

Self-insertion anchors:
  real-video background preservation
  add/remove operation balance
  anti-collapse supervision
  cycle-compatible object side packets
  fast trainer/debug data
```

Self-insertion is not “only adding.” Each inserted composite produces **two supervised records**:

```text
ADD:
  src    = clean background video b
  target = composite video x = Compose(b, object/effects)
  op     = add

REMOVE:
  src    = composite video x
  target = clean background video b
  op     = remove
```

So the operation distribution is naturally 1:1.

The important correction is that self-insertion must emit **primary mask Mo, affected mask Ma, final quadmask Q, binary generation mask G, and object side packet**, not just “a mask.” E2W’s runtime contract requires four-valued `quadmask.npy` with `{0,63,127,255}`, while the binary generation mask is only the known/generate gate; semantic edit meaning stays in the quadmask. 

---

# 1. Immediate recommendation

Do this in parallel:

```text
Track A — Kubric/VOID wrapper
  Generate 16–32 Kubric paired samples immediately.
  Convert VOID/Kubric output into E2W v0.4 anchor manifests.
  This is your true physics/Q2 anchor source.

Track B — self-insertion builder
  Generate 8–16 real-background insertion composites.
  Emit both add and remove records per composite.
  This is your real-background preservation + add/remove balance source.
```

For the **first overfit run**, use:

```text
8 Kubric pairs       -> 16 records: add + remove
8 self-insertion     -> 16 records: add + remove
-----------------------------------------------
total                -> 32 training records
```

Then scale to:

```text
Stage 1A:
  32 records, overfit only

Stage 1B:
  100–300 records, still heavily audited

Stage 1C:
  500–2k records, mostly Kubric + some self-insertion replay
```

This respects the current Phase 1 goal: prove the trainable control branch learns Q0/Q1/Q2/Q3 and operation control, without claiming final cyclic training or visual success. Your v0.4 handoff already says the branch has the split frozen legacy path plus trainable control branch and audit layer, but lacks real paired data, a real trainer, and overfit evidence. 

---

# 2. Why not just Kubric?

For Phase 1, **Kubric should be the main physics anchor**. VOID does exactly this: it creates paired counterfactual videos by simulating a scene with the target object, removing the object, keeping the other initial conditions fixed, and re-simulating; its paper reports about 1,900 Kubric pairs and 4,500 HUMOTO pairs.  Kubric itself is designed for synthetic multi-object video generation with rich annotations such as instance segmentation, depth, and optical flow. ([GitHub][1])

But Kubric-only has three problems for E2W:

First, it is sim-domain. It can prove Q2 physics supervision, but it does not prove that Wan-VACE preserves real video backgrounds under E2W’s future in-the-wild regime.

Second, the released VOID data pipeline is primarily object-removal oriented. You can reverse pairs to create add records, but E2W’s eventual cyclic add/remove training needs explicit object side packets and add-direction schema, which self-insertion gives you naturally.

Third, VOID’s own ablation says a single data domain is weaker than mixed data: Kubric-only and HUMOTO-only at 1,200 samples underperformed a mixed 1,200-sample setting, and the detailed quadmask outperformed a weaker trimask-style mask. 

So the practical answer is:

```text
Use Kubric first.
Do not use Kubric only.
Use self-insertion as the real-background, add/remove-balanced, anti-collapse anchor stream.
```

---

# 3. Target anchor-data layout

Create one dataset root:

```bash
/data/cwx/E2W/data/phase1_v04_anchors_YYYYMMDD/
├── raw/
│   ├── kubric_void/
│   └── self_insert/
├── e2w/
│   ├── videos/
│   ├── masks/
│   ├── side_packets/
│   ├── previews/
│   └── manifests/
│       ├── train_all.jsonl
│       ├── overfit_32.jsonl
│       ├── eval_16.jsonl
│       └── rejected.jsonl
└── reports/
    ├── audit_summary.json
    ├── region_stats.csv
    └── sample_contact_sheet.mp4
```

Each final E2W sample directory should look like:

```text
e2w/videos/si_000001_remove/
├── src_video.mp4
├── target_video.mp4
├── source_frames.npy              # preferred authoritative training frames
├── target_frames.npy              # preferred authoritative training frames
├── preview_src_target.mp4

e2w/masks/si_000001_remove/
├── primary_mask.npy               # Mo, bool or uint8 [T,H,W]
├── affected_mask.npy              # Ma, bool or uint8 [T,H,W]
├── quadmask.npy                   # uint8 [T,H,W], exact values {0,63,127,255}
├── generation_mask.npy            # uint8 [T,H,W], values {0,255}
├── quadmask_preview.mp4           # preview only, not authoritative

e2w/side_packets/si_000001/
├── object_side_packet.npz
├── object_side_packet.json

e2w/manifests/train_all.jsonl
```

Do not make `quadmask_preview.mp4` authoritative. MP4 compression can perturb exact values like `63` and `127`. The training/audit source should be `quadmask.npy`.

---

# 4. Quadmask construction: exact formulas

For every sample, compute two binary masks first:

```text
Mo[t,h,w] = primary object / insertion-region mask
Ma[t,h,w] = affected non-target / induced-effect mask
```

Then compute the final E2W quadmask:

```python
Q = 255 everywhere
Q[Mo & ~Ma] = 0      # Q0: primary target only
Q[Mo &  Ma] = 63     # Q1: primary + affected overlap
Q[~Mo & Ma] = 127    # Q2: affected non-target only
Q[~Mo & ~Ma] = 255   # Q3: keep
```

The generation mask is derived separately:

```python
G = np.where(Q != 255, 255, 0).astype(np.uint8)
```

This follows E2W’s contract: the binary generation mask gates editable pixels, but semantic meaning remains in `quadmask.npy`. 

Use this function as the canonical implementation:

```python
import numpy as np

ALLOWED_Q = {0, 63, 127, 255}

def build_quadmask(primary: np.ndarray, affected: np.ndarray) -> np.ndarray:
    """
    primary: bool [T,H,W], primary object/insertion region
    affected: bool [T,H,W], non-target affected/effects/counterfactual-path region
    returns: uint8 [T,H,W] with values {0,63,127,255}
    """
    if primary.shape != affected.shape:
        raise ValueError(f"shape mismatch: primary={primary.shape}, affected={affected.shape}")

    primary = primary.astype(bool)
    affected = affected.astype(bool)

    q = np.full(primary.shape, 255, dtype=np.uint8)
    q[affected & ~primary] = 127
    q[primary & affected] = 63
    q[primary & ~affected] = 0

    bad = set(np.unique(q).tolist()) - ALLOWED_Q
    if bad:
        raise ValueError(f"invalid quadmask values: {bad}")
    return q


def build_generation_mask(quadmask: np.ndarray) -> np.ndarray:
    return np.where(quadmask != 255, 255, 0).astype(np.uint8)
```

---

# 5. Self-insertion anchor types

Do not build only one self-insertion type. Build three.

## Type SI-A: static inserted object + effects

Purpose:

```text
teaches:
  Q0 object add/remove
  Q2 shadow/contact/reflection effect
  Q3 real-background preservation

weak on:
  true downstream physics
```

Example:

```text
background b:
  clean tabletop video

inserted object:
  red ball / mug / cube / toy

effects:
  contact shadow
  soft cast shadow
  edge spill / antialias collar
```

Masks:

```text
Mo = inserted object visible alpha mask

Ma = shadow mask
   ∪ contact-patch mask
   ∪ reflection mask if used
   ∪ affected edge/effect collar
```

Q1 can be created from true overlap if the effect overlaps the object silhouette. If not, create a small controlled boundary-overlap proxy:

```python
Q1_proxy = primary & boundary_band(primary, width=2_to_4_px)
affected = affected | Q1_proxy
```

Record this explicitly in metadata:

```json
"q1_source": "boundary_proxy_for_control_bootstrap"
```

Do not pretend this is physical interaction. It is a control-bootstrap label that makes Q1 nonempty for the audit and forces the branch to learn the fourth value.

## Type SI-B: moving inserted object + moving effects

Purpose:

```text
teaches temporal tubes:
  Q0 moves through time
  Q2 shadow/contact follows through time
  Q3 remains stable
```

Example:

```text
a small ball rolls across a clean floor/table video
```

Masks:

```text
Mo[t] = moving ball alpha at frame t
Ma[t] = moving shadow/contact/effect mask at frame t
```

This is useful because it prevents the control branch from learning only static first-frame placement.

## Type SI-C: synthetic blocker + affected secondary object

Purpose:

```text
teaches true Q1/Q2 semantics without full Kubric
```

Construct two synthetic overlay trajectories on a real background:

```text
primary object P:
  inserted blocker / obstacle

affected object B:
  small ball or cube whose path differs depending on P
```

Two worlds:

```text
without primary:
  B travels straight through region where P would be

with primary:
  B bounces, stops, or deflects
```

For a remove record:

```text
src    = background + primary + B_bounced
target = background + B_straight
op     = remove
```

For an add record:

```text
src    = background + B_straight
target = background + primary + B_bounced
op     = add
```

Masks:

```text
Mo = primary object mask

Ma = union(B_straight_mask, B_bounced_mask)
   ∪ shadows/effects of B
   ∪ contact/collision patch

Q1 = Mo ∩ Ma
Q2 = Ma \ Mo
```

This gives real semantic Q1: the affected object’s counterfactual path passes through the primary object region.

Use SI-C sparingly in the first set, but include at least 2–4 examples so Q1 is not just boundary proxy.

---

# 6. Self-insertion generation algorithm

Implement one script:

```text
tools/generate_self_insertion_anchors.py
```

Inputs:

```bash
--background-root      directory of clean videos
--object-bank          directory of RGBA object tubes or renderable object specs
--out                  output root
--num-composites       number of composites
--frames               81
--height               480
--width                832
--fps                  12
--seed                 integer
--types                static,moving,blocker
--save-side-packet
--save-npy-frames
```

Example command:

```bash
cd /home/cwx/E2W/.worktree/feat/phase1-v04-control-branch

ANCHOR_ROOT=/data/cwx/E2W/data/phase1_v04_anchors_$(date -u +%Y%m%dT%H%M%SZ)

python tools/generate_self_insertion_anchors.py \
  --background-root /data/cwx/E2W/data/anchor_backgrounds_clean \
  --object-bank /data/cwx/E2W/data/object_tubes_rgba \
  --out "$ANCHOR_ROOT/raw/self_insert" \
  --num-composites 8 \
  --frames 81 \
  --height 480 \
  --width 832 \
  --fps 12 \
  --types static,moving,blocker \
  --seed 13 \
  --save-side-packet \
  --save-npy-frames
```

For each composite, the script should:

```text
1. Load clean background video b.
2. Normalize to T=81, H=480, W=832.
3. Load or render object tube o with RGBA.
4. Sample insertion transform:
   - position
   - scale
   - perspective skew if needed
   - trajectory over time
   - depth layer / z-order
5. Render effects:
   - contact shadow
   - cast shadow
   - optional reflection
   - optional edge color spill / blur
6. Composite x = Compose(b, o, effects).
7. Build Mo.
8. Build Ma.
9. Build Q = quadmask(Mo, Ma).
10. Build G = generation_mask(Q).
11. Emit remove record: src=x, target=b, op=remove.
12. Emit add record: src=b, target=x, op=add.
13. Emit object_side_packet for both directions.
14. Run audit.
```

---

# 7. Self-insertion compositing details

Use this simple compositing model first:

```python
def alpha_composite(bg_rgb, obj_rgb, alpha):
    # bg_rgb, obj_rgb: float [T,H,W,3] in [0,1]
    # alpha: float [T,H,W,1] in [0,1]
    return obj_rgb * alpha + bg_rgb * (1.0 - alpha)


def apply_shadow(rgb, shadow_alpha, strength=0.35):
    # shadow_alpha: float [T,H,W,1]
    return rgb * (1.0 - strength * shadow_alpha)
```

For each frame:

```text
P_t = alpha_t > 0.5

S_t = shadow_alpha_t > 0.05
C_t = contact_shadow_alpha_t > 0.05
R_t = reflection_alpha_t > 0.05, if used
E_t = edge_collar(P_t), if used

Ma_raw_t = S_t ∪ C_t ∪ R_t ∪ E_t
```

Then gridify only the affected/effect region, not the primary object:

```python
Ma_t = gridify(dilate(Ma_raw_t, radius=4), cell_size=32)
```

VOID explicitly uses focused affected regions and gridifies them to better match inference-time mask generation; do the same for Q2 while keeping Q0 object-accurate. 

Recommended first thresholds:

```text
primary alpha threshold:      0.50
shadow alpha threshold:       0.05
minimum Q0 area:              0.2% frame area
maximum Q0 area:              25% frame area
minimum Q2 area:              0.1% frame area for effect samples
minimum Q3 area:              50% frame area
Q3 source-target MAD:         < 2/255 before video encoding
                              < 6/255 after MP4 roundtrip
```

For authoritative training, keep `source_frames.npy` and `target_frames.npy` so codec noise does not corrupt Q3 consistency.

---

# 8. Object side packet format

Every self-insertion composite should save:

```text
object_side_packet.npz
```

Recommended fields:

```python
{
  "rgba_crop_tube":        uint8 [T,h,w,4],
  "full_mask":             bool  [T,H,W],
  "bbox_xyxy":             int32 [T,4],
  "affine_2d":             float32 [T,3,3],
  "depth_order":           int32 [T],
  "shadow_mask":           uint8 [T,H,W],
  "effect_mask":           uint8 [T,H,W],
  "object_caption":        str, e.g. "a red ball",
  "trajectory_type":       str, e.g. "static", "linear", "bounce",
  "source_asset_id":       str,
  "seed":                  int
}
```

This matters for later cyclic training. The original E2W proposal’s scaling recipe is add → remove → reconstruct, with a gated control branch and preservation loss; an explicit side packet prevents the model from needing to hide object identity in the removed video. 

---

# 9. Manifest records for self-insertion

Each composite emits two manifest records.

## Remove record

```json
{
  "sample_id": "si_000001_remove",
  "dataset_source": "self_insertion",
  "cycle_role": "self_insertion_remove",
  "src_video": "e2w/videos/si_000001_remove/src_video.mp4",
  "target_video": "e2w/videos/si_000001_remove/target_video.mp4",
  "source_frames_npy": "e2w/videos/si_000001_remove/source_frames.npy",
  "target_frames_npy": "e2w/videos/si_000001_remove/target_frames.npy",
  "quadmask_npy": "e2w/masks/si_000001/quadmask.npy",
  "generation_mask_npy": "e2w/masks/si_000001/generation_mask.npy",
  "operation": "remove",
  "inverse_operation": "add",
  "object_side_packet": "e2w/side_packets/si_000001/object_side_packet.npz",
  "prompt": "A clean wooden tabletop with a white cup in the background.",
  "prompt_target_terms_forbidden": ["red ball"],
  "alignment": {
    "method": "none",
    "source_shape": [81, 480, 832],
    "target_shape": [81, 480, 832],
    "quadmask_shape": [81, 480, 832],
    "frame_mapping": "identity"
  },
  "quadmask_stats": {
    "q0_area_frac_mean": 0.035,
    "q1_area_frac_mean": 0.004,
    "q2_area_frac_mean": 0.021,
    "q3_area_frac_mean": 0.940,
    "q1_source": "boundary_proxy_for_control_bootstrap"
  }
}
```

For remove prompts, describe the target clean video. Do not describe the object being removed. VOID’s repo uses the same rule for its `prompt.json`: describe the scene after removal and do not describe the removed object. ([GitHub][2])

## Add record

```json
{
  "sample_id": "si_000001_add",
  "dataset_source": "self_insertion",
  "cycle_role": "self_insertion_add",
  "src_video": "e2w/videos/si_000001_add/src_video.mp4",
  "target_video": "e2w/videos/si_000001_add/target_video.mp4",
  "source_frames_npy": "e2w/videos/si_000001_add/source_frames.npy",
  "target_frames_npy": "e2w/videos/si_000001_add/target_frames.npy",
  "quadmask_npy": "e2w/masks/si_000001/quadmask.npy",
  "generation_mask_npy": "e2w/masks/si_000001/generation_mask.npy",
  "operation": "add",
  "inverse_operation": "remove",
  "object_side_packet": "e2w/side_packets/si_000001/object_side_packet.npz",
  "prompt": "A clean wooden tabletop with a red ball resting near the center and a soft contact shadow.",
  "alignment": {
    "method": "none",
    "source_shape": [81, 480, 832],
    "target_shape": [81, 480, 832],
    "quadmask_shape": [81, 480, 832],
    "frame_mapping": "identity"
  }
}
```

For add prompts, mentioning the object is expected. The object is the desired edited content.

---

# 10. Kubric generation plan

Use the VOID Kubric pipeline rather than writing your own Kubric generator first. The current VOID repo includes `data_generation/`, and the README says its Kubric path generates counterfactual videos with Google Scanned Objects where objects are launched at a target and removal alters target physics; it also documents a command:

```bash
python kubric_variable_objects.py --num_pairs 200 --resolution 384
```

and output format:

```text
training_data/
└── sequence_name/
    ├── rgb_full.mp4       # input with object
    ├── rgb_removed.mp4    # target with object removed and physics applied
    ├── mask.mp4           # quadmask values 0/63/127/255
    └── metadata.json
```

([GitHub][2])

Run a tiny first batch:

```bash
VOID_ROOT=/data/cwx/Edit2World-unified/external/void-model
ANCHOR_ROOT=/data/cwx/E2W/data/phase1_v04_anchors_$(date -u +%Y%m%dT%H%M%SZ)

cd "$VOID_ROOT/data_generation"
pip install kubric pybullet imageio imageio-ffmpeg

python kubric_variable_objects.py \
  --num_pairs 16 \
  --resolution 384
```

Then convert to E2W:

```bash
cd /home/cwx/E2W/.worktree/feat/phase1-v04-control-branch

python tools/convert_void_kubric_to_e2w_anchors.py \
  --void-training-data "$VOID_ROOT/training_data" \
  --out "$ANCHOR_ROOT/e2w" \
  --frames 81 \
  --height 480 \
  --width 832 \
  --write-remove \
  --write-add-reverse \
  --snap-mask-values \
  --save-quadmask-npy \
  --save-generation-mask-npy
```

If `convert_void_kubric_to_e2w_anchors.py` does not exist yet, implement it as the first converter. Its responsibilities are narrow:

```text
1. Read rgb_full.mp4.
2. Read rgb_removed.mp4.
3. Read mask.mp4 or, preferably, mask.npy if you patch VOID to save it.
4. Snap/validate mask values to {0,63,127,255}.
5. Resize videos to E2W target size.
6. Resize masks with nearest only.
7. Subsample/pad/crop to T=81, preserving 4n+1.
8. Save quadmask.npy.
9. Save generation_mask.npy = 255 where Q != 255.
10. Emit remove manifest:
      src=rgb_full, target=rgb_removed, op=remove.
11. Emit add reverse manifest:
      src=rgb_removed, target=rgb_full, op=add.
12. Extract object_side_packet from rgb_full + primary mask.
13. Run E2W v0.4 audit.
```

Patch VOID’s Kubric script if possible so the mask is saved before MP4 encoding:

```python
np.save(out_dir / "quadmask.npy", quadmask.astype(np.uint8))
```

Use `mask.mp4` only as preview unless you have to recover from it.

---

# 11. Kubric quadmask conversion

For Kubric, derive or validate masks as:

```text
P = primary object mask
A = affected non-target mask
```

If using VOID’s `mask.mp4` or `quadmask.npy`, it already encodes:

```text
0   = primary object
63  = primary + affected overlap
127 = affected region
255 = background / keep
```

The VOID README documents the same four values. ([GitHub][2])

If you need to reconstruct from simulation annotations, use:

```python
P = mask_of_removed_target_object_in_full_video

A_visual = abs(rgb_full - rgb_removed) > threshold
A_visual = A_visual & ~P

A_object = union of non-target object masks whose:
  - center trajectory differs beyond threshold, or
  - orientation differs beyond threshold, or
  - visibility/occlusion differs beyond threshold

A = gridify(dilate(A_visual | A_object, radius=4), cell_size=32)

Q = build_quadmask(P, A)
```

For add-reverse records, use the same Q spatially. It describes where the object and affected changes should be generated in the add target.

---

# 12. Audit gates

Before a sample enters `train_all.jsonl`, enforce:

```text
shape:
  src, target, quadmask all have [T,H,W] alignment
  T must be 4n+1, preferably 81

values:
  unique(quadmask) subset of {0,63,127,255}
  validate before uint8 cast

region coverage:
  Q0 nonempty
  Q3 nonempty
  Q3 area >= 50%
  Q0 not too tiny or too large
  Q2 nonempty for physics/effect samples
  Q1 nonempty for overlap-control samples

generation mask:
  values subset of {0,255}
  G = 255 where Q != 255

Q3 consistency:
  source and target should be nearly identical on Q3
  self-insertion should be especially strict

prompt:
  remove prompt must not mention the removed object
  add prompt may mention the object to be added

metadata:
  operation present
  inverse_operation present
  cycle_role present
  object_side_packet present for add or future add-back
  explicit alignment metadata present if resizing/subsampling occurred
```

The current branch already has an anchor manifest audit layer that expects fields like `src_video`, `target_video`, `quadmask_npy`, `generation_mask`, `operation`, `inverse_operation`, `object_side_packet`, `cycle_role`, prompt gates, frame alignment, Q3 consistency, and Physics-IQ rejection. 

Add this quick sanity command:

```bash
python - <<'PY'
import json, numpy as np, sys
manifest = sys.argv[1]
allowed = {0, 63, 127, 255}

for line in open(manifest):
    r = json.loads(line)
    q = np.load(r["quadmask_npy"])
    vals = set(np.unique(q).tolist())
    if not vals <= allowed:
        raise SystemExit(f"{r['sample_id']}: bad values {vals}")
    if q.ndim != 3:
        raise SystemExit(f"{r['sample_id']}: bad shape {q.shape}")
    print(r["sample_id"], q.shape, {v: int((q == v).sum()) for v in sorted(vals)})
PY "$ANCHOR_ROOT/e2w/manifests/train_all.jsonl"
```

---

# 13. First executable milestone

## Step 1 — Create anchor root

```bash
E2W=/home/cwx/E2W/.worktree/feat/phase1-v04-control-branch
ANCHOR_ROOT=/data/cwx/E2W/data/phase1_v04_anchors_$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$ANCHOR_ROOT"/{raw,kubric,self_insert,e2w,reports}
```

## Step 2 — Generate Kubric anchors

```bash
VOID_ROOT=/data/cwx/Edit2World-unified/external/void-model

cd "$VOID_ROOT/data_generation"
pip install kubric pybullet imageio imageio-ffmpeg

python kubric_variable_objects.py \
  --num_pairs 16 \
  --resolution 384
```

## Step 3 — Convert Kubric to E2W

```bash
cd "$E2W"

python tools/convert_void_kubric_to_e2w_anchors.py \
  --void-training-data "$VOID_ROOT/training_data" \
  --out "$ANCHOR_ROOT/e2w" \
  --frames 81 \
  --height 480 \
  --width 832 \
  --write-remove \
  --write-add-reverse \
  --snap-mask-values \
  --save-quadmask-npy \
  --save-generation-mask-npy
```

Expected result:

```text
16 Kubric pairs -> 32 E2W manifest records
```

## Step 4 — Generate self-insertion anchors

```bash
cd "$E2W"

python tools/generate_self_insertion_anchors.py \
  --background-root /data/cwx/E2W/data/anchor_backgrounds_clean \
  --object-bank /data/cwx/E2W/data/object_tubes_rgba \
  --out "$ANCHOR_ROOT/raw/self_insert" \
  --num-composites 8 \
  --frames 81 \
  --height 480 \
  --width 832 \
  --fps 12 \
  --types static,moving,blocker \
  --seed 13 \
  --save-side-packet \
  --save-npy-frames
```

Expected result:

```text
8 self-insertion composites -> 16 E2W manifest records
```

## Step 5 — Merge manifests

```bash
cat \
  "$ANCHOR_ROOT/e2w/manifests/kubric_train.jsonl" \
  "$ANCHOR_ROOT/e2w/manifests/self_insert_train.jsonl" \
  > "$ANCHOR_ROOT/e2w/manifests/train_all.jsonl"

python tools/make_overfit_split.py \
  --manifest "$ANCHOR_ROOT/e2w/manifests/train_all.jsonl" \
  --out "$ANCHOR_ROOT/e2w/manifests/overfit_32.jsonl" \
  --num-records 32 \
  --balance-operation \
  --balance-source kubric,self_insertion \
  --seed 13
```

## Step 6 — Audit

```bash
python tools/e2w_v04_anchor_manifest.py \
  --manifest "$ANCHOR_ROOT/e2w/manifests/overfit_32.jsonl" \
  --strict \
  --write-report "$ANCHOR_ROOT/reports/audit_summary.json"
```

If `e2w_v04_anchor_manifest.py` does not currently expose a CLI, add a thin CLI wrapper rather than duplicating audit logic.

---

# 14. First overfit set composition

Use this for `overfit_32.jsonl`:

```text
Kubric:
  8 pairs
  8 remove records
  8 add-reverse records

Self-insertion:
  8 composites
  8 remove records
  8 add records

Total:
  32 records
  16 add
  16 remove
```

Within the 8 self-insertion composites:

```text
4 SI-A static object + shadow/contact
2 SI-B moving object + moving shadow/contact
2 SI-C blocker + affected secondary object
```

Within the 8 Kubric pairs, prefer:

```text
2 collision-prevention cases
2 support/fall cases
2 obstacle/path cases
2 mixed/distractor cases
```

This gives you:

```text
Q0 object supervision
Q1 overlap supervision
Q2 affected-region supervision
Q3 preservation supervision
add/remove operation balance
real-background preservation examples
true physics examples
```

---

# 15. What counts as “good enough” anchor data?

Before training, each accepted sample should have:

```text
1. exact quadmask values:
   unique(Q) subset of {0,63,127,255}

2. valid frame count:
   T = 81 or another 4n+1 count

3. explicit alignment:
   no silent resize or frame mismatch

4. visible Q0:
   primary region actually corresponds to object/insertion region

5. meaningful Q2:
   shadow/contact/effected object/path region exists where expected

6. Q3 preservation:
   source and target are nearly identical outside editable regions

7. operation-pair symmetry:
   each composite/pair has both add and remove records

8. side packet:
   add records have object crop/mask/bbox/trajectory metadata

9. prompt gate:
   remove prompt does not leak the removed object identity

10. preview:
   contact sheet visually confirms src, target, Q, G
```

Do not start the trainer from a dataset that passes only shape/value checks. The failure mode would be training a formally valid but semantically wrong quadmask branch.

---

# 16. Minimal scripts to write

Write exactly these, in this order:

```text
1. tools/convert_void_kubric_to_e2w_anchors.py
   Purpose:
     fastest way to get true physics anchor data.

2. tools/generate_self_insertion_anchors.py
   Purpose:
     real-background add/remove-balanced anchor data.

3. tools/make_overfit_split.py
   Purpose:
     deterministic 32-record balanced split.

4. tools/render_anchor_contact_sheet.py
   Purpose:
     visual QC: src | target | quadmask | generation mask.

5. thin CLI for tools/e2w_v04_anchor_manifest.py if missing
   Purpose:
     enforce the existing audit layer from shell.
```

Do **not** write a new Kubric simulator before wrapping VOID’s generator. The VOID repo already exposes a Kubric generator and a training-data format with `rgb_full.mp4`, `rgb_removed.mp4`, `mask.mp4`, and `metadata.json`; use that first. ([GitHub][2])

---

# 17. Bottom line

For anchor data, the executable path is:

```text
1. Use VOID/Kubric immediately for real counterfactual physics pairs.
2. Convert each Kubric pair into both remove and add-reverse E2W records.
3. Build self-insertion not as “add-only,” but as paired add/remove records.
4. For self-insertion, emit Mo, Ma, Q, G, side packet, and manifest.
5. Keep quadmask.npy authoritative; previews are secondary.
6. Start with 32 balanced records and overfit VACE-1.3B control branch.
```

This gives you enough clean anchor data to test the thing that matters next: whether the v0.4 gated control branch can actually learn semantic quadmask control, rather than merely passing the v0.3 interface contract.

[1]: https://github.com/google-research/kubric "GitHub - google-research/kubric: A data generation pipeline for creating semi-realistic synthetic multi-object videos with rich annotations such as instance segmentation masks, depth maps, and optical flow. · GitHub"
[2]: https://github.com/netflix/void-model "GitHub - Netflix/void-model · GitHub"
