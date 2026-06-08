# E2W v0.3 VACE Quadmask Training Handoff

Date: 2026-06-02 UTC

Audience: external consultant / technical reviewer for training VACE to understand E2W quadmask control.

## 1. Objective

Train Wan-VACE so E2W can perform physically plausible video edits using a semantic quadmask, not only a binary inpainting/generation mask.

The target behavior is:

- preserve keep regions;
- remove or add the requested primary object;
- update affected non-target regions when physics changes;
- keep prompt text neutral/target-free for VACE continuity generation;
- keep interface success, learned visual control, and human visual success reported separately.

## 2. Current implementation status

E2W now has a v0.3 runtime path where Wan-VACE receives quadmask input at model-forward time.

Key files:

- `docs/CONTRACT.md`
- `tools/run_vace_v03_quad_experiment.py`
- `tools/run_wan_vace_quad_i2v.py`
- `tools/e2w_vace_quad_i2v.py`
- `tests/test_v03_quad_vace_contracts.py`

External VACE checkout:

- `/data/cwx/Edit2World-unified/external/VACE`

External VACE local modification:

- `/data/cwx/Edit2World-unified/external/VACE/vace/models/wan/wan_vace.py`
- Adds optional `E2W_VACE_LOW_MEM=1` VAE/model offload support.
- Quadmask semantics are installed dynamically from E2W helper code, not baked into upstream VACE.

Static validation passed:

```bash
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest \
  tests/test_v02_contracts.py \
  tests/test_v03_quad_vace_contracts.py
```

Result: 14 tests OK.

No claim is made here that visual quadmask control is learned. The new channels are initialized to preserve legacy behavior until fine-tuned.

## 3. v0.3 runtime contract

### 3.1 Required VACE inputs

The v0.3 direct wrapper prepares a command containing:

- `--src_video`: conditioning video;
- `--generation_mask`: binary known/generate mask;
- `--quadmask_npy`: semantic E2W quadmask;
- `--operation remove|add`;
- `--prompt`: descriptive prompt;
- `--frame_num`: Wan-compatible frame count, `4n+1`.

Metadata distinguishes:

- `quadmask_passed_to_backend_command`: command contains `--quadmask_npy`, `--operation`, and `--generation_mask`.
- `quadmask_consumed_by_backend`: subprocess actually ran, returned `0`, and produced `edited_video.mp4`.

Prepared runs must not claim backend consumption.

### 3.2 Quadmask values

`quadmask.npy` shape: `[T,H,W]`.

Allowed values:

| value | meaning |
|---:|---|
| `0` | primary target region |
| `63` | primary and affected overlap |
| `127` | affected non-target region |
| `255` | keep / unchanged region |

Validation rejects invalid raw values before `uint8` casting.

### 3.3 Binary generation mask

Wan-VACE still receives a binary generation mask:

| value | meaning |
|---:|---|
| `0` | keep/known source pixels |
| `255` | generate pixels |

Supported wrapper modes:

- `quadmask-editable`: generate where `quadmask != 255`.
- `future-full-frame`: keep frame 0, generate future full frames.

The binary mask is only a known/generate gate. The semantic edit meaning is carried by `quadmask.npy`.

### 3.4 Frame/shape alignment

Silent mismatch is not allowed. If quadmask shape differs from video shape, the wrapper must be called with explicit alignment, currently:

- `--align-quadmask nearest`

The metadata records source shape, target shape, method, and deterministic frame mapping.

## 4. Current quadmask model-forward mechanism

Current E2W v0.3 implementation expands Wan-VACE context input channels:

- legacy context dim: `96`
- E2W quad context dim: `416`

Mechanism:

1. legacy `vace_patch_embedding` is replaced with a wider Conv3D;
2. first 96 channels copy pretrained legacy weights;
3. new channels are zero-initialized;
4. add/remove operation embedding is zero-initialized;
5. E2W monkey-patches `forward_vace` to include operation embedding;
6. E2W monkey-patches `vace_latent` to concatenate extra quadmask-packed masks.

Additional semantic masks packed into the VACE context:

- editable: `quadmask in {0,63,127}`;
- `Q0`: `quadmask == 0`;
- `Q1`: `quadmask == 63`;
- `Q2`: `quadmask == 127`;
- `Q3`: `quadmask == 255`.

Each extra mask is encoded through the same VACE mask packing path as the binary generation mask. The helper verifies final context channel count is exactly `416`.

Training caveat: because old and new channels currently share one expanded convolution, training only the new channels requires either a gradient mask/parameter split or an adapter-style refactor. Otherwise an optimizer over the full convolution will update legacy 96-channel weights too.

## 5. Direct experiment template

Example add-mug experiment:

```bash
cd /home/cwx/E2W
RUN_DIR=/data/cwx/E2W/runs/e2w_v0_3_quad_vace_add_0076_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_vace_v03_quad_experiment.py \
  --src-video /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/edited_video.mp4 \
  --prompt "Add a yellow mug on the rotating turntable in front of the spotlight" \
  --quadmask-npy /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/quadmask.npy \
  --operation add \
  --run-dir "$RUN_DIR" \
  --generation-mask-mode quadmask-editable \
  --align-quadmask nearest \
  --cuda-visible-devices 4 \
  --sample-steps 8 \
  --run-vace
```

For this sample, explicit alignment is needed because the quadmask has 80 frames while the source video has 81 frames.

## 6. Available data

### 6.1 Local inventory

Under `/data/cwx/E2W/data/physics_iq_vlm_sft`:

- `full_videos/`: 316 videos
- `review_videos_by_seq/`: 316 videos
- `review_proxies_h264_720p/`: 316 videos
- `teacher_labels/parsed/`: 300 parsed teacher labels
- `teacher_labels/raw/`: 300 raw teacher labels
- `vlm_planner_sft_eval_v6_teacher_grounded.jsonl`: 30 eval rows with grounded spatial supervision

Parsed teacher-label stats:

- 300/300 have `quadmask_spec`;
- 300/300 are marked usable for planner SFT;
- target roles: 148 `causal_initiator`, 115 `affected_object`, 39 `distractor`, 1 `unknown`;
- visibility: 284 `clear`, 9 `partial`, 8 `brief`, 2 `unclear`.

The 30-row teacher-grounded eval mix:

- Solid Mechanics: 17
- Fluid Dynamics: 9
- Optics: 2
- Thermodynamics: 2

### 6.2 Data limitation

The current Physics-IQ videos are not paired edited/counterfactual targets.

Manifest caveat:

```json
{
  "purpose": "VLM planner weak-supervision generation only",
  "source_is_counterfactual_target": false,
  "caveat": "Physics-IQ testing videos are real full/testing continuations; do not use them as object-removal target videos for VACE training."
}
```

Use existing Physics-IQ data for:

- planner SFT/evaluation;
- quadmask/spec development;
- reconstruction or spatial-control warmup where target remains the original video;
- smoke/interface tests.

Do not use existing Physics-IQ videos as supervised object-removal targets without verified pairing.

### 6.3 Prompt readiness

Strict target-free VACE prompt status on the 30 teacher-grounded eval rows:

- 14/30 pass;
- 16/30 fail;
- smoke samples `0076`, `0077`, and `0128` fail due to target-contaminated candidate VACE text.

For VACE training/evaluation, prompts must be regenerated or manually fixed before use.

Planner SFT prompt status as of 2026-06-03:

- canonical planner prompt is final-rule-only and generated by
  `tools/e2w_v0_common.py::build_planner_user_prompt`;
- train and eval JSONLs must use the same prompt distribution before LoRA
  training;
- one-shot wrapper prompt experiments are not canonical because the current
  LoRA copied wrapper keys instead of returning top-level planner JSON;
- retraining is required after rewriting train/eval prompts to the final-rule
  contract.


## 11. Consultant questions

1. Is the current 96-to-416 input expansion a good training surface, or should E2W use a separate adapter/gate?
2. If keeping the expanded Conv3D, should legacy 96-channel weights be frozen with a gradient mask?
3. Should operation be trained as a zero-init embedding, encoded only in text, or both?
4. What is the minimal faithful Wan/VACE training loop for new-channel/adaptor training?
5. What synthetic paired data scale is likely needed before adding curated generated data?
6. How should primary and affected regions be weighted in the loss?
7. Is VACE-1.3B enough for proof-of-control before 14B scaling?

## 12. Bottom line

Current status:

- E2W has a working v0.3 interface/model-forward path for quadmask-conditioned Wan-VACE.
- The path expands VACE context from `96` to `416` channels and records command-vs-consumption metadata correctly.
- Static v0.2/v0.3 contract tests pass.
- Learned visual quadmask control is not yet demonstrated.
- Supervised training still needs paired/audited target videos.
- Planner co-training should remain out of scope until VACE quadmask control is validated independently.
