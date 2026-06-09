# CLAUDE.md

Claude Code should treat `AGENTS.md` as the canonical agent instruction file for this repository.

Before E2W work, read:

```bash
sed -n '1,260p' AGENTS.md
sed -n '1,260p' docs/E2W_SPEC.md
sed -n '1,220p' docs/E2W_PROJECT_LEDGER.md
```

If current operational status matters, also read:

```bash
sed -n '1,220p' STATUS.md
```

## Critical Rules

- `docs/E2W_SPEC.md` is the single current runtime/spec source.
- `docs/archive/` contains historical references only.
- Current VACE runtime inputs are exactly:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

- Do not use `src_video`, `source_video`, `original_video`, or `factual_source_video` as VACE runtime inputs.
- `vace_conditioning_video` is first-frame-edited conditioning video and is the only visual condition passed to VACE.
- `generation_mask` is unified full-domain generation and carries no E2W semantic edit meaning.
- `quadmask_npy` is the only semantic region contract.
- Text input is named only `vace_prompt`.
- Training manifests, target videos, side packets, and cycle roles are not defined by the current runtime spec.
- Interface success is not visual/control/research success.
- Do not weaken tests, schema, validators, or contracts just to make a run pass.
- Do not commit generated media, `.npy`, checkpoints, model weights, caches, or run directories.

Use the Python environment:

```text
/data/cwx/conda/envs/edit2world-phase1-real/bin/python
```

For code or pipeline-contract changes, run relevant static tests:

```bash
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v03_quad_vace_contracts.py
```

For doc-only changes, verify references and `git diff`; tests are not required unless executable contracts changed.
