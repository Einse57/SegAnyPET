# Draft PR: OpenVINO CPU feasibility + license BLOCKER (docs-only)

## Summary

Feasibility pass for **SegAnyPET** (primary) and **Yale Foundation-Model-for-PET-CT nnUNet** (fallback) on **CPU / OpenVINO**, with a hard stop on redistribution because **both code and weights lack a clear license**.

This is a **docs-only** draft PR on the Einse57 fork (`openvino-cpu`). No pretrained weights were downloaded or committed. No merge to upstream.

## Licenses

| Asset | License |
| --- | --- |
| SegAnyPET code (GitHub) | **None / unclean** |
| SegAnyPET weights (HF) | **None / unclean** |
| Yale PET/CT FM code | **None / unclean** |
| Yale PET/CT FM weights (Dropbox) | **None / unclean** |

Details: `LICENSE_AUDIT.md`, `BLOCKER.md`.

## Which path “ships”

**Neither pretrained path ships** until licenses clear.  
**Technical preference once cleared:** Yale **nnUNet PlainConvUNet** OpenVINO CPU (full-graph convert, strong OV↔Torch Dice). SegAnyPET 3D-attention **image encoder** also converted successfully on a fixed 128³ crop (contrary to the feared hard OV failure), but the interactive promptable graph is not yet a single IR, and licenses still block weight use.

## DoD checklist

1. **Licenses verified** → both unclean → docs-only BLOCKER (this PR).
2. **Pinned env** → `openvino_cpu/requirements.txt` + `requirements-lock.txt`; CPU torch.
3. **OpenVINO export/compile** → architecture smokes PASS for SegAnyPET encoder + nnUNet full net (random init).
4. **Agreement (Dice) vs reference** → pretrained N/A (no weights). OV vs Torch:
   - nnUNet Dice ≈ **0.996688** (thresholded logits)
   - SegAnyPET encoder cosine ≈ **0.9999057093574116**, MAE ≈ **0.010694156400859356**
5. **One-command bench** → `openvino_cpu/scripts/bench_box_only.sh` (**box-only / non-target**).
6. **Branch `openvino-cpu` + draft PR** → this PR.

## Box-only benches (non-target)

### SegAnyPET `vit_b_ori` (random)
- Params: 100509232
- Encoder forward: 3.17 s
- OV convert: PASS (encoder); infer ~0.78 s (single); agreement latency mean ~0.9411 s
- Peak RSS: 5562.5 MB

### Yale nnUNet PlainConvUNet (random)
- Params: 31195653 (~31.2M matches paper)
- Torch forward: 1.956 s @ shape [1, 2, 96, 128, 128]
- OV latency mean (5 runs): 0.2686 s
- Dice OV vs Torch: 0.996688
- Peak RSS: 3390.4 MB

## Dual-repo note

Fallback sources live in sibling clone `Foundation-Model-for-PET-CT` (`/workspace/pet-ct-fm`). After license clear, prefer forking **that** repo if nnUNet is the production OV path; until then this SegAnyPET fork carries the audit + shared feasibility scripts.

## Blockers

- **Primary:** missing SPDX / LICENSE on SegAnyPET code + HF weights.
- **Fallback equally blocked:** missing SPDX / LICENSE on Yale FM code + Dropbox weights.
- Pretrained Dice / clinical fidelity **not** measured (by policy under unclean license).
- Full SegAnyPET interactive OV graph (prompt encoder + mask decoder + multi-click loop) not packaged.
