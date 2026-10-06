# BLOCKER: SegAnyPET / Yale PET-CT FM — license unclean (docs-only)

**Status:** Docs-only draft. No pretrained weights downloaded or redistributed.  
**Date:** 2026-10-06 (PT)  
**Actor:** Einse57 fork work under `/workspace/seganypet` (+ `/workspace/pet-ct-fm` for fallback inspection).  
**Hardware note:** All numbers below are **box-only / non-target** (no Core/Xeon iron).

## Hard blocker

| Asset | Source | License found |
| --- | --- | --- |
| SegAnyPET **code** | https://github.com/YichiZhang98/SegAnyPET | **None** (`license: null` via GitHub API; no `LICENSE` file) |
| SegAnyPET **weights** | https://huggingface.co/YichiZhang98/SegAnyPET (`seganypet_v1/v2/lesion.pth`, ~1.21 GB each) | **None** (no model card `license`; HF API `cardData: null`) |
| Yale PET/CT FM **code** | https://github.com/liu-xiaofeng/Foundation-Model-for-PET-CT | **None** (`license: null`; no `LICENSE` file) |
| Yale PET/CT FM **weights** | Dropbox folder linked from README (`nnunet_v2_mim_best.pth` ~31.2M, Swin variants) | **None** stated |

**Related (not a grant for these repos):** upstream SAM / SAM-Med3D / nnU-Net are Apache-2.0, and SegAnyPET source files retain Meta “see LICENSE in root” headers — but **this repository does not ship that LICENSE**, and authors have not published a clear grant for SegAnyPET code or HF weights. Yale FM similarly says “open-source” in the paper abstract without an SPDX license on the GitHub repo or Dropbox checkpoints.

**Policy outcome (DoD §1):** both primary and fallback are **license-blocked**. Prefer fallback only if cleaner — it is **not**. Therefore this PR is **docs-only**: feasibility scaffolding + audit + box-only architecture smokes with **random init**, no weight redistribution, no claim of redistributable OpenVINO artifacts for production.

## What would unblock

1. Explicit SPDX license on SegAnyPET GitHub + Hugging Face model card covering code **and** checkpoints (or written permission).  
2. Explicit SPDX license on Foundation-Model-for-PET-CT + Dropbox weights (or written permission).  
3. Until then: **do not** download, convert, or ship pretrained `.pth` / OV IR derived from those weights in this fork.

## Path that would ship after license clear (technical)

Architecture smokes (random weights) show:

| Path | CPU smoke | OpenVINO | Notes |
| --- | --- | --- | --- |
| **SegAnyPET** `vit_b_ori` (100.5M) | PASS (encoder + prompt/decoder) | **Encoder ONNX→OV PASS** | 3D window/global attention **did convert** for fixed 128³ crop; tracer warnings on Python bools in window/rel-pos. Full interactive graph (prompt encoder + mask decoder) not packaged as single IR yet. |
| **Yale nnUNet** PlainConvUNet (~31.2M) | PASS | **PASS** (full net) | Preferred **technical** ship target for OV CPU once licensed: pure Conv3d, Dice(OV vs Torch) ≈ **0.997** on random logits. |

**Recommendation after license clear:** ship **Yale nnUNet OV CPU** as the reliable OpenVINO path; keep SegAnyPET encoder OV as optional promptable path (CUDA ≥11.7 / ~4 GB GPU in upstream README).

## Pretrained Dice / fidelity

**Not measured.** Weights were not downloaded (license). Reported Dice values are OV-vs-Torch numerical agreement on **random-init** outputs only (see `openvino_cpu/results/`).

## Box-only benches (non-target)

See `openvino_cpu/results/feasibility_*.json` and `PR_DESCRIPTION.md`.

One-command (after venv):  
`openvino_cpu/scripts/bench_box_only.sh both`

## Fork / PR intent

- Fork: Einse57/SegAnyPET (from YichiZhang98/SegAnyPET)  
- Branch: `openvino-cpu`  
- Draft PR only — **never merge / never push upstream**  
- Dual-repo note: fallback inspected at sibling `/workspace/pet-ct-fm`; prefer forking Yale FM **after** its license is cleared if nnUNet is the ship path.
