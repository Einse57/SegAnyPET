# License audit — SegAnyPET + Yale PET/CT Foundation Model

Checked: 2026-10-06 (America/Los_Angeles).

## SegAnyPET (primary)

| Check | Result |
| --- | --- |
| GitHub `license` field | `null` |
| Root `LICENSE` / `COPYING` / `NOTICE` | **Absent** |
| README license section | **Absent** |
| Hugging Face model card / `cardData.license` | **Absent** / `null` |
| File headers | Many `segment_anything/*` files: Meta copyright + “licensed under LICENSE in root” — **root LICENSE missing** |
| Provenance | README: adapted from [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) (Apache-2.0); original SAM Apache-2.0. **Does not auto-license SegAnyPET authors’ modifications or checkpoints.** |
| Weights | `seganypet_v1.pth`, `seganypet_v2.pth`, `seganypet_lesion.pth` on HF — pickle/`torch` checkpoints, no license tag |
| Third-party commentary | bio.rodeo: “The released checkpoints carry no stated license.” |

**Verdict:** **UNCLEAN / BLOCKED** for redistribution and for deriving redistributable OpenVINO IR from pretrained weights.

## Foundation-Model-for-PET-CT (fallback)

| Check | Result |
| --- | --- |
| GitHub `license` field | `null` |
| Root `LICENSE` | **Absent** |
| README license section | **Absent** |
| Paper language | Describes work as open-source; **not** an SPDX grant on the repo/weights |
| Weights | Dropbox external hosting; no license text observed in README |
| Stack deps | Uses `dynamic-network-architectures` / nnU-Net v2 PlainConvUNet pattern (nnU-Net itself Apache-2.0) — again, does not license this repo’s code/weights |

**Verdict:** **UNCLEAN / BLOCKED** — **not** cleaner than SegAnyPET.

## Decision tree (DoD)

1. SegAnyPET unclean → consider fallback if cleaner.  
2. Fallback also unclean → **docs-only BLOCKER.md draft PR** (this branch).  
3. No preference between unclean assets for shipping weights/OV IR.
