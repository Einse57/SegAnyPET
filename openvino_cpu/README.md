# OpenVINO CPU feasibility (docs + architecture smokes)

**License status:** see root [`BLOCKER.md`](../BLOCKER.md) and [`LICENSE_AUDIT.md`](../LICENSE_AUDIT.md).  
**All benches:** labeled **box-only / non-target** — not Core/Xeon iron.

## Layout

```
openvino_cpu/
  requirements.txt          # high-level pins
  requirements-lock.txt     # pip freeze from the box venv
  scripts/
    feasibility_seganypet.py
    feasibility_nnunet.py     # Yale FM PlainConvUNet architecture (sibling pet-ct-fm)
    bench_box_only.sh         # one-command latency + RSS wrapper
  results/                    # JSON + logs from box runs
  artifacts/                  # local ONNX/IR only (gitignored; not redistributed)
```

## Setup (CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r openvino_cpu/requirements.txt
```

## One-command bench

```bash
./openvino_cpu/scripts/bench_box_only.sh both   # or: nnunet | seganypet
```

Uses **random initialization** only. Does not fetch HF or Dropbox weights.

## Results summary (box, 2026-10-06)

| Path | Params | CPU forward | OV status | OV vs Torch |
| --- | ---: | --- | --- | --- |
| SegAnyPET encoder `vit_b_ori` @ 128³ | 100.5M | ~3.2 s enc | PASS (encoder IR) | cosine ≈ 0.9999 (emb); Dice-on-channel-mean not meaningful |
| Yale nnUNet PlainConvUNet @ 2×96×128×128 | 31.2M | ~2.0 s | PASS (full) | Dice ≈ 0.997 (thresholded logits) |

Peak RSS (box): SegAnyPET path ~5.5 GB; nnUNet path ~3.4 GB (includes OV convert).
