#!/usr/bin/env python3
"""SegAnyPET architecture CPU smoke + OpenVINO export attempt (random init; no weights redistributed).
Box-only / non-target bench labels apply.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from segment_anything.build_sam3D import sam_model_registry3D

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

report = {
    "path": "SegAnyPET",
    "model": "vit_b_ori (img_size=128)",
    "device": "cpu",
    "weights": "random_init (license-blocked: no HF download/redistribution)",
    "box_only": True,
    "non_target": True,
}


def peak_rss_mb() -> float:
    try:
        import resource
        # Linux: ru_maxrss is KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    print("Building vit_b_ori (random init)...")
    t0 = time.time()
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    model = model.to(device).eval()
    build_s = time.time() - t0
    nparams = sum(p.numel() for p in model.parameters())
    report["build_s"] = round(build_s, 2)
    report["nparams"] = int(nparams)
    report["rss_after_build_mb"] = round(peak_rss_mb(), 1)
    print(f"params={nparams/1e6:.1f}M build={build_s:.1f}s rss={report['rss_after_build_mb']}MB")

    # Synthetic PET crop matching expected encoder input: [B,1,128,128,128]
    x = torch.randn(1, 1, 128, 128, 128, device=device)
    # Minimal prompt tensors for decoder path via image_encoder only first
    print("CPU forward: image_encoder only...")
    t0 = time.time()
    with torch.no_grad():
        emb = model.image_encoder(x)
    enc_s = time.time() - t0
    report["encoder_forward_s"] = round(enc_s, 2)
    report["encoder_out_shape"] = list(emb.shape)
    report["rss_after_encoder_mb"] = round(peak_rss_mb(), 1)
    print(f"encoder out={tuple(emb.shape)} t={enc_s:.2f}s rss={report['rss_after_encoder_mb']}MB")

    # Try full interactive-ish path with dummy points
    print("CPU forward: prompt_encoder + mask_decoder...")
    try:
        points = torch.tensor([[[64.0, 64.0, 64.0]]], device=device)  # B=1,N=1,3
        labels = torch.tensor([[1]], device=device)
        t0 = time.time()
        with torch.no_grad():
            sparse, dense = model.prompt_encoder(
                points=(points, labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou = model.mask_decoder(
                image_embeddings=emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
        dec_s = time.time() - t0
        report["decoder_forward_s"] = round(dec_s, 2)
        report["mask_shape"] = list(low_res_masks.shape)
        report["iou_shape"] = list(iou.shape)
        report["cpu_smoke"] = "PASS"
        print(f"masks={tuple(low_res_masks.shape)} iou={tuple(iou.shape)} t={dec_s:.2f}s")
    except Exception as e:
        report["cpu_smoke"] = "FAIL"
        report["cpu_smoke_error"] = f"{type(e).__name__}: {e}"
        report["cpu_smoke_traceback"] = traceback.format_exc()
        print("Decoder path failed:", e)

    # OpenVINO: export image encoder via ONNX then convert
    report["openvino"] = {}
    onnx_path = ART / "seganypet_image_encoder.onnx"
    print("ONNX export: image_encoder...")
    try:
        t0 = time.time()
        # Prefer dynamo=False for broader compatibility with attention
        torch.onnx.export(
            model.image_encoder,
            x,
            str(onnx_path),
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=17,
            dynamo=False,
        )
        report["openvino"]["onnx_export_s"] = round(time.time() - t0, 2)
        report["openvino"]["onnx_path"] = str(onnx_path)
        report["openvino"]["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 1)
        report["openvino"]["onnx_export"] = "PASS"
        print(f"ONNX OK size={report['openvino']['onnx_size_mb']}MB")
    except Exception as e:
        report["openvino"]["onnx_export"] = "FAIL"
        report["openvino"]["onnx_error"] = f"{type(e).__name__}: {e}"
        report["openvino"]["onnx_traceback"] = traceback.format_exc()[-4000:]
        print("ONNX FAIL:", e)

    if report["openvino"].get("onnx_export") == "PASS":
        print("OpenVINO convert image_encoder...")
        try:
            import openvino as ov
            t0 = time.time()
            ov_model = ov.convert_model(str(onnx_path))
            xml_path = ART / "seganypet_image_encoder.xml"
            ov.save_model(ov_model, str(xml_path))
            report["openvino"]["convert_s"] = round(time.time() - t0, 2)
            report["openvino"]["ir_path"] = str(xml_path)
            report["openvino"]["convert"] = "PASS"
            # Compile + one infer
            core = ov.Core()
            compiled = core.compile_model(ov_model, "CPU")
            t0 = time.time()
            req = compiled.create_infer_request()
            req.infer({0: x.numpy()})
            report["openvino"]["infer_s"] = round(time.time() - t0, 2)
            report["openvino"]["status"] = "PASS"
            print(f"OV convert+infer OK t_infer={report['openvino']['infer_s']}s")
        except Exception as e:
            report["openvino"]["convert"] = "FAIL"
            report["openvino"]["status"] = "FAIL"
            report["openvino"]["ov_error"] = f"{type(e).__name__}: {e}"
            report["openvino"]["ov_traceback"] = traceback.format_exc()[-4000:]
            print("OV FAIL:", e)
    else:
        report["openvino"]["status"] = "FAIL_ONNX"

    report["rss_peak_mb"] = round(peak_rss_mb(), 1)
    out_json = OUT / "feasibility_seganypet.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("Wrote", out_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
