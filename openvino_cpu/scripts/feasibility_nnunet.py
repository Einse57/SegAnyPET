#!/usr/bin/env python3
"""Yale PET/CT Foundation Model nnUNetv2 PlainConvUNet CPU smoke + OpenVINO (random init).
Box-only / non-target. No Dropbox weight download (license unclean).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
ART = Path(__file__).resolve().parents[1] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

report = {
    "path": "Foundation-Model-for-PET-CT / nnUNetV2-3d_fullres PlainConvUNet",
    "device": "cpu",
    "weights": "random_init (license-blocked: no Dropbox download/redistribution)",
    "input_shape": [1, 2, 96, 128, 128],  # PET+CT crop from paper/README
    "box_only": True,
    "non_target": True,
}


def peak_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def build_plain_conv_unet(in_channels=2, num_classes=1):
    conv_kernel_sizes = [[3, 3, 3]] * 6
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    features_per_stage = [32, 64, 128, 256, 320, 320]
    n_conv_per_stage = [2, 2, 2, 2, 2, 2]
    n_conv_per_stage_decoder = [2, 2, 2, 2, 2]
    return PlainConvUNet(
        input_channels=in_channels,
        n_stages=6,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=conv_kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


def main():
    torch.manual_seed(0)
    print("Building PlainConvUNet (nnUNetv2 3d_fullres heuristic)...")
    t0 = time.time()
    model = build_plain_conv_unet(in_channels=2, num_classes=1).eval()
    build_s = time.time() - t0
    nparams = sum(p.numel() for p in model.parameters())
    report["build_s"] = round(build_s, 2)
    report["nparams"] = int(nparams)
    report["rss_after_build_mb"] = round(peak_rss_mb(), 1)
    print(f"params={nparams/1e6:.1f}M (paper ~31.2M) build={build_s:.1f}s")

    x = torch.randn(1, 2, 96, 128, 128)
    print("CPU forward...")
    # warmup
    with torch.no_grad():
        _ = model(x)
    t0 = time.time()
    with torch.no_grad():
        y = model(x)
    fwd_s = time.time() - t0
    report["forward_s"] = round(fwd_s, 3)
    report["output_shape"] = list(y.shape)
    report["rss_after_forward_mb"] = round(peak_rss_mb(), 1)
    report["cpu_smoke"] = "PASS"
    print(f"out={tuple(y.shape)} t={fwd_s:.3f}s rss={report['rss_after_forward_mb']}MB")

    # Reference vs itself for Dice pipeline smoke (identity)
    # Create a deterministic "reference" by running twice with same seed input
    with torch.no_grad():
        y2 = model(x)
    # Soft Dice on thresholded logits
    a = (y > 0).float().numpy().ravel()
    b = (y2 > 0).float().numpy().ravel()
    inter = float((a * b).sum())
    denom = float(a.sum() + b.sum())
    dice = (2 * inter / denom) if denom > 0 else 1.0
    report["dice_vs_reference_same_run"] = round(dice, 6)
    report["dice_note"] = "identity check (same random weights, two forwards); not pretrained fidelity"

    report["openvino"] = {}
    onnx_path = ART / "nnunet_plainconvunet.onnx"
    print("ONNX export...")
    try:
        t0 = time.time()
        torch.onnx.export(
            model,
            x,
            str(onnx_path),
            input_names=["pet_ct"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
        report["openvino"]["onnx_export"] = "PASS"
        report["openvino"]["onnx_export_s"] = round(time.time() - t0, 2)
        report["openvino"]["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 1)
        print(f"ONNX OK {report['openvino']['onnx_size_mb']}MB")
    except Exception as e:
        report["openvino"]["onnx_export"] = "FAIL"
        report["openvino"]["onnx_error"] = f"{type(e).__name__}: {e}"
        report["openvino"]["onnx_traceback"] = traceback.format_exc()[-4000:]
        print("ONNX FAIL:", e)

    if report["openvino"].get("onnx_export") == "PASS":
        print("OpenVINO convert + infer...")
        try:
            import openvino as ov
            t0 = time.time()
            ov_model = ov.convert_model(str(onnx_path))
            xml_path = ART / "nnunet_plainconvunet.xml"
            ov.save_model(ov_model, str(xml_path))
            report["openvino"]["convert"] = "PASS"
            report["openvino"]["convert_s"] = round(time.time() - t0, 2)
            core = ov.Core()
            compiled = core.compile_model(ov_model, "CPU")
            # latency bench (box-only)
            xin = x.numpy()
            # warmup
            compiled([xin])
            times = []
            for _ in range(5):
                t0 = time.time()
                out = compiled([xin])
                times.append(time.time() - t0)
            ov_out = list(out.values())[0]
            report["openvino"]["latency_s_mean"] = round(float(np.mean(times)), 4)
            report["openvino"]["latency_s_std"] = round(float(np.std(times)), 4)
            report["openvino"]["latency_s_min"] = round(float(np.min(times)), 4)
            report["openvino"]["n_runs"] = 5
            # Dice OV vs torch
            t_bin = (y.numpy() > 0).astype(np.float32).ravel()
            o_bin = (ov_out > 0).astype(np.float32).ravel()
            inter = float((t_bin * o_bin).sum())
            denom = float(t_bin.sum() + o_bin.sum())
            dice_ov = (2 * inter / denom) if denom > 0 else 1.0
            # Also correlation on logits
            flat_t = y.numpy().ravel().astype(np.float64)
            flat_o = ov_out.ravel().astype(np.float64)
            # numerical agreement
            max_abs = float(np.max(np.abs(flat_t - flat_o)))
            mae = float(np.mean(np.abs(flat_t - flat_o)))
            report["openvino"]["dice_vs_torch"] = round(dice_ov, 6)
            report["openvino"]["max_abs_diff"] = max_abs
            report["openvino"]["mae"] = mae
            report["openvino"]["status"] = "PASS"
            report["dice_ov_vs_torch"] = round(dice_ov, 6)
            print(f"OV OK latency_mean={report['openvino']['latency_s_mean']}s dice={dice_ov:.6f} mae={mae:.3e}")
        except Exception as e:
            report["openvino"]["status"] = "FAIL"
            report["openvino"]["ov_error"] = f"{type(e).__name__}: {e}"
            report["openvino"]["ov_traceback"] = traceback.format_exc()[-4000:]
            print("OV FAIL:", e)
    else:
        report["openvino"]["status"] = "FAIL_ONNX"

    report["rss_peak_mb"] = round(peak_rss_mb(), 1)
    out_json = OUT / "feasibility_nnunet.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("Wrote", out_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
